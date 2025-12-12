import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ultralytics.models.sam.modules.memory_attention import MemoryAttention, MemoryAttentionLayer
from ultralytics.models.sam.modules.blocks import RoPEAttention, CXBlock
from ultralytics.models.sam.modules.encoders import MemoryEncoder as SAMMemoryEncoder
from ultralytics.models.sam.modules.utils import compute_axial_cis, apply_rotary_enc

class YOLORoPEAttention(RoPEAttention):
    """
    Subclass of RoPEAttention that handles non-square spatial dimensions correctly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_shape = None  # (h, w)

    def set_spatial_shape(self, h, w):
        """Set the expected spatial shape for the current forward pass."""
        self.spatial_shape = (h, w)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # Separate into heads
        q_heads = self._separate_heads(q_proj, self.num_heads) # (B, nHead, N, C_head)
        k_heads = self._separate_heads(k_proj, self.num_heads)
        v_heads = self._separate_heads(v_proj, self.num_heads)

        # Determine spatial dimensions
        if self.spatial_shape is not None:
            h, w = self.spatial_shape
        else:
            # Fallback to square assumption or 1D
            num_tokens = q_heads.shape[-2]
            side = int(math.sqrt(num_tokens))
            if side * side == num_tokens:
                h, w = side, side
            else:
                h, w = 1, num_tokens

        # Recompute freqs_cis if needed and if it makes sense (grid structure)
        # Note: For Sparse Attention (TopK), K does not have grid structure.
        # But Q usually does. RoPE is applied to Q and K to encode relative position.
        # If K is sparse/unstructured, we might want to skip RoPE for K or use partial RoPE.
        # For now, we assume YOLORoPEAttention is used in Dense setup primarily,
        # or that the caller handles 'num_k_exclude_rope' or passed K with structure.
        
        # Optimization: Only compute CIS if dimensions changed
        if self.freqs_cis is None or self.freqs_cis.shape[0] != q_heads.shape[-2] or \
           (self.spatial_shape is not None and self.freqs_cis.shape[0] == q_heads.shape[-2]): 
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        
        self.freqs_cis = self.freqs_cis.to(q.device)
        
        if q_heads.shape[-2] != k_heads.shape[-2]:
             # If K len != Q len (e.g. Cross Attn with Memory), RoPE requires careful handling.
             # Standard compute_cis generates grid for max(H,W). 
             # If K is memory (longer or shorter), RoPE assumes it's a grid too?
             # Actually SAM2 RoPE implementation expects Q and K to be broadcastable or handled via repeat_freqs_k.
             # If rope_k_repeat is True, K uses the same freqs as Q (which assumes K matches Q spatial structure?)
             # OR it repeats?
             # SAM2 docs: "If rope_k_repeat is True, we repeat the frequencies for K to match".
             pass

        num_k_rope = k_heads.size(-2) - num_k_exclude_rope
        
        # Robust RoPE application
        # If K is sparse (len != grid size), apply_rotary_enc might fail or give nonsense if we force grid freqs.
        # However, for YOLOMemoryAttention (Dense), it works.
        q_enc, k_enc_part = apply_rotary_enc(
            q_heads,
            k_heads[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )
        k_heads[:, :, :num_k_rope] = k_enc_part

        # Attention
        out = F.scaled_dot_product_attention(
            q_enc, 
            k_heads, 
            v_heads,
            dropout_p=0.0,
            is_causal=False
        )

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class YOLOMemoryEncoder(nn.Module):
    """
    Lightweight encoder to compress and process features before storing in memory bank.
    Reduces VRAM usage and adds semantic processing.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.encode = CXBlock(dim=out_dim, kernel_size=3, padding=1, use_dwconv=True)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.encode(x)
        return x

class YOLOMemoryAttentionLayer(MemoryAttentionLayer):
    """
    Subclass of MemoryAttentionLayer that fixes hardcoded dims.
    """
    def __init__(self, d_model=256, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False):
        super().__init__(d_model, dim_feedforward, dropout, pos_enc_at_attn, pos_enc_at_cross_attn_keys, pos_enc_at_cross_attn_queries)
        
        # Self Attn (Dense)
        self.self_attn = YOLORoPEAttention(embedding_dim=d_model, num_heads=1, downsample_rate=1)
        
        # Cross Attn (Dense/Sparse?)
        # If we use this layer for Sparse, we might want to disable RoPE or be careful.
        # But for default behavior, we keep structure.
        self.cross_attn_image = YOLORoPEAttention(
            rope_k_repeat=True,
            embedding_dim=d_model,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=d_model, 
        )

class YOLOMemoryAttention(nn.Module):
    """
    YOLO Memory Attention module wrapping SAM2's MemoryAttention.
    """

    def __init__(self, d_model=None, max_memory=8):
        super().__init__()
        self.requested_dim = d_model
        self.d_model = None
        self.proj_in = None
        self.proj_out = None
        self.attn = None
        self.memory_encoder = None
        self.maskmem_tpos_enc = None  # Temporal positional encoding

        self.memory_bank = []
        self.max_memory = max_memory
        self.time_steps = 1
        
        # Allow subclass to override layer type
        self.layer_cls = YOLOMemoryAttentionLayer

    def _build_layers(self, in_channels):
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        self.memory_encoder = YOLOMemoryEncoder(in_channels, target_dim)

        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=True,
            layer=self.layer_cls(d_model=target_dim),
            num_layers=1,
        )

    def forward(self, x):
        """
        x: (N, C, H, W). If dim=5 (B, T, C, H, W) is handled by caller flattening it usually.
        """
        B_total, C, H, W = x.shape
        
        if self.attn is None or self.d_model is None:
            self._build_layers(x.shape[1])
            
        if self.maskmem_tpos_enc is None:
            self.maskmem_tpos_enc = nn.Parameter(torch.zeros(self.max_memory, 1, 1, self.d_model).to(x.device))
            nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        # Update spatial shape for RoPE
        for layer in self.attn.layers:
            if isinstance(layer.self_attn, YOLORoPEAttention):
                layer.self_attn.set_spatial_shape(H, W)
            if isinstance(layer.cross_attn_image, YOLORoPEAttention):
                layer.cross_attn_image.set_spatial_shape(H, W)

        x_proj = self.proj_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1)  # (N, tokens, d_model)
        
        if self.training:
            # Training mode: Expecting N = Batch * Time
            T = self.time_steps
            B = B_total // T
            x_seq = x_flat.view(B, T, -1, self.d_model) # (B, T, L, D)
            
            # Encode memory frames
            mem_encoded = self.memory_encoder(x) # (N, D, H, W)
            mem_encoded = mem_encoded.view(B, T, self.d_model, H, W)

            out_seq = []
            for t in range(T):
                curr = x_seq[:, t]  # (B, L, D)
                if t == 0:
                    memory = curr 
                else:
                    start = max(0, t - self.max_memory)
                    mem_frames = mem_encoded[:, start:t] # (B, k, D, H, W)
                    k_frames = mem_frames.shape[1]
                    
                    mem_frames_list = []
                    for i in range(k_frames):
                        t_mem = start + i
                        lag = t - t_mem
                        idx = max(0, self.max_memory - lag)
                        t_enc = self.maskmem_tpos_enc[idx] # (1, 1, D)
                        
                        feat = mem_frames[:, i] # (B, D, H, W)
                        feat_flat = feat.flatten(2).permute(0, 2, 1) # (B, L, D)
                        
                        # Add T-Pos Enc
                        # NOTE: For Sparse subclass, we might want to sparsify AFTER this.
                        # Base class assumes dense.
                        feat_flat = feat_flat + t_enc
                        mem_frames_list.append(feat_flat)

                    memory = torch.cat(mem_frames_list, dim=1) # (B, k*L, D)
                    
                    # === HOOK for Sparse Selection ===
                    if hasattr(self, 'process_memory_training'):
                        memory = self.process_memory_training(memory)
                
                res = self.attn(
                    curr=curr, 
                    memory=memory, 
                    curr_pos=torch.zeros_like(curr), 
                    memory_pos=torch.zeros_like(memory)
                )
                out_seq.append(res)
            
            out = torch.stack(out_seq, dim=1)
            out = out.view(B_total, H, W, self.d_model).permute(0, 3, 1, 2)
            
        else:
            # Inference
            curr = x_flat
            
            # === HOOK for Memory Retrieval ===
            memory = self.retrieve_memory_inference(curr)
            
            res = self.attn(
                curr=curr,
                memory=memory,
                curr_pos=torch.zeros_like(curr),
                memory_pos=torch.zeros_like(memory),
            )
            out = res.view(B_total, H, W, self.d_model).permute(0, 3, 1, 2)

            # Update Memory Bank
            with torch.no_grad():
                mem_encoded = self.memory_encoder(x) # (B, D, H, W)
                mem_flat = mem_encoded.flatten(2).permute(0, 2, 1) # (B, L, D)
                
                # === HOOK for Memory Update ===
                if hasattr(self, 'process_memory_update'):
                    mem_flat = self.process_memory_update(mem_flat)
                    
                self.memory_bank.append(mem_flat.detach())
                
            if len(self.memory_bank) > self.max_memory:
                self.memory_bank.pop(0)
                
        return self.proj_out(out) + x 

    def reset_memory(self):
        """Reset memory bank."""
        self.memory_bank = []
        
    def set_memory(self, memory):
        """
        [State Injection]
        Inject external memory state.
        Args:
            memory: List[Tensor] or Tensor representing the memory bank.
        """
        if isinstance(memory, list):
            self.memory_bank = memory
        elif isinstance(memory, torch.Tensor):
            self.memory_bank = [memory]
        elif memory is None:
            self.memory_bank = []
        else:
             # Try to iterate if it's iterable but not list/tensor
             try:
                 self.memory_bank = list(memory)
             except TypeError:
                 self.memory_bank = [memory]

    def get_memory(self):
        """
        [State Extraction]
        Retrieve current memory bank state.
        Returns:
            List[Tensor]: Current memory bank.
        """
        return self.memory_bank

    def retrieve_memory_inference(self, curr):
        """Default inference memory retrieval: concat all history with T-Pos enc"""
        if not self.memory_bank:
            return curr
        
        # Validate bank
        if self.memory_bank[0].shape[-1] != curr.shape[-1] or self.memory_bank[0].shape[0] != curr.shape[0]:
             self.memory_bank = []
             return curr
             
        mem_list = []
        num_mem = len(self.memory_bank)
        for i, m in enumerate(self.memory_bank):
            lag = num_mem - i
            idx = max(0, self.max_memory - lag)
            t_enc = self.maskmem_tpos_enc[idx] 
            
            # Note: memory bank stores (B, L, D) or (B, K, D) if sparse
            m_enc = m.to(curr.device) + t_enc
            mem_list.append(m_enc)
        
        return torch.cat(mem_list, dim=1)

# =========================================================================
# SPARSE IMPLEMENTATION
# =========================================================================

class VanillaCrossAttention(RoPEAttention):
    """
    Wrapper to use plain attention (Key/Value is sparse bag-of-features, no spatial structure)
    """
    def forward(self, q, k, v, num_k_exclude_rope=0):
        # Basic projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        
        # NO RoPE application
        # Since K is sparse/unstructured, relative position is undefined.
        # We rely purely on semantic similarity.
        
        # Attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class SparseMemoryAttention(YOLOMemoryAttention):
    """
    Sparse Memory Attention using Top-K selection.
    Disables Spatial Positional Encoding (RoPE) for Cross Attention to allow
    'Bag of Features' memory.
    """
    def __init__(self, d_model=None, max_memory=8, topk_ratio=0.1):
        super().__init__(d_model, max_memory)
        self.topk_ratio = topk_ratio
        
    def _build_layers(self, in_channels):
        """Build layers but Disable RoPE for keys in cross attention"""
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        self.memory_encoder = YOLOMemoryEncoder(in_channels, target_dim)

        # Custom Layer config for Sparse: No RoPE on Cross Attn Keys
        layer = YOLOMemoryAttentionLayer(d_model=target_dim)
        
        # Wrapper to use plain attention (Key/Value is sparse bag-of-features, no spatial structure)

        
        # Replace
        layer.cross_attn_image = VanillaCrossAttention(
            embedding_dim=target_dim,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=target_dim
        )
        
        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=True,
            layer=layer,
            num_layers=1,
        )

    def select_topk_features(self, features):
        """
        Selects top-k features from (B, L, D) tensor based on L2 norm.
        """
        B, L, D = features.shape
        k = max(1, int(L * self.topk_ratio))
        
        # Score: L2 Norm
        scores = torch.norm(features, dim=-1) # (B, L)
        topk_scores, topk_indices = torch.topk(scores, k, dim=1) # (B, k)
        
        # Gather (B, k, D)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        features_sparse = torch.gather(features, 1, topk_indices_expanded)
        
        return features_sparse

    def process_memory_training(self, memory):
        """
        Called during training loop. Memory is (B, T_hist*L, D).
        We select TopK from the ENTIRE history buffer?
        Or per frame?
        Current implementation concats frames first.
        We can select Top Total K.
        """
        return self.select_topk_features(memory)

    def process_memory_update(self, mem_flat):
        """
        Called during inference update. mem_flat is (B, L, D).
        We only store TopK in the bank.
        """
        return self.select_topk_features(mem_flat)
