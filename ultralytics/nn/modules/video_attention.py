import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ultralytics.models.sam.modules.memory_attention import MemoryAttention, MemoryAttentionLayer
from ultralytics.models.sam.modules.blocks import RoPEAttention, CXBlock
from ultralytics.models.sam.modules.encoders import MemoryEncoder as SAMMemoryEncoder
from ultralytics.models.sam.modules.utils import compute_axial_cis, apply_rotary_enc, compute_global_cis

def apply_global_rotary_enc(xq, xk, freqs_cis_q, freqs_cis_k):
    """
    Apply batched Global RoPE.
    xq: (B, H, L, D)
    freqs_cis_q: (B, L, D/2) (complex)
    """
    # View as complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    
    # Broadcast freqs: (B, L, D/2) -> (B, 1, L, D/2) to match (B, H, L, D/2)
    # Ensure dimensions match
    if freqs_cis_q.ndim == xq_.ndim - 1:
        freqs_q = freqs_cis_q.unsqueeze(1)
    else:
        freqs_q = freqs_cis_q # Hope shapes align or it's unbatched
        
    xq_out = torch.view_as_real(xq_ * freqs_q).flatten(3)
    
    xk_out = None
    if xk is not None and xk.numel() > 0:
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if freqs_cis_k.ndim == xk_.ndim - 1:
            freqs_k = freqs_cis_k.unsqueeze(1)
        else:
            freqs_k = freqs_cis_k
        xk_out = torch.view_as_real(xk_ * freqs_k).flatten(3)
        xk_out = xk_out.type_as(xk)
        
    return xq_out.type_as(xq), xk_out

class GlobalRoPEAttention(RoPEAttention):
    """
    Apply RoPE using explicitly passed Global Coordinates (B, N, 2).
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_coords: torch.Tensor = None, k_coords: torch.Tensor = None, num_k_exclude_rope: int = 0) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # Separate into heads
        q_heads = self._separate_heads(q_proj, self.num_heads)
        k_heads = self._separate_heads(k_proj, self.num_heads)
        v_heads = self._separate_heads(v_proj, self.num_heads)
        
        # Apply Global RoPE if coords provided
        if q_coords is not None and k_coords is not None:
             # q_coords: (B, Nq, 2) -> split x, y
             freqs_cis_q = compute_global_cis(q_coords[..., 0], q_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             
             freqs_cis_k = compute_global_cis(k_coords[..., 0], k_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             
             num_k_rope = k_heads.size(-2) - num_k_exclude_rope
             
             # Use custom apply to handle batched freqs
             q_heads, k_heads_part = apply_global_rotary_enc(
                 q_heads, 
                 k_heads[:, :, :num_k_rope], 
                 freqs_cis_q, 
                 freqs_cis_k
             )
             
             k_heads[:, :, :num_k_rope] = k_heads_part
        
        # Attention
        out = F.scaled_dot_product_attention(
            q_heads, 
            k_heads, 
            v_heads,
            dropout_p=0.0,
            is_causal=False
        )

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class GlobalRoPEMemoryAttentionLayer(MemoryAttentionLayer):
    """
    Subclass that passes coordinates to cross_attn_image and self_attn.
    Use this with GlobalRoPEAttention.
    """
    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        # Hijack query_pos to pass q_coords/k_coords to self_attn
        tgt2 = self.norm1(tgt)
        
        # Self-Attention on 'curr' (tgt2)
        # q=k=v=tgt2
        tgt2 = self.self_attn(
            q=tgt2, 
            k=tgt2, 
            v=tgt2, 
            q_coords=query_pos,
            k_coords=query_pos # Self-attn uses same coords
        )
        
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None, # Hijacked to pass q_coords
        pos: torch.Tensor | None,       # Hijacked to pass k_coords
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        # We assume query_pos and pos contain COORDS, not embeddings.
        # So we do NOT add them to tgt/memory.
        
        tgt2 = self.norm2(tgt)
        
        # Pass coords explicitly
        tgt2 = self.cross_attn_image(
            q=tgt2,
            k=memory,
            v=memory,
            q_coords=query_pos,
            k_coords=pos,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class VanillaCrossAttention(GlobalRoPEAttention):
    """
    Legacy class for backward compatibility with checkpoints.
    Inherits GlobalRoPEAttention to support new coordinate-aware forward usage
    even if loaded from old checkpoints.
    """
    pass

 
class YOLORoPEAttention(RoPEAttention):
    """
    Subclass of RoPEAttention that handles non-square spatial dimensions correctly.
    Also supports Global RoPE if q_coords/k_coords passed (for Sparse Attention).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_shape = None  # (h, w)

    def set_spatial_shape(self, h, w):
        """Set the expected spatial shape for the current forward pass."""
        self.spatial_shape = (h, w)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_coords: torch.Tensor = None, k_coords: torch.Tensor = None, num_k_exclude_rope: int = 0) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # Separate into heads
        q_heads = self._separate_heads(q_proj, self.num_heads) # (B, nHead, N, C_head)
        k_heads = self._separate_heads(k_proj, self.num_heads)
        v_heads = self._separate_heads(v_proj, self.num_heads)
        
        # === GLOBAL ROPE PATH (Sparse) ===
        if q_coords is not None and k_coords is not None:
             freqs_cis_q = compute_global_cis(q_coords[..., 0], q_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             freqs_cis_k = compute_global_cis(k_coords[..., 0], k_coords[..., 1], self.internal_dim // self.num_heads, 10000.0)
             
             num_k_rope = k_heads.size(-2) - num_k_exclude_rope
             
             q_heads, k_heads_part = apply_global_rotary_enc(
                 q_heads, 
                 k_heads[:, :, :num_k_rope], 
                 freqs_cis_q, 
                 freqs_cis_k
             )
             k_heads[:, :, :num_k_rope] = k_heads_part
             
             # Attention
             out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0, is_causal=False)
             out = self._recombine_heads(out)
             out = self.out_proj(out)
             return out

        # === STANDARD ROPE PATH (Dense) ===
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

        # Optimization: Only compute CIS if dimensions changed
        if self.freqs_cis is None or self.freqs_cis.shape[0] != q_heads.shape[-2] or \
           (self.spatial_shape is not None and self.freqs_cis.shape[0] == q_heads.shape[-2]): 
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        
        self.freqs_cis = self.freqs_cis.to(q.device)
        
        if q_heads.shape[-2] != k_heads.shape[-2]:
             pass

        num_k_rope = k_heads.size(-2) - num_k_exclude_rope
        
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
    Now supports coordinate passing for Global RoPE.
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

    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        # Hijack query_pos to pass q_coords/k_coords to self_attn
        tgt2 = self.norm1(tgt)
        
        # Self-Attention on 'curr' (tgt2)
        # q=k=v=tgt2
        # Check if self_attn supports coords (YOLORoPEAttention/GlobalRoPEAttention do)
        tgt2 = self.self_attn(
            q=tgt2, 
            k=tgt2, 
            v=tgt2, 
            q_coords=query_pos,
            k_coords=query_pos # Self-attn uses same coords
        )
        
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None, # Hijacked to pass q_coords
        pos: torch.Tensor | None,       # Hijacked to pass k_coords
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        # We assume query_pos and pos contain COORDS, not embeddings.
        # So we do NOT add them to tgt/memory.
        
        tgt2 = self.norm2(tgt)
        
        # Pass coords explicitly
        tgt2 = self.cross_attn_image(
            q=tgt2,
            k=memory,
            v=memory,
            q_coords=query_pos,
            k_coords=pos,
            num_k_exclude_rope=num_k_exclude_rope,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

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


class SparseMemoryAttention(YOLOMemoryAttention):
    """
    Sparse Memory Attention using Top-K selection with Global RoPE.
    Preserves spatial awareness by tracking coordinates of sparse tokens.
    """
    def __init__(self, d_model=None, max_memory=8, topk_ratio=0.1):
        super().__init__(d_model, max_memory)
        self.topk_ratio = topk_ratio
        
    def _build_layers(self, in_channels):
        """Build layers with GlobalRoPEMemoryAttentionLayer"""
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        self.memory_encoder = YOLOMemoryEncoder(in_channels, target_dim)

        # Custom Layer config for Sparse: Global RoPE
        # We use GlobalRoPEMemoryAttentionLayer which passes 'pos' (coords) to 'cross_attn_image'
        layer = GlobalRoPEMemoryAttentionLayer(
            d_model=target_dim,
            pos_enc_at_attn=False, # We do RoPE inside attention, no additive PE
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False
        )
        
        # Replace cross_attn with GlobalRoPEAttention
        layer.cross_attn_image = GlobalRoPEAttention(
            rope_k_repeat=True,
            embedding_dim=target_dim,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=target_dim
        )
        
        # Replace self_attn with GlobalRoPEAttention to handle flattened inputs + coords
        layer.self_attn = GlobalRoPEAttention(
            embedding_dim=target_dim,
            num_heads=1,
            downsample_rate=1
        )
        
        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=False, # No additive PE at input
            layer=layer,
            num_layers=1,
        )

    def select_topk_features(self, features, coords):
        """
        Selects top-k features from (B, L, D) tensor based on L2 norm.
        Also returns corresponding coords (B, L, 2).
        """
        B, L, D = features.shape
        k = max(1, int(L * self.topk_ratio))
        
        # Score: L2 Norm
        scores = torch.norm(features, dim=-1) # (B, L)
        topk_scores, topk_indices = torch.topk(scores, k, dim=1) # (B, k)
        
        # Gather Features (B, k, D)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        features_sparse = torch.gather(features, 1, topk_indices_expanded)
        
        # Gather Coords (B, k, 2)
        topk_indices_coords = topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        coords_sparse = torch.gather(coords, 1, topk_indices_coords)
        
        return features_sparse, coords_sparse

    def forward(self, x):
        """
        Override forward to handle Coordinate Generation and Tracking.
        x: (N, C, H, W)
        """
        B_total, C, H, W = x.shape
        
        if self.attn is None or self.d_model is None:
            self._build_layers(x.shape[1])
            
        # Generate Grid Coordinates for this batch
        # Assuming local inputs (0..H, 0..W). If random_crop is used, strict spatial consistency is lost 
        # unless offsets are provided (which they aren't).
        # However, for inference (full img) or no-crop training, this is sufficient.
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).float() # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B_total, 1, 1, 1) # (B, H, W, 2)
        grid_flat = grid.flatten(1, 2) # (B, L, 2)

        x_proj = self.proj_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1)  # (B, L, D)
        
        if self.training:
            T = self.time_steps
            B = B_total // T
            
            x_seq = x_flat.view(B, T, -1, self.d_model) # (B, T, L, D)
            grid_seq = grid_flat.view(B, T, -1, 2)      # (B, T, L, 2)
            
            # Encode memory frames
            mem_encoded = self.memory_encoder(x) # (N, D, H, W)
            mem_encoded = mem_encoded.view(B, T, self.d_model, H, W)

            out_seq = []
            for t in range(T):
                curr = x_seq[:, t]  # (B, L, D)
                curr_coords = grid_seq[:, t] # (B, L, 2)
                
                if t == 0:
                    memory = curr
                    memory_coords = curr_coords
                else:
                    start = max(0, t - self.max_memory)
                    mem_frames = mem_encoded[:, start:t] # (B, k, D, H, W)
                    k_frames = mem_frames.shape[1]
                    
                    mem_list = []
                    coord_list = []
                    
                    for i in range(k_frames):
                        # Extract features
                        feat = mem_frames[:, i] # (B, D, H, W)
                        feat_flat = feat.flatten(2).permute(0, 2, 1) # (B, L, D)
                        
                        # Extract coords (reusing grid since we assume constant H,W/crop for clip?)
                        # WARNING: If random_crop changes per frame in clip, this grid reuse is wrong.
                        # But standard VideoDataset often crops consistency.
                        # We use the grid corresponding to this frame.
                        # In this simple impl, we reuse the current grid assuming aligned clips.
                        coords_flat = grid_seq[:, start + i] # (B, L, 2)
                        
                        # Sparsify NOW
                        feat_sparse, coords_sparse = self.select_topk_features(feat_flat, coords_flat)
                        
                        mem_list.append(feat_sparse)
                        coord_list.append(coords_sparse)

                    if mem_list:
                        memory = torch.cat(mem_list, dim=1) # (B, k*topk, D)
                        memory_coords = torch.cat(coord_list, dim=1) # (B, k*topk, 2)
                    else:
                        # Fallback
                        memory = curr
                        memory_coords = curr_coords
                
                # FORCE DISABLE additive PE in base MemoryAttention (loaded from ckpt)
                # We are passing raw coords, not embeddings.
                if self.attn.pos_enc_at_input:
                    self.attn.pos_enc_at_input = False
                
                # Pass COORDS as pos/query_pos
                res = self.attn(
                    curr=curr, 
                    memory=memory, 
                    curr_pos=curr_coords, 
                    memory_pos=memory_coords
                )
                out_seq.append(res)
            
            out = torch.stack(out_seq, dim=1)
            out = out.view(B_total, H, W, self.d_model).permute(0, 3, 1, 2)
            
        else:
            # Inference
            curr = x_flat
            curr_coords = grid_flat
            
            # Retrieve Memory (Tuple: feats, coords)
            memory, memory_coords = self.retrieve_memory_inference(curr, curr_coords)
            
            # FORCE DISABLE additive PE in base MemoryAttention
            if self.attn.pos_enc_at_input:
                self.attn.pos_enc_at_input = False
                
            res = self.attn(
                curr=curr,
                memory=memory,
                curr_pos=curr_coords, 
                memory_pos=memory_coords
            )
            out = res.view(B_total, H, W, self.d_model).permute(0, 3, 1, 2)


            # Update Memory Bank
            with torch.no_grad():
                mem_encoded = self.memory_encoder(x) 
                mem_flat = mem_encoded.flatten(2).permute(0, 2, 1) # (B, L, D)
                
                # Sparsify
                mem_sparse, coords_sparse = self.select_topk_features(mem_flat, curr_coords)
                
                self.memory_bank.append((mem_sparse.detach(), coords_sparse.detach()))
                
            if len(self.memory_bank) > self.max_memory:
                self.memory_bank.pop(0)
                
        return self.proj_out(out) + x 

    def retrieve_memory_inference(self, curr, curr_coords):
        """Return (memory_feats, memory_coords)"""
        if not self.memory_bank:
            return curr, curr_coords
        
        # Memory Bank stores tuples (feat, coord)
        # Check shapes
        m0_feat, m0_coord = self.memory_bank[0]
        if m0_feat.shape[0] != curr.shape[0]:
             self.memory_bank = []
             return curr, curr_coords
             
        mem_feats = []
        mem_coords = []
        for m_feat, m_coord in self.memory_bank:
            mem_feats.append(m_feat.to(curr.device))
            mem_coords.append(m_coord.to(curr.device))
            
        return torch.cat(mem_feats, dim=1), torch.cat(mem_coords, dim=1)

