import torch
import torch.nn as nn
import math
from ultralytics.models.sam.modules.memory_attention import MemoryAttention, MemoryAttentionLayer
from ultralytics.models.sam.modules.blocks import RoPEAttention, CXBlock
from ultralytics.models.sam.modules.encoders import MemoryEncoder as SAMMemoryEncoder
from ultralytics.models.sam.modules.utils import compute_axial_cis, apply_rotary_enc

import torch.nn.functional as F

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
            # Verify consistency if possible
            if h * w != q_heads.shape[-2]:
                # Fallback if shape doesn't match token count (e.g. if flattened differently)
                # But we trust the caller provided correct H, W for the current Q.
                pass
        else:
            # Fallback to square assumption or 1D
            num_tokens = q_heads.shape[-2]
            side = int(math.sqrt(num_tokens))
            if side * side == num_tokens:
                h, w = side, side
            else:
                h, w = 1, num_tokens

        # Recompute freqs_cis if needed
        # We check if we need to update based on shape or if it's not set
        # Note: self.freqs_cis might be on wrong device or wrong shape
        if self.freqs_cis is None or self.freqs_cis.shape[0] != q_heads.shape[-2] or \
           (self.spatial_shape is not None and self.freqs_cis.shape[0] == q_heads.shape[-2]): 
           # The last condition is tricky: if shape matches but H,W changed (e.g. 16x4 vs 8x8), we should recompute.
           # But standard RoPE doesn't store H,W. 
           # We will just always recompute if spatial_shape is set, to be safe.
           # Optimization: Cache it? For now, just compute.
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        
        self.freqs_cis = self.freqs_cis.to(q.device)
        
        if q_heads.shape[-2] != k_heads.shape[-2]:
             assert self.rope_k_repeat

        num_k_rope = k_heads.size(-2) - num_k_exclude_rope
        
        q_enc, k_enc_part = apply_rotary_enc(
            q_heads,
            k_heads[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )
        k_heads[:, :, :num_k_rope] = k_enc_part

        # Attention (Optimized using F.scaled_dot_product_attention)
        # Uses Flash Attention / Efficient Attention backend if available
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
        # CXBlock for efficient feature processing (depthwise convs)
        self.encode = CXBlock(dim=out_dim, kernel_size=3, padding=1, use_dwconv=True)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.encode(x)
        return x

class YOLOMemoryAttentionLayer(MemoryAttentionLayer):
    """
    Subclass of MemoryAttentionLayer that fixes the hardcoded embedding dimensions
    in the original SAM2 implementation to support variable d_model.
    """
    def __init__(self, d_model=256, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False):
        # Initialize parent (will create layers with hardcoded 256 dims)
        super().__init__(d_model, dim_feedforward, dropout, pos_enc_at_attn, pos_enc_at_cross_attn_keys, pos_enc_at_cross_attn_queries)
        
        # Overwrite with correct dimensions AND correct RoPE implementation
        self.self_attn = YOLORoPEAttention(embedding_dim=d_model, num_heads=1, downsample_rate=1)
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
    It expects input x of shape (B*T, C, H, W) during training, where T is time steps.
    It relies on an external attribute (e.g., model.time_steps) or a fixed window.
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

    def forward(self, x):
        """
        x: (N, C, H, W)
        """
        B_total, C, H, W = x.shape
        
        if self.attn is None or self.d_model is None:
            self._build_layers(x.shape[1])
            
        # Initialize temporal position encoding if needed
        if self.maskmem_tpos_enc is None:
            # Shape: (max_memory, 1, 1, d_model)
            self.maskmem_tpos_enc = nn.Parameter(torch.zeros(self.max_memory, 1, 1, self.d_model).to(x.device))
            nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        # Pass spatial shape to RoPE layers
        for layer in self.attn.layers:
            if isinstance(layer.self_attn, YOLORoPEAttention):
                layer.self_attn.set_spatial_shape(H, W)
            if isinstance(layer.cross_attn_image, YOLORoPEAttention):
                layer.cross_attn_image.set_spatial_shape(H, W)


        # Process input
        # Note: We use proj_in for the 'curr' input to attention
        x_proj = self.proj_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1)  # (N, tokens, d_model)
        
        if self.training:
            # Training mode: Expecting N = Batch * Time
            T = self.time_steps
            B = B_total // T
            
            # Reshape to (B, T, L, D)
            x_seq = x_flat.view(B, T, -1, self.d_model)
            
            # For memory encoding in training, we need to process the sequence.
            # We can process all frames through memory encoder efficiently.
            # But memory encoder expects (N, C, H, W).
            # x is (N, C, H, W).
            mem_encoded = self.memory_encoder(x) # (N, D, H, W)
            
            # Apply Temporal Positional Encoding to Memory Features?
            # SAM2 applies it BEFORE flattening.
            # But here we are in batch mode. We need to handle T dimension.
            
            # Let's split by T to apply t-pos encoding correctly
            mem_encoded = mem_encoded.view(B, T, self.d_model, H, W)

            out_seq = []
            for t in range(T):
                curr = x_seq[:, t]  # (B, L, D)
                if t == 0:
                    # For the first frame, use itself as memory (no history)
                    memory = curr 
                else:
                    start = max(0, t - self.max_memory)
                    # Extract memory frames
                    mem_frames = mem_encoded[:, start:t] # (B, k, D, H, W)
                    k = mem_frames.shape[1]
                    
                    # Apply Temporal Positional Encoding
                    # We need to add t_pos embedding based on distance to current frame t.
                    # The memory frames are at indices [start, start+1, ..., t-1]
                    # Distance to t: [t-start, t-(start+1), ..., 1]
                    # SAM2 uses index: self.num_maskmem - t_pos - 1
                    # Let's simplify: Add embedding corresponding to "lag"
                    
                    mem_frames_list = []
                    for i in range(k):
                        # absolute time index of this memory frame
                        t_mem = start + i
                        # how far back is this frame? lag = t - t_mem (1 means previous frame)
                        lag = t - t_mem
                        
                        # Map lag to index in maskmem_tpos_enc
                        # If lag=1 (most recent), use index max_memory-1 (like SAM2)?
                        # SAM2 logic: t_pos in range(1, num_maskmem). t_pos=1 is most recent.
                        # Index = num_maskmem - t_pos - 1
                        # If t_pos=1 -> index = num_maskmem - 2.
                        # Let's just map 0 to "furthest" and max_memory-1 to "nearest" (lag=1).
                        # Or simply: index = (self.max_memory - lag) % self.max_memory
                        
                        # Safe index clamping
                        idx = max(0, self.max_memory - lag)
                        t_enc = self.maskmem_tpos_enc[idx] # (1, 1, D)
                        
                        feat = mem_frames[:, i] # (B, D, H, W)
                        # Flatten first to align with t_enc (1, 1, D)
                        feat_flat = feat.flatten(2).permute(0, 2, 1) # (B, L, D)
                        
                        # Add t-pos encoding
                        feat_flat = feat_flat + t_enc
                        mem_frames_list.append(feat_flat)

                    memory = torch.cat(mem_frames_list, dim=1) # (B, k*L, D)
                
                res = self.attn(
                    curr=curr, 
                    memory=memory, 
                    curr_pos=torch.zeros_like(curr), 
                    memory_pos=torch.zeros_like(memory)
                )
                out_seq.append(res)
            
            # Stack back: (B, T, L, D)
            out = torch.stack(out_seq, dim=1)
            out = out.view(B_total, H, W, self.d_model).permute(0, 3, 1, 2) # (N, D, H, W)
            
        else:
            # Inference mode
            curr = x_flat # (B, L, D)
            
            if not self.memory_bank:
                # If no memory, use current (encoded) as memory or just curr?
                # If we use curr as memory, we should encode it?
                # But for t=0, usually we just want self-attention.
                # However, cross-attn will run.
                memory = curr
            else:
                # Check if batch size or token count matches
                if self.memory_bank[0].shape[0] != curr.shape[0] or self.memory_bank[0].shape[1] != curr.shape[1]:
                    self.memory_bank = []
                    memory = curr
                else:
                    # self.memory_bank stores (B, L, D)
                    # We need to apply t-pos encoding here too!
                    # Memory bank is ordered from oldest to newest.
                    # Newest is at index -1 (lag=1).
                    
                    mem_list = []
                    num_mem = len(self.memory_bank)
                    for i, m in enumerate(self.memory_bank):
                        # i=0 is oldest. i=num_mem-1 is newest.
                        # lag = num_mem - i
                        lag = num_mem - i
                        idx = max(0, self.max_memory - lag)
                        
                        t_enc = self.maskmem_tpos_enc[idx] # (1, 1, D)
                        # m is (B, L, D)
                        # Ensure t_enc matches m dimensions
                        m_enc = m.to(curr.device) + t_enc
                        mem_list.append(m_enc)

                    memory = torch.cat(mem_list, dim=1)

            res = self.attn(
                curr=curr,
                memory=memory,
                curr_pos=torch.zeros_like(curr),
                memory_pos=torch.zeros_like(memory),
            )
            out = res.view(B_total, H, W, self.d_model).permute(0, 3, 1, 2)

            # Update memory bank with ENCODED feature
            # We encode 'x' (the raw input)
            with torch.no_grad():
                mem_encoded = self.memory_encoder(x) # (B, D, H, W)
                # No t-pos encoding stored in bank (it depends on relative lag)
                mem_flat = mem_encoded.flatten(2).permute(0, 2, 1) # (B, L, D)
                self.memory_bank.append(mem_flat.detach())
                
            if len(self.memory_bank) > self.max_memory:
                self.memory_bank.pop(0)
                
        return self.proj_out(out) + x # Residual connection

    def reset_memory(self):
        self.memory_bank = []

    def _build_layers(self, in_channels):
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        
        # Input projection for 'curr' path
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        
        # Output projection
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        
        # Memory Encoder (new)
        # It takes raw input channels and outputs target_dim
        self.memory_encoder = YOLOMemoryEncoder(in_channels, target_dim)

        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=True,
            layer=YOLOMemoryAttentionLayer(d_model=target_dim),
            num_layers=1,
        )
