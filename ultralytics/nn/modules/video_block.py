import torch
import torch.nn as nn
import math
from ultralytics.models.sam.modules.memory_attention import MemoryAttention, MemoryAttentionLayer
from ultralytics.models.sam.modules.blocks import RoPEAttention
from ultralytics.models.sam.modules.encoders import MemoryEncoder as SAMMemoryEncoder
from ultralytics.models.sam.modules.utils import compute_axial_cis

class YOLORoPEAttention(RoPEAttention):
    """
    Subclass of RoPEAttention that handles non-square spatial dimensions correctly.
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
        # Calculate actual spatial dimensions if possible, otherwise fallback to sqrt (which fails for non-square)
        # Here we assume 1D RoPE if we can't determine 2D, or just try to handle flattened tokens better.
        # SAM2 implementation forces 2D grid assumption in compute_axial_cis.
        
        # For YOLO, we can't easily recover H, W from just q (which is B, N, C).
        # But we know N = H * W.
        # If we just want it to run without crashing, we can ensure freqs_cis matches N.
        # But compute_axial_cis expects end_x, end_y.
        
        # Hack: if not square, maybe we can just use 1D RoPE logic or force a 1D grid?
        # If we set end_x = N, end_y = 1?
        
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # Separate into heads
        q_heads = self._separate_heads(q_proj, self.num_heads) # (B, nHead, N, C_head)
        k_heads = self._separate_heads(k_proj, self.num_heads)
        v_heads = self._separate_heads(v_proj, self.num_heads)

        # Correctly update freqs_cis for current token count
        num_tokens = q_heads.shape[-2]
        current_cis_tokens = self.freqs_cis.shape[0]
        
        if current_cis_tokens != num_tokens:
            # If num_tokens is a perfect square, assume square.
            side = int(math.sqrt(num_tokens))
            if side * side == num_tokens:
                w, h = side, side
            else:
                # If not square, we treat it as 1D sequence (W=num_tokens, H=1)
                # or we could try to find factors. 1D is safer if we don't know.
                # But compute_axial_cis does outer product of x and y freqs.
                # If H=1, y_freqs might be trivial?
                w, h = num_tokens, 1
            
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)

        # Standard forward logic from parent, but using our local variables
        # Apply rotary position encoding
        self.freqs_cis = self.freqs_cis.to(q.device)
        
        # We need to access apply_rotary_enc from where it is imported in parent or import it here
        from ultralytics.models.sam.modules.utils import apply_rotary_enc
        
        if q_heads.shape[-2] != k_heads.shape[-2]:
             assert self.rope_k_repeat

        num_k_rope = k_heads.size(-2) - num_k_exclude_rope
        
        # The parent implementation re-calls _separate_heads etc. 
        # We should just call parent logic but we need to make sure self.freqs_cis is correct BEFORE calling parent.
        # But parent method recalculates w = h = sqrt(...) and overwrites self.freqs_cis!
        # So we MUST override the entire forward method.
        
        q_enc, k_enc_part = apply_rotary_enc(
            q_heads,
            k_heads[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )
        k_heads[:, :, :num_k_rope] = k_enc_part

        # Attention
        _, _, _, c_per_head = q_heads.shape
        attn = q_enc @ k_heads.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v_heads

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

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

        x_proj = self.proj_in(x)
        x_flat = x_proj.flatten(2).permute(0, 2, 1)  # (N, tokens, d_model)
        
        if self.training:
            # Training mode: Expecting N = Batch * Time
            T = self.time_steps
            B = B_total // T
            
            # Reshape to (B, T, L, D)
            x_seq = x_flat.view(B, T, -1, self.d_model)
            
            out_seq = []
            for t in range(T):
                curr = x_seq[:, t]  # (B, L, D)
                if t == 0:
                    memory = curr
                else:
                    start = max(0, t - self.max_memory)
                    memory = x_seq[:, start:t].flatten(1, 2)  # (B, k*L, D)
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
            curr = x_flat # (B, L, D) -> Already B, L, D from permute(0, 2, 1)
            
            # Memory handling:
            # If we use batch_first=True in MemoryAttention (default), it expects (Batch, SeqLen, Dim).
            # However, the assertion at line 280 `curr.shape[1] == memory.shape[1]` implies strict length check
            # IF it interprets dim 1 as Batch Size (i.e. Seq-First input).
            # Let's try transposing to Seq-First: (L, B, D)
            
            curr_t = curr.transpose(0, 1) # (L, B, D)
            
            if not self.memory_bank:
                memory_t = curr_t # (L, B, D)
            else:
                # self.memory_bank stores (B, L, D). 
                # Concat along time/L dim -> (B, M*L, D).
                # Transpose -> (M*L, B, D).
                memory_cat = torch.cat(self.memory_bank, dim=1)
                memory_t = memory_cat.transpose(0, 1)

            res_t = self.attn(
                curr=curr_t,
                memory=memory_t,
                curr_pos=torch.zeros_like(curr_t),
                memory_pos=torch.zeros_like(memory_t),
            )
            # res_t is (L, B, D). Transpose back to (B, L, D).
            out = res_t.transpose(0, 1).view(B_total, H, W, self.d_model).permute(0, 3, 1, 2)

            self.memory_bank.append(curr.detach())
            if len(self.memory_bank) > self.max_memory:
                self.memory_bank.pop(0)
                
        return self.proj_out(out) + x # Residual connection

    def reset_memory(self):
        self.memory_bank = []

    def _build_layers(self, in_channels):
        target_dim = self.requested_dim or in_channels
        self.d_model = target_dim
        self.proj_in = nn.Conv2d(in_channels, target_dim, 1) if in_channels != target_dim else nn.Identity()
        self.proj_out = nn.Conv2d(target_dim, in_channels, 1) if target_dim != in_channels else nn.Identity()
        self.attn = MemoryAttention(
            d_model=target_dim,
            pos_enc_at_input=True,
            layer=YOLOMemoryAttentionLayer(d_model=target_dim),
            num_layers=1,
        )
