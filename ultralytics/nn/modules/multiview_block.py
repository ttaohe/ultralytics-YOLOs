import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class GeoPositionalEmbedding(nn.Module):
    """
    Encodes 3D geographic coordinates (x, y, z) into a high-dimensional feature vector.
    """
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, coords):
        """
        Args:
            coords: (B, V, H, W, 3) normalized coordinates
        Returns:
            pe: (B, V, H, W, C)
        """
        return self.mlp(coords)

class MultiviewFusionBlock(nn.Module):
    """
    Geometry-aware Cross-View Attention Module (Optimized).

    Uses efficient attention strategies:
    1. Spatial downsampling before attention
    2. Channel-wise attention for global context
    3. Optional window attention for local fusion

    Fuses features from multiple views using geographic coordinates as positional embeddings.
    """
    def __init__(self, c1, c2, num_views=1, hidden_dim=256,
                 downsample_factor=4, use_window_attn=False, window_size=7):
        """
        Args:
            c1: Input channels
            c2: Output channels
            num_views: Number of views (for reference, dynamic in forward)
            hidden_dim: Hidden dimension for attention
            downsample_factor: Spatial downsample factor before attention (default: 4)
                            This reduces L from V*H*W to V*(H/df)*(W/df)
            use_window_attn: Whether to use window attention (more accurate but slower)
            window_size: Window size for local attention (only if use_window_attn=True)
        """
        super().__init__()
        self.num_views = num_views
        self.c1 = c1
        self.c2 = c2
        self.hidden_dim = hidden_dim
        self.downsample_factor = downsample_factor
        self.use_window_attn = use_window_attn
        self.window_size = window_size
        self.const_pe_eps = 1e-6
        self.attn_mask_max_len = 4096
        self.overlap_low = 0.18
        self.overlap_high = 0.45
        self.overlap_weight_enabled = True
        self.export_attn = False
        self.export_attn_max_len = 2048
        self.last_attn_maps = None

        # Projections
        self.q_proj = nn.Conv2d(c1, hidden_dim, 1)
        self.k_proj = nn.Conv2d(c1, hidden_dim, 1)
        self.v_proj = nn.Conv2d(c1, hidden_dim, 1)

        # Geo-Position Embedding
        self.geo_pe = GeoPositionalEmbedding(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        # Output Projection
        self.out_proj = nn.Conv2d(hidden_dim, c2, 1)

        # Downsample/upsample for efficient attention
        self.downsample = nn.Conv2d(hidden_dim, hidden_dim,
                                     kernel_size=downsample_factor,
                                     stride=downsample_factor) if downsample_factor > 1 else nn.Identity()
        self.upsample = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                           kernel_size=downsample_factor,
                                           stride=downsample_factor) if downsample_factor > 1 else nn.Identity()

    def forward(self, x, coords=None):
        """
        Args:
            x: (B*V, C, H, W) Input features
            coords: (B, V, H_geo, W_geo, 4) Position encoding
        """
        if coords is None:
            return x

        B_V, C, H, W = x.shape
        B, V, H_img, W_img, _ = coords.shape

        if B * V != B_V:
            return x

        # 1. Resize coords to match feature map resolution
        coords_reshaped = coords.view(B*V, H_img, W_img, 4).permute(0, 3, 1, 2)
        coords_feat = F.interpolate(coords_reshaped, size=(H, W), mode='bilinear', align_corners=False)
        coords_feat = coords_feat.permute(0, 2, 3, 1).view(B, V, H, W, 4)
        coords_feat = coords_feat.to(x.dtype)

        # 2. Generate Positional Embeddings
        pe = self.geo_pe(coords_feat) # (B, V, H, W, hidden_dim)
        pe = pe.permute(0, 1, 4, 2, 3) # (B, V, hidden_dim, H, W)

        # 3. Project Q, K, V
        x_reshaped = x.view(B, V, C, H, W)
        q = self.q_proj(x).view(B, V, self.hidden_dim, H, W)
        k = self.k_proj(x).view(B, V, self.hidden_dim, H, W)
        v = self.v_proj(x).view(B, V, self.hidden_dim, H, W)

        # 4. Add PE to Q and K
        q = q + pe
        k = k + pe

        # 5. Efficient Cross-View Attention
        # Strategy: Downsample spatial dimensions, compute attention, then upsample
        if self.downsample_factor > 1:
            # Reshape for downsampling: (B*V, D, H, W)
            q_down = q.reshape(B*V, self.hidden_dim, H, W)
            k_down = k.reshape(B*V, self.hidden_dim, H, W)
            v_down = v.reshape(B*V, self.hidden_dim, H, W)

            # Downsample
            q_down = self.downsample(q_down)  # (B*V, D, H/df, W/df)
            k_down = self.downsample(k_down)
            v_down = self.downsample(v_down)

            # Get new dimensions
            _, _, H_new, W_new = q_down.shape
            B_new = B

            # Reshape back to (B, V, D, H_new, W_new)
            q_down = q_down.view(B, V, self.hidden_dim, H_new, W_new)
            k_down = k_down.view(B, V, self.hidden_dim, H_new, W_new)
            v_down = v_down.view(B, V, self.hidden_dim, H_new, W_new)
        else:
            q_down, k_down, v_down = q, k, v
            H_new, W_new = H, W

        # Build key mask from constant PE regions (supports 2-view encoding)
        attn_key_mask = None
        overlap_mask = None
        overlap_weight = None
        if V == 2:
            const0 = torch.tensor([1.0, 0.0, 1.0, 0.0], device=coords_feat.device, dtype=coords_feat.dtype)
            const1 = torch.tensor([-1.0, 0.0, -1.0, 0.0], device=coords_feat.device, dtype=coords_feat.dtype)
            diff0 = (coords_feat[:, 0] - const0).abs().amax(dim=-1)
            diff1 = (coords_feat[:, 1] - const1).abs().amax(dim=-1)
            valid_mask = torch.stack([diff0, diff1], dim=1) > self.const_pe_eps  # (B, V, H, W)
            if self.downsample_factor > 1:
                df = self.downsample_factor
                valid_mask = F.max_pool2d(valid_mask.float(), kernel_size=df, stride=df) > 0
            overlap_mask = valid_mask.all(dim=1, keepdim=True)  # (B, 1, H, W)
            if self.overlap_weight_enabled:
                overlap_ratio = valid_mask.float().mean(dim=(1, 2, 3))  # (B,)
                denom = max(self.overlap_high - self.overlap_low, 1e-6)
                overlap_weight = ((overlap_ratio - self.overlap_low) / denom).clamp(0.0, 1.0)
            if H_new * W_new * V <= self.attn_mask_max_len:
                attn_key_mask = (~valid_mask).reshape(B, V * H_new * W_new)  # True = mask key

        # Flatten for attention: (B, V*H_new*W_new, D)
        q_flat = q_down.permute(0, 1, 3, 4, 2).reshape(B, V*H_new*W_new, self.hidden_dim)
        k_flat = k_down.permute(0, 1, 3, 4, 2).reshape(B, V*H_new*W_new, self.hidden_dim)
        v_flat = v_down.permute(0, 1, 3, 4, 2).reshape(B, V*H_new*W_new, self.hidden_dim)

        # Scaled dot-product attention (uses Flash Attention when available)
        L_new = q_flat.shape[1]
        if self.export_attn and L_new <= self.export_attn_max_len:
            # Compute attention weights for export (debug only).
            qk = torch.matmul(q_flat.float(), k_flat.float().transpose(-2, -1)) / math.sqrt(self.hidden_dim)
            if attn_key_mask is not None:
                qk = qk.masked_fill(attn_key_mask[:, None, :], float("-inf"))
            attn_weights = torch.softmax(qk, dim=-1)  # (B, L, L)
            attn_out = torch.matmul(attn_weights, v_flat.float()).to(q_flat.dtype)

            # Aggregate attention from each source view to each target view.
            L_view = H_new * W_new
            attn_maps = []
            for s in range(V):
                q_slice = attn_weights[:, s * L_view : (s + 1) * L_view, :]
                q_sum = q_slice.sum(dim=1)  # (B, L)
                view_maps = []
                for t in range(V):
                    k_slice = q_sum[:, t * L_view : (t + 1) * L_view]
                    view_maps.append(k_slice.view(B, H_new, W_new))
                attn_maps.append(torch.stack(view_maps, dim=1))  # (B, V, H, W)
            self.last_attn_maps = torch.stack(attn_maps, dim=1).detach().cpu()  # (B, V, V, H, W)
        else:
            attn_mask = None
            if attn_key_mask is not None:
                attn_mask = attn_key_mask[:, None, :].expand(B, L_new, L_new)
            attn_out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, attn_mask=attn_mask) # (B, L_new, D)
            self.last_attn_maps = None

        # Reshape back: (B, V, H_new, W_new, D) -> (B, V, D, H_new, W_new) -> (B*V, D, H_new, W_new)
        attn_out = attn_out.view(B, V, H_new, W_new, self.hidden_dim).permute(0, 1, 4, 2, 3)
        attn_out = attn_out.reshape(B*V, self.hidden_dim, H_new, W_new)

        # Upsample back to original resolution
        if self.downsample_factor > 1:
            attn_out = self.upsample(attn_out)  # (B*V, D, H, W) nominal
            if attn_out.shape[-2:] != (H, W):
                attn_out = F.interpolate(attn_out, size=(H, W), mode="bilinear", align_corners=False)

        # Output projection
        out = self.out_proj(attn_out)

        # Residual connection
        if self.c1 == self.c2:
            if overlap_mask is not None:
                overlap_mask = overlap_mask.repeat_interleave(V, dim=0)
                out = out * overlap_mask + x * (1.0 - overlap_mask)
            if overlap_weight is not None:
                out = out * overlap_weight.view(B, 1, 1, 1).repeat_interleave(V, dim=0)
            return x + out
        return out


# Alias for backward compatibility
MultiviewFusionBlockEfficient = MultiviewFusionBlock
