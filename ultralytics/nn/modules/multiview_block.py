import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Geometry-aware Cross-View Attention Module.
    Fuses features from multiple views using geographic coordinates as positional embeddings.
    """
    def __init__(self, c1, c2, num_views=1, hidden_dim=256):
        super().__init__()
        self.num_views = num_views # Placeholder, dynamic in forward
        self.c1 = c1
        self.c2 = c2
        self.hidden_dim = hidden_dim
        
        # Projections
        self.q_proj = nn.Conv2d(c1, hidden_dim, 1)
        self.k_proj = nn.Conv2d(c1, hidden_dim, 1)
        self.v_proj = nn.Conv2d(c1, hidden_dim, 1)
        
        # Geo-Position Embedding
        self.geo_pe = GeoPositionalEmbedding(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Output Projection
        self.out_proj = nn.Conv2d(hidden_dim, c2, 1)

    def forward(self, x, coords=None):
        """
        Args:
            x: (B*V, C, H, W) Input features
            coords: (B, V, H_geo, W_geo, 3) Geographic coordinates. 
                    Note: H_geo, W_geo might differ from H, W, need interpolation.
        """
        if coords is None:
            # Fallback if no coords provided (e.g. standard inference without geo data)
            return x 
            
        B_V, C, H, W = x.shape
        # Infer B and V from coords
        # coords: (B, V, H_img, W_img, 3)
        B, V, H_img, W_img, _ = coords.shape
        
        if B * V != B_V:
            # Mismatch in batch size, possibly due to partial batch or different logic
            # For now, assume strict alignment
            return x

        # 1. Resize coords to match feature map resolution (H, W)
        # coords is (B, V, H_img, W_img, 3) -> (B*V, 3, H_img, W_img)
        coords_reshaped = coords.view(B*V, H_img, W_img, 3).permute(0, 3, 1, 2)
        
        # Interpolate to (H, W)
        coords_feat = F.interpolate(coords_reshaped, size=(H, W), mode='bilinear', align_corners=False)
        
        # Back to (B, V, H, W, 3) for PE
        coords_feat = coords_feat.permute(0, 2, 3, 1).view(B, V, H, W, 3)

        # 2. Generate Positional Embeddings
        pe = self.geo_pe(coords_feat) # (B, V, H, W, hidden_dim)
        pe = pe.permute(0, 1, 4, 2, 3) # (B, V, hidden_dim, H, W)

        # 3. Project Q, K, V
        # x is (B*V, C, H, W) -> (B, V, C, H, W)
        x_reshaped = x.view(B, V, C, H, W)
        
        # Apply projections on flattened batch (B*V)
        q = self.q_proj(x).view(B, V, self.hidden_dim, H, W)
        k = self.k_proj(x).view(B, V, self.hidden_dim, H, W)
        v = self.v_proj(x).view(B, V, self.hidden_dim, H, W)

        # 4. Add PE to Q and K
        q = q + pe
        k = k + pe
        
        # 5. Prepare for Attention
        # Flatten spatial dimensions: (B, V, D, H, W) -> (B, V, D, H*W)
        # We want to attend across ALL views.
        # Global Attention: Query from View i attends to Keys from ALL Views (0..V-1)
        
        # Reshape for scaled_dot_product_attention: (Batch, Num_Heads, Seq_Len, Dim)
        # Here we treat 'V' as part of the sequence length or batch?
        # We want cross-view. 
        # Let's flatten V and H, W together: (B, V*H*W, D)
        
        q_flat = q.permute(0, 1, 3, 4, 2).reshape(B, V*H*W, self.hidden_dim) # (B, N_tokens, D)
        k_flat = k.permute(0, 1, 3, 4, 2).reshape(B, V*H*W, self.hidden_dim)
        v_flat = v.permute(0, 1, 3, 4, 2).reshape(B, V*H*W, self.hidden_dim)
        
        # Attention
        # (B, L, D) x (B, L, D) -> (B, L, L) - This is very expensive if H*W is large!
        # H, W at P3 is 80x80 = 6400. V=3 -> 19200 tokens. 
        # 19200^2 is huge.
        # Optimization: We might want to restrict attention or downsample.
        # For now, implementing full attention as per design, but noting the cost.
        
        attn_out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat) # (B, L, D)
        
        # Reshape back
        attn_out = attn_out.view(B, V, H, W, self.hidden_dim).permute(0, 1, 4, 2, 3) # (B, V, D, H, W)
        attn_out = attn_out.reshape(B*V, self.hidden_dim, H, W)
        
        # Output projection
        out = self.out_proj(attn_out)
        
        # Residual connection
        if self.c1 == self.c2:
            return x + out
        return out
