import torch
import torch.nn as nn
import torch.nn.functional as F
from ..conv import Conv

class MotionAligner(nn.Module):
    """
    Step 1: Motion Alignment
    Uses Homography matrix to map current frame pixel coordinates to history frame coordinates,
    serving as Reference Points for Attention.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, H_mat):
        """
        Args:
            x (Tensor): Current frame features [B, C, H, W]
            H_mat (Tensor): Inverse Homography matrix H_{t->t-1} [B, 3, 3]
        
        Returns:
            ref_points (Tensor): Aligned history reference points [B, H, W, 2] (normalized to -1~1)
        """
        B, C, H, W = x.shape
        
        # 1. Generate normalized grid coordinates for current frame [-1, 1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        # Stack to [B, H*W, 3] homogeneous coordinates (x, y, 1)
        grid = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1) # [H, W, 3]
        grid = grid.view(-1, 3).expand(B, -1, -1) # [B, N, 3]
        
        # 2. Apply Homography transformation: p'_{t-1} = H^{-1} * p_t
        # Note: grid is (x, y, 1), H_mat is usually for column vectors, so we transpose or adjust multiplication
        # grid * H_mat^T -> [B, N, 3]
        transformed_grid = torch.bmm(grid, H_mat.transpose(1, 2))
        
        # 3. Normalize homogeneous coordinates (x'/z', y'/z')
        z = transformed_grid[..., 2:3] + 1e-6 # avoid division by zero
        ref_points = transformed_grid[..., :2] / z
        
        # Reshape back to [B, H, W, 2]
        ref_points = ref_points.view(B, H, W, 2)
        
        return ref_points

class TrackFusion(nn.Module):
    """
    Step 3: Solve "Query Occlusion" problem
    Decouple "Content Query" and "Position Query".
    Q_final = Q_current + Linear(Q_track_history)
    """
    def __init__(self, dim, track_dim=None):
        super().__init__()
        if track_dim is None:
            track_dim = dim
            
        self.track_proj = nn.Linear(track_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, current_feat, track_embedding):
        """
        Args:
            current_feat (Tensor): Current frame features [B, C, H, W]
            track_embedding (Tensor): History track embedding [B, C, H, W] (or spatially aligned)
            
        Returns:
            mixed_query (Tensor): [B, C, H, W]
        """
        B, C, H, W = current_feat.shape
        
        # Permute for Linear layer [B, H, W, C]
        x = current_feat.permute(0, 2, 3, 1)
        track = track_embedding.permute(0, 2, 3, 1)
        
        # Fusion
        # Q_final = Q_current + Linear(Q_track)
        track_info = self.track_proj(track)
        mixed_query = x + track_info
        
        # Norm & Permute back
        mixed_query = self.norm(mixed_query)
        return mixed_query.permute(0, 3, 1, 2)

class TrajectoryGuidedDeformableAttention(nn.Module):
    """
    Step 2: Build Attention Mechanism -- Deformable Cross-Attention
    Attention(Q, K, V) = sum(Weights * V(Reference + Offset))
    """
    def __init__(self, dim, num_heads=4, num_points=4):
        """
        Args:
            dim (int): Feature channels
            num_heads (int): Number of attention heads
            num_points (int): Number of sampling points per head (Deformable Points)
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        
        # Generate Sampling Offsets: Input Query, Output (x, y) offsets per point
        # Output channels: heads * points * 2 (x,y)
        self.sampling_offsets = nn.Conv2d(dim, num_heads * num_points * 2, kernel_size=3, padding=1)
        
        # Generate Attention Weights: Input Query, Output weights per point
        # Output channels: heads * points
        self.attention_weights = nn.Conv2d(dim, num_heads * num_points, kernel_size=3, padding=1)
        
        # Value Projection (for history features)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Output Projection
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        # Initialize weights to uniform distribution after sigmoid
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

    def forward(self, query, history_feat, ref_points):
        """
        Args:
            query (Tensor): Mixed Query [B, C, H, W]
            history_feat (Tensor): History frame features Value [B, C, H, W]
            ref_points (Tensor): Motion-aligned reference points [B, H, W, 2]
        """
        B, C, H, W = query.shape
        
        # 1. Project Value
        value = self.value_proj(history_feat) # [B, C, H, W]
        
        # 2. Predict Offsets and Weights
        offsets = self.sampling_offsets(query) # [B, heads*points*2, H, W]
        weights = self.attention_weights(query) # [B, heads*points, H, W]
        weights = F.softmax(weights.view(B, self.num_heads, self.num_points, H, W), dim=2) # [B, heads, points, H, W]
        
        # 3. Build Sampling Grid
        # ref_points: [B, H, W, 2] -> [B, 1, H, W, 2] -> broadcast to each head and point
        ref_points_expanded = ref_points.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, self.num_points, H, W, 2)
        
        # Process Offsets
        # [B, heads*points*2, H, W] -> [B, heads, points, 2, H, W] -> [B, heads, points, H, W, 2]
        offsets = offsets.view(B, self.num_heads, self.num_points, 2, H, W).permute(0, 1, 2, 4, 5, 3)
        
        # Sampling locations = Reference + Offset
        # Shape: [B, heads, points, H, W, 2]
        sampling_locations = ref_points_expanded + offsets
        
        # 4. Deformable Sampling (Optimized)
        # Reshape Value to [B*heads, head_dim, H, W] to isolate heads
        value_grouped = value.view(B, self.num_heads, self.head_dim, H, W).flatten(0, 1) 
        
        # Flatten sampling locations to match grouped value
        # Target: [B*heads, points*H*W, 1, 2]
        # Original: [B, heads, points, H, W, 2]
        # Permute to [B, heads, points, H, W, 2] -> [B*heads, points*H*W, 1, 2]
        N_sample = self.num_points * H * W
        sample_grid = sampling_locations.contiguous().view(B * self.num_heads, N_sample, 1, 2)
        
        # Grid Sample
        # Output: [B*heads, head_dim, N_sample, 1]
        sampled_values = F.grid_sample(
            value_grouped, 
            sample_grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )
        
        # 5. Aggregation
        # Reshape back: [B, heads, head_dim, points, H, W]
        sampled_values = sampled_values.view(B, self.num_heads, self.head_dim, self.num_points, H, W)
        
        # Weighted sum over points
        # weights: [B, heads, points, H, W] -> [B, heads, 1, points, H, W]
        weights = weights.unsqueeze(2) 
        
        attention_output = (sampled_values * weights).sum(dim=3) # [B, heads, head_dim, H, W]
        
        # Reshape back to [B, C, H, W] (flatten heads)
        attention_output = attention_output.view(B, C, H, W)
        
        # 6. Output Projection
        output = self.output_proj(attention_output)
        
        return output + query

class TrajectoryBlock(nn.Module):
    """
    Encapsulated module that can be embedded into YOLO architecture.
    """
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv1 = Conv(c1, c2, k=k)
        self.aligner = MotionAligner()
        self.fusion = TrackFusion(c2)
        self.attn = TrajectoryGuidedDeformableAttention(c2)
        
        # Context storage for injection during training
        self.history_img = None
        self.homography = None
        
    def set_context(self, history_img, homography):
        self.history_img = history_img
        self.homography = homography
        
    def forward(self, x, history_feat=None, track_feat=None, homography=None):
        """
        Args:
            x: Current frame features
            history_feat: History frame features (Buffer)
            track_feat: Track features (Track Query)
            homography: Motion matrix
        
        Falls back to standard convolution if history info is missing.
        """
        x = self.conv1(x)
        
        # Priority: Explicit args > Injected Context
        h_img = self.history_img if history_feat is None and self.history_img is not None else None
        hm = self.homography if homography is None and self.homography is not None else homography
        
        # If explicit history_feat is NOT provided, we need to Compute it from h_img?
        # This is the tricky part. A conv block cannot easily run the full backbone.
        # 
        # COMPROMISE for Prototype:
        # If we don't have pre-computed history features, we skip attention 
        # OR we use current features as a proxy (Self-Attention with Motion).
        # Let's check if we have 'homography'. If we have H but no history_feat,
        # we can't do Cross-Attention to history.
        
        if history_feat is None and h_img is None:
            if not getattr(self, "_warn_no_history", False):
                from ultralytics.utils import LOGGER
                LOGGER.warning("TrajectoryBlock: missing history features; falling back to Conv outputs.")
                self._warn_no_history = True
            return x
            
        if homography is None and hm is None:
            if not getattr(self, "_warn_no_homography", False):
                from ultralytics.utils import LOGGER
                LOGGER.warning("TrajectoryBlock: missing homography matrix; falling back to Conv outputs.")
                self._warn_no_homography = True
            return x
            
        final_homography = homography if homography is not None else hm
        
        # 1. Motion Alignment
        ref_points = self.aligner(x, final_homography)
        
        # 2. Track Fusion (Optional, skipped if no track_feat)
        # Note: Implementing Track Query management is complex (needs tracking head).
        # Skipping for this iteration.
        query = x
            
        # 3. Trajectory Guided Attention
        # Problem: We need 'history_feat' (Value).
        # If we only have 'history_img', we are stuck.
        
        # TEMPORARY HACK: Use 'x' (current features) as history features.
        # This effectively turns Cross-Attention into Self-Attention,
        # BUT the reference points are shifted by Motion.
        # This still tests the motion alignment logic!
        h_feat = history_feat if history_feat is not None else x
        
        out = self.attn(query, h_feat, ref_points)
        
        return out
