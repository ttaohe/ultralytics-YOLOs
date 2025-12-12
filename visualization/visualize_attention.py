
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.nn.modules.video_attention import SparseMemoryAttention, GlobalRoPEAttention

class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.captured_data = {}
        self.original_methods = {}
        self.image_buffer = [] # Store raw images for visualization

    def __enter__(self):
        self._monkey_patch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore()

    def _monkey_patch(self):
        """
        Replace methods in SparseMemoryAttention and GlobalRoPEAttention
        to capture intermediate data.
        """
        # Initialize to None for safety
        self.attn_module = None
        self.target_module = None
        
        # 1. Locate SparseMemoryAttention instance
        for m in self.model.modules():
            if isinstance(m, SparseMemoryAttention):
                self.target_module = m
                break
        
        if self.target_module is None:
            print("Warning: No SparseMemoryAttention found.")
            return

        self.target_module._viz_indices_list = []

        # 2. Patch select_topk_features
        if not hasattr(self.target_module, 'original_select_topk'):
            self.target_module.original_select_topk = self.target_module.select_topk_features
            # Bind the visualizer's patched method to the module instance
            self.target_module.select_topk_features = self.patched_select_topk_features.__get__(self.target_module, self.target_module.__class__)
        
        # 3. Locate GlobalRoPEAttention module instance
        # SparseMemoryAttention -> attn (MemoryAttention) -> layers (ModuleList)
        if hasattr(self.target_module, 'attn') and hasattr(self.target_module.attn, 'layers'):
            if len(self.target_module.attn.layers) > 0:
                curr_layer = self.target_module.attn.layers[0] # Assume layer 0
                if hasattr(curr_layer, 'cross_attn_image') and isinstance(curr_layer.cross_attn_image, GlobalRoPEAttention):
                    self.attn_module = curr_layer.cross_attn_image
        
        if self.attn_module is None:
            print("Warning: No GlobalRoPEAttention found within SparseMemoryAttention.")
            return

        # 4. Patch forward of GlobalRoPEAttention
        if not hasattr(self.attn_module, 'original_attn_forward'):
            self.attn_module.original_attn_forward = self.attn_module.forward
            # Bind the visualizer's patched method to the module instance
            self.attn_module.forward = self.patched_attn_forward.__get__(self.attn_module, self.attn_module.__class__)

    def _restore(self):
        if self.target_module and hasattr(self.target_module, 'original_select_topk'):
            self.target_module.select_topk_features = self.target_module.original_select_topk
            del self.target_module.original_select_topk
            
        if self.attn_module and hasattr(self.attn_module, 'original_attn_forward'):
            self.attn_module.forward = self.attn_module.original_attn_forward
            del self.attn_module.original_attn_forward

    def patched_select_topk_features(self, features, coords, k_ratio=None):
        """
        Patched method for SparseMemoryAttention.select_topk_features.
        'self' here refers to the SparseMemoryAttention instance.
        """
        # Capture indices
        k_ratio = k_ratio if k_ratio is not None else self.topk_ratio
        
        # features: [B, L, D]
        B, L_total, D = features.shape
        k = max(1, int(L_total * k_ratio))
        
        # Calculate Norm
        norm = torch.norm(features, dim=-1) # [B, L]
        _, indices = torch.topk(norm, k, dim=1)
        
        # Save indices to the module instance for later retrieval by the visualizer
        # We accumulate indices in a list to track history matching the memory bank
        if not hasattr(self, '_viz_indices_list'):
             self._viz_indices_list = []
        
        self._viz_indices_list.append(indices.detach().cpu())
        
        # Call original method
        return self.original_select_topk(features, coords)

    def patched_attn_forward(self, q, k, v, q_coords=None, k_coords=None, num_k_exclude_rope=0):
        """
        Patched method for GlobalRoPEAttention.forward.
        'self' here refers to the GlobalRoPEAttention instance.
        """
        print("DEBUG: patched_attn_forward called!")
        # Standard Projection
        q_ = self.q_proj(q)
        k_ = self.k_proj(k)
        v_ = self.v_proj(v)

        q_ = self._separate_heads(q_, self.num_heads)
        k_ = self._separate_heads(k_, self.num_heads)
        v_ = self._separate_heads(v_, self.num_heads)
        
        # Save Q, K, and scale to the module instance for on-demand attention calculation
        self._viz_context = {
            'q': q_.detach().cpu(), 
            'k': k_.detach().cpu(),
            'scale': (q_.size(-1)) ** -0.5,
            # We might want to save coords if we want to viz them or debug Global RoPE
            'q_coords': q_coords.detach().cpu() if q_coords is not None else None,
            'k_coords': k_coords.detach().cpu() if k_coords is not None else None
        }
        
        # Pass to proper forward which handles Global Apply etc.
        # We cannot just call SDPA because GlobalRoPEAttention.forward does rotation logic.
        # BUT, calling self.original_attn_forward will repeat the projections.
        
        # In this specific monkey patch, we usually want to intercept input/output.
        # But GlobalRoPEAttention logic is inside forward.
        
        # If we call original_attn_forward, it will do everything.
        # We just want to capture context.
        # So we can just call original_attn_forward directly.
        
        return self.original_attn_forward(q, k, v, q_coords, k_coords, num_k_exclude_rope)


    def _get_letterbox_params(self, src_shape, dst_size):
        """
        Calculate Letterbox scaling and padding.
        src_shape: (H, W)
        dst_size: int (imgsz)
        """
        h0, w0 = src_shape
        r = min(dst_size / h0, dst_size / w0)
        
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = dst_size - new_unpad[0], dst_size - new_unpad[1]
        dw, dh = dst_size - new_unpad[0], dst_size - new_unpad[1]
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        return r, (dw, dh)

    def visualize(self, current_frame_img, memory_frames_imgs, query_box=None, imgsz=640):
        # Retrieve captured data
        if not hasattr(self.attn_module, '_viz_context'):
            print("No attention data captured.")
            return None, None
            
        context = self.attn_module._viz_context
        q_tensor = context['q'] # [B, H, Nq, D]
        k_tensor = context['k'] # [B, H, Nk, D]
        scale = context['scale']
        
        Nq = q_tensor.shape[2]
        Nk = k_tensor.shape[2]
        grid_dim = int(np.sqrt(Nq))
        
        if grid_dim * grid_dim != Nq:
            print(f"Warning: Non-square feature map (Nq={Nq}).")
        
        # Letterbox Params
        model_stride = imgsz / grid_dim 
        H, W = current_frame_img.shape[:2]
        ratio, (pad_w, pad_h) = self._get_letterbox_params((H, W), imgsz)
        
        # Determine Sample Points (Center + 9 Random)
        sample_points_norm = []
        if query_box is not None:
             cx, cy, w, h = query_box
             sample_points_norm.append((cx, cy))
             np.random.seed(42)
             for _ in range(9):
                 rx = cx + (np.random.rand() - 0.5) * w
                 ry = cy + (np.random.rand() - 0.5) * h
                 sample_points_norm.append((rx, ry))
        else:
            sample_points_norm.append((0.5, 0.5))

        # Prepare Current Frame (Draw Green Dots)
        vis_curr = current_frame_img.copy()
        q_indices = []

        for (qx_norm, qy_norm) in sample_points_norm:
            qx_norm = min(max(qx_norm, 0.0), 1.0)
            qy_norm = min(max(qy_norm, 0.0), 1.0)

            qx_pixel = qx_norm * W
            qy_pixel = qy_norm * H
            
            # Draw sample point
            cv2.circle(vis_curr, (int(qx_pixel), int(qy_pixel)), 3, (0, 255, 0), -1)

            # Map to Grid
            q_lb_x = qx_pixel * ratio + pad_w
            q_lb_y = qy_pixel * ratio + pad_h
            gx = int(q_lb_x / model_stride)
            gy = int(q_lb_y / model_stride)
            gx = min(max(gx, 0), grid_dim - 1)
            gy = min(max(gy, 0), grid_dim - 1)
            q_idx = gy * grid_dim + gx
            
            if q_idx < Nq:
                q_indices.append(q_idx)

        if not q_indices:
            return None, None
            
        # Vectorized Attention Calculation
        q_idx_tensor = torch.tensor(q_indices, device=q_tensor.device)
        q_vecs = torch.index_select(q_tensor, 2, q_idx_tensor) # [B, H, N_samples, D]
        
        attn_logits = torch.matmul(q_vecs, k_tensor.transpose(-2, -1)) * scale
        attn_weights = attn_logits.softmax(dim=-1)
        
        # Average over heads -> [N_samples, Nk]
        attn_dist = attn_weights.mean(dim=1).squeeze(0) 
        
        # Get Top-1 Target for each sample
        vals, indices = attn_dist.topk(1, dim=-1)
        vals, indices = vals.flatten().cpu().numpy(), indices.flatten().cpu().numpy()
        
        # --- Memory Alignment Fix ---
        if not hasattr(self.target_module, '_viz_indices_list'):
             print("No memory indices list found.")
             return None, None
             
        # Align accumulated indices with current Memory Tensor (FIFO)
        # k_tensor has 'Nk' tokens. We need to find the suffix of _viz_indices_list that sums to Nk.
        all_indices_list = self.target_module._viz_indices_list
        relevant_indices = []
        accumulated_k = 0
        suffix_start_idx = -1
        
        # Iterate backwards
        for i in range(len(all_indices_list) - 1, -1, -1):
            curr_k = all_indices_list[i].shape[1]
            accumulated_k += curr_k
            relevant_indices.insert(0, all_indices_list[i])
            if accumulated_k == Nk:
                suffix_start_idx = i
                break
            elif accumulated_k > Nk:
                print(f"Warning: Index alignment mismatch (Accum={accumulated_k}, Nk={Nk}).")
                return None, None
        
        if suffix_start_idx == -1:
            # Could happen if Nk is 0 or mismatch
            if Nk == 0: return vis_curr, []
            print(f"Warning: Could not align indices (Nk={Nk}).")
            return None, None

        # Slice memory frames to match the indices suffix
        # relevant_indices corresponds to the full set of frames in the Memory Bank.
        # memory_frames_imgs corresponds to the last N frames we have images for.
        
        num_memory_indices = len(relevant_indices)
        num_available_imgs = len(memory_frames_imgs)
        
        # Calculate lag. If lag > 0, we are missing images for the oldest memory frames.
        # We can still visualize the matched ones.
        img_start_offset = num_available_imgs - num_memory_indices
        
        # Determine the aligned images list
        # We essentially want to map: relevant_indices[i] -> memory_frames_imgs[img_start_offset + i]
        # Valid 'i' are those where (img_start_offset + i) >= 0 and < num_available_imgs
        
        vis_mems_unaligned = [img.copy() for img in memory_frames_imgs]
        
        # Calculate Offsets for aligned indices (MUST use full relevant_indices for correct Key addressing)
        k_sizes = [ind.shape[1] for ind in relevant_indices]
        offsets = [0] + list(np.cumsum(k_sizes))
        
        # Draw Results
        print(f"Visualizing {len(indices)} points mapping to aligned history (Buffer: {num_available_imgs} / Memory: {num_memory_indices} frames):")
        
        def draw_triangle(img, pt, size, color):
            x, y = pt
            p1 = (x, y - size)
            p2 = (x - size, y + size)
            p3 = (x + size, y + size)
            cv2.drawContours(img, [np.array([p1, p2, p3])], 0, color, -1)
        
        for i, (g_idx, val) in enumerate(zip(indices, vals)):
            # Find frame in relevant_indices
            frame_idx = -1
            for f in range(len(offsets) - 1):
                if offsets[f] <= g_idx < offsets[f+1]:
                    frame_idx = f
                    break
            
            if frame_idx == -1: continue

            # Map to Image Index
            # img_idx = img_start_offset + frame_idx
            # Wait, let's trace:
            # If buffer has 8 images [0..7], memory has 3 indices. 
            # Means memory is for [5, 6, 7]? Or [0, 1, 2]? 
            # 'relevant_indices' is the suffix of ALL history. 
            # 'memory_frames_imgs' is the END of the stream buffer.
            # So relevant_indices[-1] SHOULD align with memory_frames_imgs[-1].
            
            # So:
            # matched_img_idx = (len(imgs) - 1) - (len(indices) - 1 - frame_idx)
            #               = len(imgs) - len(indices) + frame_idx
            
            img_idx = num_available_imgs - num_memory_indices + frame_idx
            
            if img_idx < 0:
                # We don't have this image (too old)
                continue
            if img_idx >= num_available_imgs:
                # Should not happen if alignment correct
                continue

            local_k_idx = g_idx - offsets[frame_idx]
            dense_grid_idx = relevant_indices[frame_idx][0, local_k_idx].item()
            
            # Grid -> Pixel
            m_gy = dense_grid_idx // grid_dim
            m_gx = dense_grid_idx % grid_dim
            
            m_lb_x = m_gx * model_stride + model_stride // 2
            m_lb_y = m_gy * model_stride + model_stride // 2
            
            mx = int((m_lb_x - pad_w) / ratio)
            my = int((m_lb_y - pad_h) / ratio)
            
            print(f"frame_idx: {frame_idx} (img: {img_idx}) score: {val:.4f}", mx, my)
            draw_triangle(vis_mems_unaligned[img_idx], (mx, my), 6, (0, 0, 255))
            
        return vis_curr, vis_mems_unaligned

