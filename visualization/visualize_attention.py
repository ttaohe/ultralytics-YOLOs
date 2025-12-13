
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
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
        
        # Infer Grid Dimensions from q_coords if available
        # q_coords is (B, Nq, 2) containing (x, y) grid coordinates
        if 'q_coords' in context and context['q_coords'] is not None:
             q_coords = context['q_coords'][0] # Take first batch
             # q_coords are likely 0..W-1, 0..H-1
             # We assume they cover the full grid.
             max_x = q_coords[:, 0].max().item()
             max_y = q_coords[:, 1].max().item()
             
             # If strictly grid coordinates:
             w_feat = int(max_x) + 1
             h_feat = int(max_y) + 1
             
             if w_feat * h_feat != Nq:
                 # Fallback if coords aren't perfect grid (e.g. sparse queries?)
                 # But MemoryAttention inference q is usually full dense 'curr'.
                 print(f"Warning: Coords inferred dims ({w_feat}x{h_feat}={w_feat*h_feat}) != Nq ({Nq}). Fallback to sqrt.")
                 side = int(np.sqrt(Nq))
                 w_feat, h_feat = side, side
        else:
             # Fallback
             side = int(np.sqrt(Nq))
             w_feat, h_feat = side, side
        
        grid_dim = w_feat # For legacy compat mostly, but we should use w/h explicitly
        
        if w_feat * h_feat != Nq:
            print(f"Warning: Non-rectangular feature map (Nq={Nq}, inferred {w_feat}x{h_feat}).")
        
        # Letterbox Params
        # Check if non-square stride is needed
        # Assuming original image was square-padded to imgsz, 
        # then feature map should respect aspect ratio of that padded image.
        # Actually YOLO usually pads to stride multiple.
        # model_stride is how many pixels one grid cell represents.
        model_stride_w = imgsz / w_feat
        model_stride_h = imgsz / h_feat 
        H, W = current_frame_img.shape[:2]
        ratio, (pad_w, pad_h) = self._get_letterbox_params((H, W), imgsz)
        


        # Prepare Current Frame (Draw Green Dots)
        vis_curr = current_frame_img.copy()

        # 2. Select Query Tokens
        # REFACTOR: Use the Center Point of the query box instead of random sampling.
        # This focuses the attention analysis on the core of the object.
        q_indices = []
        
        if query_box is not None:
            # query_box is (cx_norm, cy_norm, w_norm, h_norm)
            cx_norm, cy_norm, w_norm, h_norm = query_box
            
            # Convert normalized center to pixel coordinates
            cx_pixel = cx_norm * W
            cy_pixel = cy_norm * H
            
            # Draw sample point
            cv2.circle(vis_curr, (int(cx_pixel), int(cy_pixel)), 3, (0, 255, 0), -1)

            # Map to Feature Grid
            # Apply Inverse Letterbox & Stride
            q_lb_x = cx_pixel * ratio + pad_w
            q_lb_y = cy_pixel * ratio + pad_h
            
            gx = int(q_lb_x / model_stride_w)
            gy = int(q_lb_y / model_stride_h)
            
            # Clamp to grid size
            gx = min(max(gx, 0), w_feat - 1)
            gy = min(max(gy, 0), h_feat - 1)
            
            # Flatten index
            center_idx = gy * w_feat + gx
            
            if center_idx < Nq:
                q_indices.append(center_idx)
                
            if len(q_indices) == 0:
                 print("Warning: Object center maps to invalid feature index.")
                 return None, None
        else:
            # Fallback to image center if no query_box
            cx_pixel = W / 2.0
            cy_pixel = H / 2.0
            cv2.circle(vis_curr, (int(cx_pixel), int(cy_pixel)), 3, (0, 255, 0), -1)

            q_lb_x = cx_pixel * ratio + pad_w
            q_lb_y = cy_pixel * ratio + pad_h
            
            gx = int(q_lb_x / model_stride_w)
            gy = int(q_lb_y / model_stride_h)
            
            gx = min(max(gx, 0), w_feat - 1)
            gy = min(max(gy, 0), h_feat - 1)
            
            center_idx = gy * w_feat + gx
            if center_idx < Nq:
                q_indices.append(center_idx)


        if not q_indices:
            return None, None
            
        # Vectorized Attention Calculation
        q_idx_tensor = torch.tensor(q_indices, device=q_tensor.device)
        q_vecs = torch.index_select(q_tensor, 2, q_idx_tensor) # [B, H, N_samples, D]
        
        attn_logits = torch.matmul(q_vecs, k_tensor.transpose(-2, -1)) * scale
        attn_weights = attn_logits.softmax(dim=-1)
        
        # Average over heads -> [N_samples, Nk]
        attn_dist = attn_weights.mean(dim=1).squeeze(0) 
        
        # Get Top-5 Target for each sample
        vals, indices = attn_dist.topk(5, dim=-1)
        # Flatten for processing: (N_samples, 5) -> (N_samples * 5)
        # We need to keep track of rank: (Sample 0 Rank 0, Sample 0 Rank 1... Sample 9 Rank 4)
        vals_flat = vals.flatten().cpu().numpy()
        indices_flat = indices.flatten().cpu().numpy()
        
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
        print(f"Visualizing Heatmap Overlay (Score-Weighted) to EACH aligned history frame (Buffer: {num_available_imgs} / Memory: {num_memory_indices} frames):")
        
        def draw_marker(img, pt, size, color, shape='circle'):
            x, y = pt
            if shape == 'triangle':
                p1 = (x, y - size)
                p2 = (x - size, y + size)
                p3 = (x + size, y + size)
                cv2.drawContours(img, [np.array([p1, p2, p3])], 0, color, -1)
                cv2.drawContours(img, [np.array([p1, p2, p3])], 0, (0,0,0), 1)
            else:
                cv2.circle(img, (x, y), size, color, -1)

        # Iterate over each memory frame explicitly
        for f_idx in range(len(offsets) - 1):
            start_k = offsets[f_idx]
            end_k = offsets[f_idx+1]
            
            # Slice attention: [N_samples, K_frame]
            attn_slice = attn_dist[:, start_k:end_k]
            
            # 1. Aggregate Score: Find the "Best Score having any sample"
            # Max pooling across samples says "If *any* sample liked this key strongly, it's hot."
            scores, _ = attn_slice.max(dim=0) # [K_frame]
            scores = scores.cpu().numpy()
            
            if len(scores) == 0: continue

            # 2. Select Top-K Features
            # Instead of a heatmap, we visualize discrete points of attention.
            top_k = 256 
            if len(scores) > top_k:
                top_args = np.argsort(scores)[-top_k:]
                sorted_scores = scores[top_args]
                sorted_indices = np.atleast_1d(relevant_indices[f_idx][0].cpu().numpy())[top_args]
            else:
                top_args = np.argsort(scores)
                sorted_scores = scores[top_args]
                sorted_indices = np.atleast_1d(relevant_indices[f_idx][0].cpu().numpy())[top_args]
            
            # 3. Prepare Colors (Jet Colormap)
            # Map score rank to color: Low (Blue) -> High (Red)
            # We use rank-based coloring or score-based? 
            # Score-based is better if we normalize by max.
            # But rank-based ensures visibility even if distribution is peaked.
            # Let's use simple Min-Max normalization of the Top-K scores for coloring.
            
            if len(sorted_scores) > 0:
                s_min = sorted_scores.min()
                s_max = sorted_scores.max()
                if s_max > s_min:
                    norm_scores = (sorted_scores - s_min) / (s_max - s_min)
                else:
                    norm_scores = np.zeros_like(sorted_scores)
                
                # Map 0..1 to 0..255 for colormap
                color_indices = (norm_scores * 255).astype(np.uint8)
                # Apply colormap to a 1D strip
                colors_bgr = cv2.applyColorMap(color_indices.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
            
            # 4. Draw Markers
            # Resize Logic: We need to map feature grid coords -> Original Image Coords
            # We use the same rigorous logic as the Red Triangle.
            
            img_idx = num_available_imgs - num_memory_indices + f_idx
            if img_idx < 0 or img_idx >= num_available_imgs:
                continue
            
            # Draw on a copy first? No, draw directly on the frame copy in the list
            # We need to make sure we don't overwrite the same image if reused?
            # vis_mems_unaligned is a list of copies, so safe.
            target_img = vis_mems_unaligned[img_idx]
            
            for i, idx in enumerate(sorted_indices):
                # 1. Feature Grid -> Input Image Coords (with Padding)
                gy = idx // w_feat
                gx = idx % w_feat
                
                # Center of the grid cell
                y_input = (gy + 0.5) * model_stride_h
                x_input = (gx + 0.5) * model_stride_w
                
                # 2. Input Image -> Original Image Coords (Remove Padding)
                # x_real = (x_input - pad_w) / ratio
                # y_real = (y_input - pad_h) / ratio
                
                x_real = (x_input - pad_w) / ratio
                y_real = (y_input - pad_h) / ratio
                
                pt = (int(x_real), int(y_real))
                
                # Color from our map
                color = tuple(int(c) for c in colors_bgr[i])
                
                # Draw small circle
                # Larger for higher scores?
                radius = 2 if i < (len(sorted_indices) - 10) else 4
                cv2.circle(target_img, pt, radius, color, -1)
                
                # Draw Triangle for the absolute Top-1 (Last one)
                if i == len(sorted_indices) - 1:
                     draw_marker(target_img, pt, 8, (0, 0, 255), shape='triangle')
                     
            # No overlay blending needed, we drew directly on the image.

        return vis_curr, vis_mems_unaligned

        # Original global logic removed/replaced by above frame-wise logic
        """
        for i, (g_idx, val) in enumerate(zip(indices_flat, vals_flat)):
             ...
        """
        
        return vis_curr, vis_mems_unaligned

