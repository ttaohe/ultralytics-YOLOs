
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from ultralytics.nn.modules.video_attention import SparseMemoryAttention, GlobalRoPEAttention, apply_global_rotary_enc
from ultralytics.models.sam.modules.utils import compute_global_cis

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

    def patched_select_topk_features(self, features, coords, query=None, k_ratio=None):
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
        return self.original_select_topk(features, coords, query=query)

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
            return None, None, None
            
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
        # REFACTOR: Use the SNAP-TO-GRID Center Point (Anchor Point) instead of Regression Center.
        # This aligns visualization with the actual feature token responsible for prediction.
        q_indices = []
        
        if query_box is not None:
            # query_box is (cx_norm, cy_norm, w_norm, h_norm)
            cx_norm, cy_norm, w_norm, h_norm = query_box
            
            # 1. Convert normalized predictions to Input Image Space (Letterboxed)
            # Coordinates in the model's input tensor (e.g. 640x640)
            cx_input = cx_norm * W * ratio + pad_w
            cy_input = cy_norm * H * ratio + pad_h
            
            # 2. Snap to Grid (Anchor Point)
            # Round to nearest stride center
            gx = int(cx_input / model_stride_w)
            gy = int(cy_input / model_stride_h)
            
            # Clamp to valid grid range
            gx = min(max(gx, 0), w_feat - 1)
            gy = min(max(gy, 0), h_feat - 1)
            
            # 3. Calculate Snapped Pixel Coordinates (for Visualization)
            # Re-project grid center back to visual image space
            snapped_cx_input = (gx + 0.5) * model_stride_w
            snapped_cy_input = (gy + 0.5) * model_stride_h
            
            # DEBUG: Print Snap Details
            print(f"DEBUG: Box Center (Input): ({cx_input:.2f}, {cy_input:.2f}) -> Grid ({gx}, {gy}) -> Anchor Point: ({snapped_cx_input:.2f}, {snapped_cy_input:.2f})")
            print(f"DEBUG: Snap Delta: ({snapped_cx_input - cx_input:.2f}, {snapped_cy_input - cy_input:.2f}) pixels")
            
            snapped_cx_real = (snapped_cx_input - pad_w) / ratio
            snapped_cy_real = (snapped_cy_input - pad_h) / ratio
            
            # Draw Green Dot at the Snapped Anchor Point
            cv2.circle(vis_curr, (int(snapped_cx_real), int(snapped_cy_real)), 4, (0, 255, 0), -1)
            
            # Flatten index for Attention Query
            center_idx = gy * w_feat + gx
            
            if center_idx < Nq:
                q_indices.append(center_idx)
                
            if len(q_indices) == 0:
                 print("Warning: Object center maps to invalid feature index.")
                 return None, None, None
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
            return None, None, None
            
        # 3. RoPE A/B Test Calculation
        q_idx_tensor = torch.tensor(q_indices, device=q_tensor.device)
        q_vecs_raw = torch.index_select(q_tensor, 2, q_idx_tensor) # [B, H, N_samples, D] (Raw Content)
        
        # --- A. Content-Based Attention (No RoPE) ---
        attn_logits_noch = torch.matmul(q_vecs_raw, k_tensor.transpose(-2, -1)) * scale
        attn_weights_noch = attn_logits_noch.softmax(dim=-1)
        attn_dist_noch = attn_weights_noch.mean(dim=1).squeeze(0) # [N_samples, Nk]
        
        # --- B. Real Model Attention (With RoPE) ---
        # We need to manually apply RoPE to Q and K
        if 'q_coords' in context and context['q_coords'] is not None and 'k_coords' in context and context['k_coords'] is not None:
             q_coords_all = context['q_coords']
             k_coords_all = context['k_coords']
             
             # Compute Freqs
             dim = q_tensor.shape[-1]
             # Assuming single head or head_dim
             # q_tensor is [B, H, N, D]. RoPE expects last dim to be D.
             # D is head_dim.
             
             freqs_cis_q = compute_global_cis(q_coords_all[..., 0], q_coords_all[..., 1], dim, 10000.0).to(q_tensor.device)
             freqs_cis_k = compute_global_cis(k_coords_all[..., 0], k_coords_all[..., 1], dim, 10000.0).to(k_tensor.device)
             
             # Apply Global RoPE to BOTH Q and K in a single call
             # Note: apply_global_rotary_enc expects (B, H, N, D) tensors and
             # corresponding (B, N, D/2) complex freqs for each.
             q_rope, k_rope = apply_global_rotary_enc(
                 q_tensor,
                 k_tensor,
                 freqs_cis_q,
                 freqs_cis_k
             )
             
             # Now select Q vectors
             q_vecs_rope = torch.index_select(q_rope, 2, q_idx_tensor)
             
             attn_logits_rope = torch.matmul(q_vecs_rope, k_rope.transpose(-2, -1)) * scale
             attn_weights_rope = attn_logits_rope.softmax(dim=-1)
             attn_dist_rope = attn_weights_rope.mean(dim=1).squeeze(0)
        else:
             print("Warning: RoPE coordinates missing. Fallback to No-RoPE for both.")
             attn_dist_rope = attn_dist_noch


        # Helper to process distribution (kept for potential future use)
        def get_top_scored_indices(attn_dist, available_indices):
             # attn_dist: [N_samples, Nk] (we take max over samples)
             scores, _ = attn_dist.max(dim=0) # [K_frame]
             scores = scores.cpu().numpy()
             if len(scores) == 0:
                 return [], []
             
             top_k = 64 # Show top 64 points per frame
             if len(scores) > top_k:
                top_args = np.argsort(scores)[-top_k:]
             else:
                top_args = np.argsort(scores)
             
             sorted_scores = scores[top_args]
             sorted_indices = np.atleast_1d(available_indices[0].cpu().numpy())[top_args]
             
             return sorted_indices, sorted_scores

        
        # --- Memory Alignment Fix ---
        if not hasattr(self.target_module, '_viz_indices_list'):
             print("No memory indices list found.")
             return None, None, None
             
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
            if Nk == 0:
                return vis_curr, [], []
            print(f"Warning: Could not align indices (Nk={Nk}).")
            return None, None, None
        
        num_memory_indices = len(relevant_indices)
        num_available_imgs = len(memory_frames_imgs)
        
        # Separate visualizations for No-RoPE and With-RoPE
        vis_mems_no_rope = [img.copy() for img in memory_frames_imgs]
        vis_mems_rope = [img.copy() for img in memory_frames_imgs]
        
        # Calculate Offsets for aligned indices (MUST use full relevant_indices for correct Key addressing)
        k_sizes = [ind.shape[1] for ind in relevant_indices]
        offsets = [0] + list(np.cumsum(k_sizes))
        
        # Draw Results
        print(f"Visualizing RoPE A/B Test (Red=With RoPE, Blue=No RoPE, brightness ~ score):")
        
        def draw_marker(img, pt, size, color, shape='circle'):
            x, y = pt
            if shape == 'triangle':
                p1 = (x, y - size)
                p2 = (x - size, y + size)
                p3 = (x + size, y + size)
                cv2.drawContours(img, [np.array([p1, p2, p3])], 0, color, -1)
                # cv2.drawContours(img, [np.array([p1, p2, p3])], 0, (255,255,255), 1)
            elif shape == 'cross':
                 # Draw X
                 cv2.line(img, (x-size, y-size), (x+size, y+size), color, 2)
                 cv2.line(img, (x-size, y+size), (x+size, y-size), color, 2)
            else:
                cv2.circle(img, (x, y), size, color, -1)

        # Iterate over each memory frame explicitly
        for f_idx in range(len(offsets) - 1):
            start_k = offsets[f_idx]
            end_k = offsets[f_idx+1]
            
            # Slice attention: [N_samples, K_frame]
            attn_slice_noch = attn_dist_noch[:, start_k:end_k]
            attn_slice_rope = attn_dist_rope[:, start_k:end_k]
            
            curr_indices = relevant_indices[f_idx] # [B, K_frame] (Indices of sparse tokens in that frame)
            
            # ------------------------------------------------------------------
            # 1) 构建 dense heatmap：每个 sparse token 都参与，得到网格上的连续热力图
            # ------------------------------------------------------------------
            scores_no_all, _ = attn_slice_noch.max(dim=0)   # [K_frame]
            scores_rope_all, _ = attn_slice_rope.max(dim=0) # [K_frame]
            scores_no_all = scores_no_all.cpu().numpy()
            scores_rope_all = scores_rope_all.cpu().numpy()

            # 初始化特征网格尺度上的热力图 (h_feat, w_feat)
            heat_no = np.zeros((h_feat, w_feat), dtype=np.float32)
            heat_rope = np.zeros((h_feat, w_feat), dtype=np.float32)

            K_frame = curr_indices.shape[1]
            for local_idx in range(K_frame):
                grid_idx = curr_indices[0, local_idx].item()
                gy = grid_idx // w_feat
                gx = grid_idx % w_feat
                if gy < 0 or gy >= h_feat or gx < 0 or gx >= w_feat:
                    continue
                # 使用 max 聚合，避免同一位置被多个 token 覆盖时丢失峰值
                heat_no[gy, gx] = max(heat_no[gy, gx], float(scores_no_all[local_idx]))
                heat_rope[gy, gx] = max(heat_rope[gy, gx], float(scores_rope_all[local_idx]))

            def normalize_heatmap(h):
                h_max = h.max()
                h_min = h.min()
                if h_max - h_min < 1e-6:
                    return np.zeros_like(h)
                return (h - h_min) / (h_max - h_min + 1e-6)

            heat_no = normalize_heatmap(heat_no)
            heat_rope = normalize_heatmap(heat_rope)
            
            img_idx = num_available_imgs - num_memory_indices + f_idx
            if img_idx < 0 or img_idx >= num_available_imgs:
                continue
            
            target_img_no = vis_mems_no_rope[img_idx]
            target_img_rope = vis_mems_rope[img_idx]

            Hm, Wm = target_img_no.shape[:2]

            # 将热力图插值到图像分辨率
            heat_no_resized = cv2.resize(heat_no, (Wm, Hm), interpolation=cv2.INTER_CUBIC)
            heat_rope_resized = cv2.resize(heat_rope, (Wm, Hm), interpolation=cv2.INTER_CUBIC)

            heat_no_uint8 = (heat_no_resized * 255).astype(np.uint8)
            heat_rope_uint8 = (heat_rope_resized * 255).astype(np.uint8)

            heat_no_color = cv2.applyColorMap(heat_no_uint8, cv2.COLORMAP_JET)
            heat_rope_color = cv2.applyColorMap(heat_rope_uint8, cv2.COLORMAP_JET)

            # 叠加到原始帧上，得到 dense heatmap 风格的可视化
            alpha = 0.6
            cv2.addWeighted(heat_no_color, alpha, target_img_no, 1 - alpha, 0, dst=target_img_no)
            cv2.addWeighted(heat_rope_color, alpha, target_img_rope, 1 - alpha, 0, dst=target_img_rope)

            # 不再叠加稀疏散点，只保留连续的彩色热力图，便于整体 pattern 观察

        return vis_curr, vis_mems_no_rope, vis_mems_rope

        # Original global logic removed/replaced by above frame-wise logic
        """
        for i, (g_idx, val) in enumerate(zip(indices_flat, vals_flat)):
             ...
        """
        
        return vis_curr, vis_mems_no_rope, vis_mems_rope

