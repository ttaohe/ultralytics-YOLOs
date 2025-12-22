
import random
import numpy as np

class TargetMask:
    """
    Randomly masks a portion of the target object's bounding box in the current frame.
    This forces the model to rely on history frames (Cross-Attention) to recover features.
    
    Args:
        p (float): Probability of applying mask to a given object.
        mask_ratio (float): Maximum ratio of the bbox area to mask (0.0 - 1.0).
                            The mask size will be random up to this ratio.
    """
    def __init__(self, p=0.5, mask_ratio=0.6):
        self.p = p
        self.mask_ratio = mask_ratio

    def __call__(self, labels):
        """
        Apply mask to current frame (channels 0-3) based on GT bboxes.
        Assumes labels['img'] is a Stacked Tensor or Numpy array.
        """
        # print(f"TargetMask: __call__ invoked. P={self.p}")
        if random.random() > self.p:
            # print("TargetMask: Skipped (Global P)")
            return labels
        
        # print("TargetMask: Running")
            
        img = labels.get("img") # Expected (6, H, W) Tensor or (H, W, 6) Numpy
        instances = labels.get("instances")
        
        if img is None:
            return labels
            
        # Ensure we are working with Numpy HWC for easier manipulation
        # If Tensor [C, H, W], convert to Numpy temporarily?
        # Dataset transforms usually run on Numpy arrays BEFORE ToTensor.
        # Let's assume input is Numpy (H, W, C) from VisDroneVideoDataset.__getitem__
        
        is_tensor = False
        import torch
        if isinstance(img, torch.Tensor):
            is_tensor = True
            # [C, H, W] -> [H, W, C]
            img_np = img.permute(1, 2, 0).cpu().numpy().copy()
        else:
            img_np = img.copy() # (H, W, C)

        h_img, w_img, c_img = img_np.shape
        
        # Only mask Current Frame (Assumed to be first 3 channels? Or last 3?)
        # VisDroneVideoDataset stacks [Current, History] -> Current is 0:3
        # Wait, verify video_dataset.py:
        # stacked_img = np.concatenate((current_img, img_hist), axis=2) 
        # So channels 0,1,2 are Current. 3,4,5 are History.
        
        # We mask Current (0:3)
        
        # Try to get bboxes from 'instances' (Pre-Format) or 'bboxes' (Post-Format)
        if instances is not None:
             bboxes = instances.bboxes
        else:
             bboxes = labels.get("bboxes")
             
        if bboxes is None or len(bboxes) == 0:
            return labels
            
        # Convert to Numpy if Tensor
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            
        # Check Normalization & Format
        # If max <= 1.0, assume Normalized.
        # If Post-Format, it is usually XYWH.
        # We need Pixel XYXY for masking.
        
        is_normalized = bboxes.max() <= 1.05
        # Assumption: If None/Post-Format -> XYWH. If Instances -> XYXY.
        # But 'instances' object usually wraps raw boxes.
        # Safe heuristic: Check if normalized. If normalized -> likely XYWH (standard YOLO).
        
        pixel_boxes = []
        if is_normalized:
            for box in bboxes:
                # XYWH Normalized -> Pixel XYXY
                xc, yc, w, h = box
                x1 = (xc - w/2) * w_img
                y1 = (yc - h/2) * h_img
                x2 = (xc + w/2) * w_img
                y2 = (yc + h/2) * h_img
                pixel_boxes.append([x1, y1, x2, y2])
        else:
            # Assume Pixel XYXY (or XYWH?)
            # Usually pre-format is Pixel XYXY.
            # Let's assume Pixel XYXY.
            pixel_boxes = bboxes
            
        # print(f"TargetMask: Processing {len(pixel_boxes)} boxes. Normalized={is_normalized}")

        for i, box in enumerate(pixel_boxes):
            # Apply per-object probability inside this function? 
            # Or self.p applies to the whole image?
            # User said "randomly mask target object regions". 
            # Usually we want some objects masked, some not.
            if random.random() > 0.5: # Hardcoded 50% chance per object if global p passed
                 continue
                 
            # Box coordinates
            x1, y1, x2, y2 = box
            
            # Clip to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_img, int(x2))
            y2 = min(h_img, int(y2))
            
            bw = x2 - x1
            bh = y2 - y1
            
            # print(f"  Box {i}: {x1},{y1} {bw}x{bh}")
            
            if bw <= 0 or bh <= 0:
                continue
                
            # Random Mask Size
            # Mask Area <= box_area * mask_ratio
            # Let's define mask width/height ratio
            
            # Simple approach: Mask a sub-rect
            # mw = random.uniform(0.1, self.mask_ratio) * bw -> No, sqrt(ratio) for dims
            # Let's just say mask dim is ratio of box dim
            
            ratio_w = random.uniform(0.5, self.mask_ratio) # e.g. 0.5 to mask_ratio
            ratio_h = random.uniform(0.5, self.mask_ratio) 
            
            mw = int(bw * ratio_w)
            mh = int(bh * ratio_h)
            
            if mw > 0 and mh > 0:
                # Determine fill value based on image type/range
                # YOLO Format transform typically outputs Float32 in [0, 1]
                if np.issubdtype(img_np.dtype, np.floating) and img_np.max() <= 1.0:
                    fill_value = 114.0 / 255.0
                else:
                    fill_value = 114
                    
                # Random position of mask inside box
                mx = random.randint(x1, x2 - mw)
                my = random.randint(y1, y2 - mh)
                
                # print(f"    Masking {mx},{my} {mw}x{mh} with {fill_value}")
                
                # Fill with mean color
                # C channels: 0~3
                img_np[my:my+mh, mx:mx+mw, 0:3] = fill_value

        # Restore format
        if is_tensor:
            labels["img"] = torch.from_numpy(img_np).permute(2, 0, 1) # [C, H, W]
        else:
            labels["img"] = img_np
            
        return labels
