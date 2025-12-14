
import os
import cv2
import torch
import numpy as np
from ultralytics.models.yolo.video.train import SAM2VideoTrainer
from ultralytics.utils import colorstr

# Setup environment
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"

def denormalize_image(img_tensor):
    """
    Convert normalized float tensor (C, H, W) to BGR numpy array (H, W, C) [0-255].
    Assumes standard YOLO format: float32 [0, 1].
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Check if it was normalized (values <= 1.0)
    if img_np.max() <= 1.05: # Allow small epsilon
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
        
    return np.ascontiguousarray(img_np) # HWC BGR

def run_debug():
    print(colorstr("blue", "bold", "Starting TargetMask Debug Script..."))
    
    # Output directory
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    args = dict(
        model='ultralytics/cfg/models/12/yolo12-video-sparse-p3.yaml', 
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=1,
        imgsz=640,
        batch=1,   # Batch 1 for easier per-image inspection
        project='runs/debug',
        name='debug_target_mask',
        device='cpu', 
        workers=0,
        
        # TargetMask Config - High ratio for visibility
        mask_ratio=0.8, 
        
        # Other Augs - Turn off Mosaic to make it cleaner? 
        # Actually want to see if it works WITH Mosaic.
        mosaic=1.0, 
        mixup=0.0,
        
        random_crop_size=640,
        random_crop_prob=1.0,
    )
    
    print(f"Configuring Trainer with mask_ratio={args['mask_ratio']}")
    trainer = SAM2VideoTrainer(overrides=args)
    
    # Load model to get stride (Required for build_dataset)
    print("Loading model to get stride...")
    trainer.model = trainer.get_model(cfg=args['model'], weights=None, verbose=False)
    
    # Build Dataset
    # We need to manually set training mode to True to enable augmentations
    train_path = "/home/hetao/graduate/data/VisDrone-VID-yolo-cp/images/train"
    print(f"Building dataset from {train_path}...")
    dataset = trainer.build_dataset(img_path=train_path, mode="train", batch=1)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Dataset Augment Status: {getattr(dataset, 'augment', 'UNKNOWN')}")
    print(f"Checking TargetMask status: {getattr(dataset, 'target_mask_aug', 'NOT FOUND')}")
    if hasattr(dataset, 'target_mask_aug'):
        print(f"  Ratio: {dataset.target_mask_aug.mask_ratio}")
        print(f"  P: {dataset.target_mask_aug.p}")
        # FORCE p=1.0 for Debugging
        print("  Forcing P=1.0 for debug sequence.")
        dataset.target_mask_aug.p = 1.0
        
    print("\nIterating through first 10 samples...")
    
    for i in range(10):
        # Determine index (random or sequential?)
        # Dataset access via index
        idx = i 
        
        # Load sample
        # Returns dict with 'img' (Stack), 'bboxes', 'cls'
        batch = dataset[idx]
        
        img_stack = batch['img'] # Tensor [6, H, W] or [3, H, W]
        print(f"\nSample {i}: Keys {batch.keys()}")
        print(f"Sample {i}: Img Shape {img_stack.shape}")
        
        # Helper to get bboxes
        if 'instances' in batch:
            bboxes = batch['instances'].bboxes
        elif 'bboxes' in batch:
            bboxes = batch['bboxes']
        else:
            print("No instances or bboxes found!")
            bboxes = []
            
        # Debug: Print bboxes stats
        if len(bboxes) > 0:
            if isinstance(bboxes, torch.Tensor):
                print(f"BBoxes stats: max={bboxes.max()}, min={bboxes.min()}")
                print(f"Sample bbox: {bboxes[0]}")
            else:
                print(f"BBoxes stats (Numpy): max={bboxes.max()}, min={bboxes.min()}")
        else:
            print("BBoxes empty")

        # Extract Current Frame (Channels 0-3)
        img_curr = img_stack[:3]
        img_hist = img_stack[3:] if img_stack.shape[0] > 3 else None
        
        # Denormalize Image
        viz_curr = denormalize_image(img_curr)
        h, w = viz_curr.shape[:2]
        
        # Check for mask color
        # Gray mask is 114 (in uint8)
        total_mask_pixels = 0
        
        # Prepare BBoxes for visualization (Denormalize if needed)
        # If 'bboxes' is tensor, it's typically normalized xywh or xyxy
        # Check values
        final_boxes = []
        if len(bboxes) > 0:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()
            
            # Check range to guess normalization
            if bboxes.max() <= 1.05:
                # Normalized, multiply by dims
                # Format is usually [cls, x, y, w, h] or [x, y, w, h]? 
                # In Dataset.__getitem__, 'bboxes' is usually [N, 4] (xywh normalized) 
                # AFTER Format transform.
                # Let's assume xywh normalized.
                # Need to convert to xyxy for drawing.
                # x_c, y_c, w_b, h_b = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
                # x1 = (x_c - w_b/2) * w
                # y1 = (y_c - h_b/2) * h
                # x2 = (x_c + w_b/2) * w
                # y2 = (y_c + h_b/2) * h
                
                for box in bboxes:
                    # Depending on exact format. 
                    # If it has 4 cols: xywh? xyxy?
                    # YOLODataset output 'bboxes' is (N, 4) normalized xywh.
                    xc, yc, wb, hb = box
                    x1 = (xc - wb/2) * w
                    y1 = (yc - hb/2) * h
                    x2 = (xc + wb/2) * w
                    y2 = (yc + hb/2) * h
                    final_boxes.append([x1, y1, x2, y2])
            else:
                # Unnormalized (likely xyxy or xywh), treat as pixels
                # But check if xywh or xyxy. 
                # Without knowing for sure, let's assume XYXY if max > 1, but YOLO 'Format' outputs normalized.
                # Let's just print finding.
                final_boxes.append(bboxes[0]) # Placeholder, fix inside loop
                pass # Fix logic below
        
        # Draw BBoxes
        for box in (final_boxes if final_boxes else bboxes if bboxes.max() > 1 else []):
            if len(final_boxes) == 0: # If we fell through
                 if len(box) == 4:
                    x1, y1, x2, y2 = box
                 else: 
                     continue
            else:
                x1, y1, x2, y2 = box
                
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Clip
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            # Draw rectangle (Green)
            cv2.rectangle(viz_curr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Check ROI for mask color
            roi = viz_curr[y1:y2, x1:x2]
            # Simple check: deviation from gray
            # Exact 114 might be affected by interpolation if resized? 
            # But TargetMask is applied AFTER resize/crop! 
            # So it should be exact 114 if no other color augs applied after.
            # Wait, Albumentations/HSV might run after?
            # VideoDataset: super().__init__() -> builds transforms.
            # v8_transforms pipeline: Mosaic -> CopyPaste -> RandomPerspective -> MixUp -> Albumentations -> HSV -> Flip -> Format
            # VideoDataset.__getitem__:
            # 1. super().__getitem__ (Runs ALL transforms including Format)
            # 2. TargetMask (Runs on Formatted Tensor/Numpy)
            # So TargetMask is the VERY LAST step before unstacking.
            # So pixel values should be EXACTLY what we set (modulo float conversion).
            
            # Check for gray pixels (114, 114, 114) with small tolerance
            mask_mask = np.all(np.abs(roi - 114) < 2, axis=-1)
            total_mask_pixels += np.sum(mask_mask)

        print(f"  Found {total_mask_pixels} gray pixels (value ~114) inside boxes.")
        
        # Save images
        # Current Frame
        out_path_curr = os.path.join(output_dir, f"sample_{i:02d}_curr.jpg")
        cv2.imwrite(out_path_curr, viz_curr)
        
        # History Frame (if exists)
        if img_hist is not None:
            viz_hist = denormalize_image(img_hist)
            out_path_hist = os.path.join(output_dir, f"sample_{i:02d}_hist.jpg")
            cv2.imwrite(out_path_hist, viz_hist)
            
    print(f"\nDebug finished. Check '{output_dir}' for visualizations.")

if __name__ == "__main__":
    run_debug()
