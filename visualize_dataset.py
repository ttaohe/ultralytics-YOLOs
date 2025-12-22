
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from ultralytics.data.video_dataset import VisDroneVideoDataset

def visualize_samples():
    """Visualize samples from the dataset to check alignment."""
    
    # Configuration
    DATASET_PATH = "/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/datasets/VisDrone-vid.yaml"
    VID_STRIDE = 5
    RANDOM_CROP_SIZE = 640
    RANDOM_CROP_PROB = 1.0 # Enable random crop to see if it causes issues
    VISUALIZE_COUNT = 10
    OUTPUT_DIR = "visualization_debug"
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load dataset config
    with open(DATASET_PATH) as f:
        data_cfg = yaml.safe_load(f)
        
    print(f"Initializing Dataset with vid_stride={VID_STRIDE}, random_crop_prob={RANDOM_CROP_PROB}...")
    
    # Initialize Dataset
    import os
    img_dir = os.path.join(data_cfg['path'], data_cfg['train'])
    dataset = VisDroneVideoDataset(
        img_path=str(img_dir),
        data=data_cfg,
        task="detect",
        vid_stride=VID_STRIDE,
        random_crop_size=RANDOM_CROP_SIZE,
        random_crop_prob=RANDOM_CROP_PROB,
        augment=True,
    )
    
    print(f"Dataset loaded. Total images: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Raw label[0] keys: {dataset.labels[0].keys()}")
        if 'instances' in dataset.labels[0]:
             print(f"Raw label[0] has instances: {dataset.labels[0]['instances']}")
    
    for i in range(min(VISUALIZE_COUNT, len(dataset))):
        print(f"Visualizing sample {i}...")
        try:
            sample = dataset.get_image_and_label(i)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
            
        # Get Image (First 3 channels if stacked)
        img = sample['img']
        if isinstance(img, np.ndarray):
            # If (H, W, 6), take first 3
            if img.ndim == 3 and img.shape[2] == 6:
                img = img[:, :, :3]
            # Ensure BGR
            # VisDroneVideoDataset loads as BGR via cv2 usually
        
        # Get BBoxes
        if 'instances' in sample:
             instances = sample['instances']
             # Ensure xyxy format for drawing
             instances.convert_bbox('xyxy')
             bboxes = instances.bboxes
             bbox_format = 'xyxy'
             normalized = instances.normalized
             print(f"  Bboxes from instances: {len(bboxes)} boxes, normalized={normalized}")
        elif 'bboxes' in sample:
             bboxes = sample['bboxes'] # Normalized or not? YOLODataset typically returns normalized xywh
             bbox_format = sample.get('bbox_format', 'xywh')
             normalized = sample.get('normalized', True)
        else:
             print("  No bboxes found.")
             bboxes = []
             bbox_format = 'xyxy'
             normalized = False
        
        h, w = img.shape[:2]
        print(f"  Image shape: {h}x{w}")
        
        # Draw
        img_vis = img.copy()
        for box in bboxes:
            # Convert to xyxy pixel coords
            if bbox_format == 'xywh':
                x, y, bw, bh = box
                if normalized:
                    x *= w
                    y *= h
                    bw *= w
                    bh *= h
                x1 = int(x - bw/2)
                y1 = int(y - bh/2)
                x2 = int(x + bw/2)
                y2 = int(y + bh/2)
            else: # xyxy
                x1, y1, x2, y2 = box
                if normalized:
                    x1 *= w
                    y1 *= h
                    x2 *= w
                    y2 *= h
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw crop window info if available
        crop_ori = sample.get('crop_window_ori')
        if crop_ori:
             cv2.putText(img_vis, f"CropOri: {crop_ori}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             
        # Save
        out_path = f"{OUTPUT_DIR}/sample_{i}.jpg"
        cv2.imwrite(out_path, img_vis)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    visualize_samples()
