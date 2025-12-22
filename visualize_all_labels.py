
import cv2
import yaml
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.video_dataset import VisDroneVideoDataset

def visualize_all_labels():
    """Visualize ALL samples from the dataset labels."""
    
    # Configuration
    DATASET_CFG = "ultralytics/cfg/datasets/VisDrone-vid.yaml"
    OUTPUT_DIR = "runs/visualize/visdrone_all_labels"
    VID_STRIDE = 1  # Visualize every frame (Dense)
    MAX_SAMPLES = None # Set to None to visualize ALL, or int limit
    
    # Ensure output dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load dataset config
    with open(DATASET_CFG) as f:
        data_cfg = yaml.safe_load(f)
        
    print(f"Initializing Dataset from {DATASET_CFG}...")
    
    # Initialize Dataset (Use Val set? or Train? User said 'Visdrone-VID labels', likely Train)
    # Let's assume Train first.
    img_dir = os.path.join(data_cfg['path'], data_cfg['train'])
    
    dataset = VisDroneVideoDataset(
        img_path=str(img_dir),
        data=data_cfg,
        task="detect",
        # User implies checking labels, so raw checks are better.
        # But VisDroneVideoDataset defaults to augment=False if mode!='train' usually?
        # Let's set augment=False to see raw labels alignment.
        augment=False, 
        vid_stride=VID_STRIDE,
        random_crop_size=0, # No crop, full image
        random_crop_prob=0.0,
    )
    
    print(f"Dataset loaded. Total images: {len(dataset)}")
    
    limit = len(dataset)
    if MAX_SAMPLES is not None:
        limit = min(limit, MAX_SAMPLES)
        
    print(f"Visualizing {limit} samples to {OUTPUT_DIR}...")
    
    for i in tqdm(range(limit)):
        try:
            sample = dataset.get_image_and_label(i)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
            
        # Get Image
        img = sample['img']
        # If tensor, convert to numpy
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
             # Check if stacked (H, W, 6)
             if img.ndim == 3 and img.shape[2] == 6:
                 img = img[:, :, :3]
             # If float, convert
             if img.dtype == np.float32 or img.dtype == np.float64:
                 img = (img * 255).astype(np.uint8)
                 
        h, w = img.shape[:2]
        img_vis = img.copy()
        
        # Get BBoxes
        if 'instances' in sample:
             instances = sample['instances']
             instances.convert_bbox('xyxy')
             bboxes = instances.bboxes
             normalized = instances.normalized
             # Instances does not hold cls info, it's separate in sample['cls']
             cls = sample['cls'].flatten() if 'cls' in sample else np.zeros(len(bboxes))
        elif 'bboxes' in sample:
             bboxes = sample['bboxes']
             normalized = True # YOLODataset defaults
             cls = sample['cls']
        else:
             bboxes = []
             
        # Draw
        for j, box in enumerate(bboxes):
            if normalized:
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)
            else:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
            c = int(cls[j])
            color = ((c * 50) % 255, (c * 80) % 255, (c * 130) % 255)
                
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_vis, str(c), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Retrieve file name for meaningful output name
        im_file = Path(dataset.im_files[i])
        # Use parent folder name + file name to distinguish video sequences
        # e.g. uav0000123_00000/0000001.jpg -> uav0000123_00000_0000001.jpg
        out_name = f"{im_file.parent.name}_{im_file.name}"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        cv2.imwrite(out_path, img_vis)

    print(f"Done. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_all_labels()
