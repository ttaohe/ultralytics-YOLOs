
import os
import torch
import numpy as np
import yaml
from ultralytics.data.video_dataset import VisDroneVideoDataset
# Mock augmentation config
from types import SimpleNamespace

def verify_stacking():
    print("Verifying VisDroneVideoDataset Stacking...")
    
    # Needs a real path. Let's use the one in train_video.py or VisDrone-vid.yaml
    # VisDrone-vid.yaml path: ultralytics/cfg/datasets/VisDrone-vid.yaml
    # We need to read it to get the path.
    # Assuming standard path structure: /home/hetao/data/visdrone/VisDrone2019-VID-train/sequences/...
    # Let's try to mock the dataset path if possible, or just use the class if I can find valid images.
    # Actually, simpler: Use `ultralytics/cfg/datasets/VisDrone-vid.yaml` and parse it.
    
    try:
        with open('ultralytics/cfg/datasets/VisDrone-vid.yaml') as f:
            data_cfg = yaml.safe_load(f)
        path = data_cfg.get('path', '')
        train_path = os.path.join(path, data_cfg.get('train', ''))
        # If absolute path is needed, we might need to adjust.
        # But let's assume the user has the data.
        
        print(f"Dataset path: {train_path}")
        
    except Exception as e:
        print(f"Skipping path check, using hardcoded path assumption: {e}")
        train_path = "/home/hetao/data/visdrone/VisDrone2019-VID-train/sequences"

    # Create dummy hyp with mosaic=1.0 to trigger augmentations
    hyp = SimpleNamespace(
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        cutmix=0.0, # Fix AttributeError
        copy_paste_mode='flip',
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mask_ratio=4,
        overlap_mask=True,
        bgr=0.0,
        box=0.05,
        cls=0.5,
    )

    # Initialize Dataset
    # We need to find at least one image file.
    # Let's search for one.
    # Or just try to init and if it fails due to no images, we know.
    try:
        ds = VisDroneVideoDataset(
            img_path=train_path,
            imgsz=640,
            batch_size=4,
            augment=True,
            hyp=hyp,
            rect=False,
            # check_images=False, # Invalid
            stride=32,
            random_crop_prob=1.0,
            random_crop_size=640, # Test High-Res Path
            data={'channels': 3} # Fix NoneType error
        )
        print(f"Dataset initialized. Found {len(ds)} images.")
        
        # Manually populate buffer to prevent Mosaic crash (IndexError in random.choices)
        # Mosaic typically uses self.dataset.buffer which is filled by the loader.
        # In this standalone script, we must fill it manually.
        if hasattr(ds, 'buffer'):
            import random
            # Fill buffer with random valid indices
            for _ in range(min(100, len(ds))):
                idx = random.randint(0, len(ds)-1)
                ds.buffer.append(idx)
        
        if len(ds) > 0:
             # Get one item
             item = ds[0]
             img = item['img']
             hist = item['history_img']
             
             print(f"Item 0 img shape: {img.shape}")
             print(f"Item 0 history shape: {hist.shape}")
             print(f"Item 0 img type: {type(img)}")
             
             if not isinstance(img, torch.Tensor):
                 print("FAILURE: img is not a Tensor!")
             if not isinstance(hist, torch.Tensor):
                 print("FAILURE: history_img is not a Tensor!")
                 
             if img.shape != (3, 640, 640):
                 print(f"FAILURE: img shape mismatch! Got {img.shape}")
             else:
                 print("SUCCESS: img shape is correct (3, 640, 640)")
                 
             if hist.shape != (3, 640, 640):
                 print(f"FAILURE: history_img shape mismatch! Got {hist.shape}")
             else:
                 print("SUCCESS: history_img shape is correct (3, 640, 640)")
                 
             # Check if they look like mosaic (optional, hard to verify)
             # Check homography
             H = item['homography']
             print(f"Homography: \n{H}")
             if H.shape == (3, 3):
                 print("SUCCESS: Homography shape correct.")
                 
    except Exception as e:
        print(f"Verification Failed with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_stacking()
