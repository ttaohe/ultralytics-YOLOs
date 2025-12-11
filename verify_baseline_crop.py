
import os
import sys
import torch
import numpy as np
from ultralytics.data.build import build_yolo_dataset
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from ultralytics.utils import colorstr

def verify_baseline():
    print(colorstr("blue", "bold", "Verifying Baseline Random Crop & Augmentations..."))
    
    # 1. Setup Config using Default
    cfg = get_cfg(DEFAULT_CFG)
    
    # Override with baseline settings
    cfg.imgsz = 640
    cfg.random_crop_size = 640
    cfg.random_crop_prob = 1.0
    cfg.rect = False
    cfg.cache = False
    cfg.single_cls = False
    cfg.task = "detect"
    cfg.fraction = 1.0
    
    # Enable all risky augmentations to test disable logic
    cfg.mosaic = 1.0
    cfg.mixup = 0.5
    cfg.copy_paste = 0.5
    # cutmix is not in DEFAULT_CFG usually, but let's try setting it if supported
    # Actually cutmix is usually handled via hyp dict if not in cfg object
    # But get_cfg returns a namespace that accepts new attributes? No, IterableSimpleNamespace might not.
    # Let's rely on what build_yolo_dataset does.
    
    # 2. Setup Data Dict (mimic VisDrone-vid.yaml)
    data = {
        "path": "/home/hetao/graduate/data/VisDrone-VID-yolo",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van", 5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"},
        "nc": 10,
        "video_mode": True # This triggers VisDroneVideoDataset
    }
    
    img_path = os.path.join(data["path"], data["train"])
    
    # 3. Build Dataset
    print(f"Building dataset with mosaic={cfg.mosaic}, mixup={cfg.mixup}, copy_paste={cfg.copy_paste}")
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=img_path,
        batch=4,
        data=data,
        mode="train",
        rect=False,
        stride=32
    )
    
    print(f"Dataset Class: {type(dataset).__name__}")
    
    # Check transforms
    if hasattr(dataset, 'transforms'):
        print("Checking transforms...")
        found_risky = False
        for t in dataset.transforms.transforms:
            name = type(t).__name__
            print(f"  - {name}")
            if name in ['Mosaic', 'MixUp', 'CopyPaste', 'CutMix']:
                print(colorstr("red", f"FAILURE: {name} transform is still present!"))
                found_risky = True
        
        if not found_risky:
            print(colorstr("green", "SUCCESS: All risky transforms (Mosaic, MixUp, CopyPaste, CutMix) are disabled."))
    
    # 4. Verify Crop
    print("Calling get_image_and_label(0)...")
    try:
        label = dataset.get_image_and_label(0)
        img = label['img']
        print(f"Returned image shape: {img.shape}")
        
        if img.shape[0] == 640 and img.shape[1] == 640:
            print("SUCCESS: Image cropped to 640x640")
        else:
            print(f"FAILURE: Image shape mismatch. Expected (640, 640), got {img.shape[:2]}")
            
    except Exception as e:
        print(f"Error during get_image_and_label: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_baseline()
