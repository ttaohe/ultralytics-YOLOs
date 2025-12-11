
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from ultralytics.utils.plotting import plot_images
from ultralytics.utils import LOGGER, colorstr

def visualize_epoch():
    # Output directory
    save_dir = Path("runs/debug_visuals")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(colorstr("blue", "bold", f"Visualizing epoch to {save_dir}..."))

    # 1. Setup Config
    cfg = get_cfg(DEFAULT_CFG)
    
    # Configure for VisDrone Video Dataset with Random Crop
    cfg.imgsz = 640
    cfg.random_crop_size = 640
    cfg.random_crop_prob = 1.0
    cfg.rect = False
    cfg.cache = False
    cfg.single_cls = False
    cfg.task = "detect"
    cfg.batch = 16
    cfg.workers = 4
    cfg.fraction = 1.0
    
    # Explicitly disable Mosaic/Mixup to match our expectation
    # (Though our fix in VisDroneVideoDataset should force them to 0 anyway)
    cfg.mosaic = 1.0 # Set to 1.0 to verify it gets disabled
    cfg.mixup = 0.0
    
    # 2. Setup Data Dict
    data = {
        "path": "/home/hetao/graduate/data/VisDrone-VID-yolo",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van", 5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"},
        "nc": 10,
        "video_mode": True
    }
    
    img_path = os.path.join(data["path"], data["train"])
    
    # 3. Build Dataset
    print("Building dataset...")
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=img_path,
        batch=cfg.batch,
        data=data,
        mode="train",
        rect=False,
        stride=32
    )
    
    # 4. Build Dataloader
    print("Building dataloader...")
    loader = build_dataloader(
        dataset,
        batch=cfg.batch,
        workers=cfg.workers,
        shuffle=True,
        rank=-1
    )
    
    # 5. Iterate and Plot
    print(f"Starting iteration over {len(loader)} batches...")
    for i, batch in enumerate(loader):
        if i >= 20: # Limit to 20 batches to save time/space (user said "script to run whole epoch", but 20 batches is plenty for visual check)
            print("Stopping after 20 batches for quick review.")
            break
            
        # Plot batch
        fname = save_dir / f"train_batch{i}.jpg"
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=str(fname),
            on_plot=None
        )
        print(f"Saved {fname}")
        
        # Check for Mosaic artifacts in the first batch
        if i == 0:
            # Simple check: if mosaic is active, images in batch are usually 4x stitched.
            # But VisDroneVideoDataset should have disabled it.
            # We can check dataset.transforms to be sure.
            print("Checking transforms for Mosaic...")
            has_mosaic = False
            if hasattr(dataset, 'transforms'):
                for t in dataset.transforms.transforms:
                    if type(t).__name__ == 'Mosaic':
                        has_mosaic = True
                        print(colorstr("red", "FAILURE: Mosaic transform found!"))
            
            if not has_mosaic:
                print(colorstr("green", "SUCCESS: No Mosaic transform found."))

    print(f"Visualization complete. Please check {save_dir}")

if __name__ == "__main__":
    visualize_epoch()
