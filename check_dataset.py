import sys
import os
import yaml
import torch
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.cfg import IterableSimpleNamespace

def check_visdrone_loading():
    # 1. Load Data Config
    data_yaml_path = 'ultralytics/cfg/datasets/VisDrone-vid.yaml'
    print(f"Loading config from {data_yaml_path}")
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Resolve path
    # The yaml has 'path: ...' which might be absolute or relative.
    dataset_root = data.get('path')
    train_path = data.get('train')
    
    if not os.path.isabs(train_path):
        img_path = os.path.join(dataset_root, train_path)
    else:
        img_path = train_path
        
    print(f"Dataset root: {dataset_root}")
    print(f"Image path: {img_path}")

    # 2. Create Dummy Config
    cfg = IterableSimpleNamespace(
        imgsz=640,
        rect=False,
        cache=False, # Disable cache for debugging to ensure we load fresh
        single_cls=False,
        task='detect',
        classes=None,
        fraction=1.0,
        # Hyps for augmentation
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0, # Added
        copy_paste=0.0, # Added
        copy_paste_mode='flip', # Added
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        # Color/Mask hyps
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mask_ratio=4,
        overlap_mask=True,
        bgr=0.0,
        auto_augment=None,
        erasing=0.4,
        crop_fraction=1.0,
    )

    # 3. Build Dataset
    print("Building dataset...")
    try:
        dataset = build_yolo_dataset(
            cfg=cfg,
            img_path=img_path,
            batch=4,
            data=data,
            mode='train',
            rect=False,
            stride=32
        )
        print(f"Dataset built successfully. Class: {type(dataset).__name__}")
        print(f"Dataset Length: {len(dataset)}")
    except Exception as e:
        print(f"Error building dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Build DataLoader
    print("Building dataloader...")
    dataloader = build_dataloader(
        dataset,
        batch=4,
        workers=0, # Use 0 for simple debugging/trace
        shuffle=True, # Changed to True to simulate real training
        rank=-1
    )
    
    print("Starting iteration (checking first 10 batches)...")
    
    # 5. Iterate
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
            
        print(f"\n--- Batch {i} ---")
        if isinstance(batch, dict):
            # Print Image Files
            if 'im_file' in batch:
                print("Current Frames:")
                for idx, f in enumerate(batch['im_file']):
                     print(f"  [{idx}] {f}")
            
            if 'history_im_file' in batch:
                print("History Frames:")
                for idx, f in enumerate(batch['history_im_file']):
                     print(f"  [{idx}] {f}")

            # Check tensors
            if 'img' in batch:
                print(f"Current Image Batch Shape: {batch['img'].shape}")
            
            if 'history_img' in batch:
                hist_imgs = batch['history_img']
                if isinstance(hist_imgs, torch.Tensor):
                     print(f"History Image Batch Shape: {hist_imgs.shape}")
                elif isinstance(hist_imgs, (list, tuple)):
                     print(f"History Image Batch is List of length {len(hist_imgs)}")
                     if len(hist_imgs) > 0:
                         print(f"  Element 0 shape: {hist_imgs[0].shape}")

        else:
             print("Batch is not a dict", type(batch))

if __name__ == "__main__":
    check_visdrone_loading()

