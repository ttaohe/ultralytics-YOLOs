
import os
import sys
import torch
from ultralytics.models.yolo.video.train import SAM2VideoTrainer

# Setup environment to mimic training
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"

def verify():
    print("Initializing SAM2VideoTrainer to verify augmentations...")
    
    # Args copied from train_video_sparse.py (lines 19-50)
    args = dict(
        model='ultralytics/cfg/models/12/yolo12-video-sparse-p3.yaml', 
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=100,
        imgsz=640,
        batch=4,   
        project='runs/train-video-sparse',
        name='verify_augmentations',
        device='cpu', # Use CPU for verification to avoid claiming GPU memory just for config check
        workers=0, # Avoid muliprocessing issues in script
        use_homography=False,
        random_crop_size=640,
        random_crop_prob=1.0,
        val_imgsz=1920,
        
        # Augmentation Params
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=0.0,
        mixup=0.0,
        translate=0.0,
        scale=0.0,
        
        # Explicitly passing other defaults just in case they matter
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5, # Default is usually 0.5, checking if it is on
        flipud=0.0,
    )
    
    # Create trainer
    trainer = SAM2VideoTrainer(overrides=args)
    
    # Initialize model to get stride
    # Force model loading
    print("Loading model to get stride...")
    trainer.model = trainer.get_model(cfg=args['model'], weights=None, verbose=False)
    
    # Manually trigger dataset build
    # Using the real train path found in YAML
    train_path = "/home/hetao/graduate/data/VisDrone-VID-yolo-cp/images/train"
    
    print(f"Building dataset from {train_path}...")
    # Mode='train' triggers augmentation build
    dataset = trainer.build_dataset(img_path=train_path, mode="train", batch=4)
    
    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    
    print(f"Dataset Class: {type(dataset).__name__}")
    
    # Check Random Crop (Handled by VisDroneVideoDataset manually)
    print(f"\n[Manual Augmentations in VisDroneVideoDataset]")
    print(f"  • random_crop_size: {getattr(dataset, 'random_crop_size', 'N/A')}")
    print(f"  • random_crop_prob: {getattr(dataset, 'random_crop_prob', 'N/A')}")
    print(f"  • use_homography:   {getattr(dataset, 'use_homography', 'N/A')}")
    
    # Check Transforms (Built by YOLODataset)
    print(f"\n[Standard Augmentations in YOLODataset.transforms]")
    if hasattr(dataset, 'transforms') and dataset.transforms is not None:
        transforms = dataset.transforms
        print(f"  Compose Object: {transforms}")
        
        if hasattr(transforms, 'transforms'):
            for i, t in enumerate(transforms.transforms):
                name = type(t).__name__
                print(f"  {i+1}. {name}")
                if name == 'Mosaic':
                    print(f"     - prob: {t.p}")
                elif name == 'MixUp':
                    print(f"     - prob: {t.p}")
                elif name == 'RandomPerspective':
                    print(f"     - degrees: {t.degrees}")
                    print(f"     - translate: {t.translate}")
                    print(f"     - scale: {t.scale}")
                    print(f"     - shear: {t.shear}")
                    print(f"     - perspective: {t.perspective}")
                elif name == 'RandomHSV':
                    print(f"     - details: {t.__dict__}")
                elif name == 'Albumentations':
                    print(f"     - p: {t.p}")
                # Flip is often handled specially or embedded? 
                # In v8_transforms, FlipLR is just a generic function/class? 
                # Let's check attributes if possible.
                
    else:
        print("  dataset.transforms is None or missing!")

    # Check Flip (Stored in dataset attributes usually by YOLODataset logic before passing to transforms? 
    # Actually YOLODataset usually handles flip inside v8_transforms)
    # But VisDroneVideoDataset explicitly disabled it in __init__:
    # self.fliplr = 0.0
    
    print(f"\n[Flip Config in Dataset Instance]")
    print(f"  • fliplr: {getattr(dataset, 'fliplr', 'N/A')}")
    print(f"  • flipud: {getattr(dataset, 'flipud', 'N/A')}")
    
    # Check Hyp
    print(f"\n[Hyperparameters (dataset.hyp)]")
    if hasattr(dataset, 'hyp'):
        hyp = dataset.hyp
        # If it's a dict
        if isinstance(hyp, dict):
            print(f"  • mosaic: {hyp.get('mosaic')}")
            print(f"  • mixup:  {hyp.get('mixup')}")
            print(f"  • scale:  {hyp.get('scale')}")
            print(f"  • translate: {hyp.get('translate')}")
        else: # Namespace
             print(f"  • mosaic: {getattr(hyp, 'mosaic', 'N/A')}")
             print(f"  • mixup:  {getattr(hyp, 'mixup', 'N/A')}")
             print(f"  • scale:  {getattr(hyp, 'scale', 'N/A')}")
             print(f"  • translate: {getattr(hyp, 'translate', 'N/A')}")


if __name__ == "__main__":
    verify()
