import os
import cv2
import matplotlib
matplotlib.use('Agg')
import torch
import argparse
from ultralytics import YOLO

def validate():
    print(f"[LAUNCH] val_baseline_1920.py pid={os.getpid()}")
    
    # Load the best model
    model_path = 'runs/train-baseline-sparse_data/yolo12n-baseline6/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Validation arguments
    args = dict(
        data='ultralytics/cfg/datasets/VisDrone-vid-masked.yaml',
        imgsz=1920,      # Validation at 1920px as requested
        batch=1,         # Low batch size to avoid OOM at high res
        device='0',
        split='val',     # Use validation set
        project='runs/val-baseline-1920',
        name='yolo12n-baseline6-val1920',
        conf=0.25,      # Standard validation confidence (or 0.25 if user wants optimization) - using default usually 0.001 for mAP
        task='detect',
        baseline_mode=True, # Ensure dataloader skips history frames like training
    )
    
    print("Starting validation with args:", args)
    metrics = model.val(**args)
    
    print("\nValidation Complete")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    validate()
