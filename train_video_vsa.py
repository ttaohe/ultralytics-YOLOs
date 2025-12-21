
import os
# 禁用 OpenCV 的 OpenCL 以避免在错误的 GPU 上创建上下文
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
import cv2
cv2.setUseOptimized(False)

import matplotlib
matplotlib.use('Agg')

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
from ultralytics.models.yolo.video.vsa_train import VSAVideoTrainer

warnings.filterwarnings("ignore")
print(f"[LAUNCH] train_video_vsa.py pid={os.getpid()}")

def train():
    args = dict(
        model='ultralytics/cfg/models/v10/yolov10l-video-vsa.yaml',  # VSA Config
        data='ultralytics/cfg/datasets/VisDrone-vid-masked.yaml',   
        epochs=100,
        imgsz=640,
        
        # VSA Long Training Config
        batch=2,        # Global Batch=2 (1 per GPU) for T-frame clips
        time_steps=8,   # Number of frames per clip
        vid_stride=5,   # Sparse sampling: every 5th frame (~20% sampling)
                        # Temporal span = 8 * 5 = 40 frames of history
        
        project='runs/train-video-vsa',
        name='yolo10-vsa-T8-stride5',
        device='0,3',   # Dual GPU usage
        workers=2,
        
        # Augmentation (Mosaic/Mixup disabled for VSA spatial consistency)
        mosaic=0.0,
        mixup=0.0,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Random Crop (High-Res) - REQUIRED for small object detection
        random_crop_size=640,
        random_crop_prob=1.0,  # Always apply random crop
        
        # Validation at high resolution for small objects
        val_imgsz=1920,
        
        # Optimize
        amp=True,
    )
    
    # Initialize VSA Trainer
    trainer = VSAVideoTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    train()
