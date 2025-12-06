import os
# 禁用 OpenCV 的 OpenCL 以避免在错误的 GPU 上创建上下文
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
import cv2
cv2.setUseOptimized(False)  # 可选：禁用特定优化以确保纯 CPU 运行

import warnings
from ultralytics.models.yolo.video.train import SAM2VideoTrainer

warnings.filterwarnings("ignore")
print(f"[LAUNCH] train_video.py pid={os.getpid()}")

def train():
    args = dict(
        model='ultralytics/cfg/models/12/yolo12-video.yaml', 
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=100,
        imgsz=800, 
        batch=16,   
        project='runs/train-video',
        name='yolo12-sam2-video',
        device='0,3',
        workers=8,
        
        mosaic=0.0,
        mixup=0.0,
        scale=0.0,
        degrees=0.0,
        translate=0.0,
        shear=0.0,
        perspective=0.0, 
    )
    
    trainer = SAM2VideoTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    train()

