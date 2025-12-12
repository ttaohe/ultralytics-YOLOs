import os
# 禁用 OpenCV 的 OpenCL 以避免在错误的 GPU 上创建上下文
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
import cv2
cv2.setUseOptimized(False)  # 可选：禁用特定优化以确保纯 CPU 运行

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
from ultralytics.models.yolo.video.train import SAM2VideoTrainer

warnings.filterwarnings("ignore")
print(f"[LAUNCH] train_video.py pid={os.getpid()}")

def train():
    args = dict(
        model='ultralytics/cfg/models/12/yolo12-video-p2-early_memory_fusioin.yaml', 
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=100,
        imgsz=640,  # 目标输入尺寸，如果原图足够大，会直接从原图 crop 到该尺寸（保留小目标信息）
        batch=8,   
        project='runs/train-video',
        name='yolo12-sam2-video',
        device='0,3',
        workers=4,
        use_homography=False,
        random_crop_size=640,  # 显式指定，确保开启 High-Res Crop
        random_crop_prob=1.0,  # 100% 概率开启 Random Crop，结合 Mosaic 实现 Super Mosaic
        
        # [EXPERIMENTAL] 验证时使用高分辨率 (1920)，训练时使用低分辨率 (640)
        # 修复 Scale Mismatch 问题：验证时使用原图分辨率
        val_imgsz=1920,
        
        # 增强参数
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,    # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,    # image HSV-Value augmentation (fraction)
        mosaic=1.0,   # 开启 Super Mosaic (已通过 Channel Stacking 支持时序同步)
        # mixup=0.0,  # 保持关闭
        # translate=0.0, # 保持关闭，random_crop 已经提供了平移效果

        scale=0.5,    # 开启 Scale 缩放 (范围 0.5-1.5)
        # degrees=0.0,
        # translate=0.0,
        # shear=0.0,
        # perspective=0.0, 
        
        # NMS Performance Tuning
        max_det=100,  # Limit max detections
        conf=0.01,    # Raise validation conf threshold
    )
    
    trainer = SAM2VideoTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    train()

