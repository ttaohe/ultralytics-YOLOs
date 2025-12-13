import os
# 禁用 OpenCV 的 OpenCL 以避免在错误的 GPU 上创建上下文
# os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
import cv2
# cv2.setUseOptimized(False)  # 可选：禁用特定优化以确保纯 CPU 运行

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")
print(f"[LAUNCH] train_yolo12n_baseline.py pid={os.getpid()}")

def train():
    # 使用相同的参数，但使用标准的 YOLOv8/12 Trainer (非 Video)
    args = dict(
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=100,
        imgsz=640, 
        batch=8,   
        project='runs/train-baseline',
        name='yolo12n-baseline',
        device='0',
        workers=4,  # Reduce workers to avoid "ancdata" error (file descriptor limit)
        # random_crop_size 默认等于 imgsz，如果原图足够大，直接从原图 crop（不 resize）
        # 如果原图不够大，会先 resize 到 imgsz
        random_crop_size=640,  # 显式指定，确保开启 High-Res Crop
        random_crop_prob=1.0,  # 1.0 = Super Mosaic Strategy

        
        # [EXPERIMENTAL] 验证时使用高分辨率 (1280)，训练时使用低分辨率 (640)
        val_imgsz=1920,
        
        # 保持与 Video 训练完全一致的增强参数
        mosaic=1.0, # 开启 Super Mosaic
        # mixup=0.0,
        scale=0.5, # 开启 Scale 缩放 (范围 0.5-1.5)
        # degrees=0.0,
        # translate=0.0,
        # shear=0.0,
        # perspective=0.0, 
        
        # 优化：开启 Baseline Mode，让 Dataset 跳过历史帧加载 (4x IO -> 1x IO)
        baseline_mode=True,

        # NMS Performance Tuning
        max_det= 100,  # Limit max detections to prevent NMS timeout during early training
        conf= 0.01,    # Raise conf threshold for validation to reduce candidate count
    )
    
    # 加载官方 YOLO12n 模型 (假设存在，或者使用 yolo12n.yaml)
    # 注意：如果 yolo12n.pt 不存在，会自动下载。
    # Use load('yolo12n.pt') to transfer weights to custom P2 architecture
    model = YOLO('/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/models/12/yolo12-baseline-p2.yaml').load('runs/train/train_yolov10l-p234_visroneDet_epoch300_imgsz800_batch4_mixup02_cutmix02/weights/best.pt')
    
    model.train(**args)

if __name__ == '__main__':
    train()

