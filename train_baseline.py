import os
# 禁用 OpenCV 的 OpenCL 以避免在错误的 GPU 上创建上下文
# os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
import cv2
# cv2.setUseOptimized(False)  # 可选：禁用特定优化以确保纯 CPU 运行

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
import argparse
from pathlib import Path
from ultralytics import YOLO

warnings.filterwarnings("ignore")
print(f"[LAUNCH] train_yolo12n_baseline.py pid={os.getpid()}")

def train():
    parser = argparse.ArgumentParser(description='Train YOLO Baseline Model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (e.g., runs/train-baseline/yolo12n-baseline/weights/last.pt)')
    opt = parser.parse_args()
    
    # 使用相同的参数，但使用标准的 YOLOv8/12 Trainer (非 Video)
    args = dict(
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=100,
        imgsz=640, 
        batch=8,   
        project='runs/train-baseline-sparse_data',
        name='yolo12n-baseline',
        device='0',
        workers=4,  # Reduce workers to avoid "ancdata" error (file descriptor limit)
        # random_crop_size 默认等于 imgsz，如果原图足够大，直接从原图 crop（不 resize）
        # 如果原图不够大，会先 resize 到 imgsz
        random_crop_size=640,  # 显式指定，确保开启 High-Res Crop
        random_crop_prob=1.0,  # 1.0 = Super Mosaic Strategy

        
        # [EXPERIMENTAL] 验证时使用高分辨率 (1280)，训练时使用低分辨率 (640)
        val_imgsz=1280,
        
        # 保持与 Video 训练完全一致的增强参数
        mosaic=1.0, # 开启 Super Mosaic
        mixup=0.5,
        scale=0.5, # 开启 Scale 缩放 (范围 0.5-1.5)
        # degrees=0.0,
        # translate=0.0,
        # shear=0.0,
        # perspective=0.0, 
        
        # 优化：开启 Baseline Mode，让 Dataset 跳过历史帧加载 (4x IO -> 1x IO)
        baseline_mode=True,

        # NMS Performance Tuning
        conf= 0.25,    # Raise conf threshold for validation to reduce candidate count
        vid_stride=5,  # Enable sparse sampling for baseline too
    )
    
    # Handle resume logic
    if opt.resume:
        resume_path = Path(opt.resume)
        if resume_path.exists():
            print(f"[RESUME] Resuming from: {resume_path}")
            model = YOLO(str(resume_path))
            # When resuming, call train with resume=True and don't pass conflicting args
            model.train(resume=True)
        else:
            print(f"[ERROR] Checkpoint not found: {resume_path}")
            return
    else:
        # Fresh training
        model = YOLO('/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/models/12/yolo12-baseline-p2.yaml').load('runs/train/train_yolov10l-p234_visroneDet_epoch300_imgsz800_batch4_mixup02_cutmix02/weights/best.pt')
        model.train(**args)

if __name__ == '__main__':
    train()

