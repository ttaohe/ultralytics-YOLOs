
import os
# ============ DDP DEBUG: 设置更长的超时时间和调试选项 ============
os.environ["NCCL_TIMEOUT"] = "7200"  # 2小时超时
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # 阻塞等待模式，方便调试
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"  # 显示 C++ 堆栈跟踪
# 可选：启用 NCCL 详细日志（会产生大量输出，需要时取消注释）
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
# ============================================================

# 禁用 OpenCV 的 OpenCL 以避免在错误的 GPU 上创建上下文
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
# 让 DataLoader 在 worker 卡住时不要无限等待（仅 workers>0 时生效，见 ultralytics/data/build.py）
os.environ.setdefault("ULTRALYTICS_DATALOADER_TIMEOUT", "120")

# 为每次训练生成一个 run_id，确保 /tmp 下的调试日志不会和历史 run 混在一起
# 注意：该环境变量会被 DDP 子进程继承，从而所有 rank 写到同一组 run_id 文件中
os.environ.setdefault("VSA_RUN_ID", str(os.getpid()))
import cv2
cv2.setUseOptimized(False)

import matplotlib
matplotlib.use('Agg')

import torch
# NOTE: set_sharing_strategy('file_system') can cause issues with DDP dataloader
# Only use this for single GPU training
# torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
from ultralytics.models.yolo.video.vsa_train import VSAVideoTrainer

warnings.filterwarnings("ignore")
print(f"[LAUNCH] train_video_vsa.py pid={os.getpid()}")

def train():
    # ============ Memory Optimization Options ============
    # Option 1: Use lite model (max_memory=16, global_topk=256) - RECOMMENDED
    # Option 2: Reduce time_steps (fewer frames per clip)
    # Option 3: Lower val_imgsz (validation resolution)
    # Option 4: Use gradient accumulation
    # =====================================================
    
    fast_debug = os.getenv("VSA_FAST_DEBUG", "0") == "1"

    args = dict(
        # === MODEL: Use LITE version for lower VRAM ===
        model='ultralytics/cfg/models/v10/yolov10l-video-vsa-lite.yaml',  # LITE: max_memory=16, topk=256
        # model='ultralytics/cfg/models/v10/yolov10l-video-vsa.yaml',  # FULL: max_memory=100, topk=512
        
        data='ultralytics/cfg/datasets/VisDrone-vid-masked.yaml',   
        epochs=2 if fast_debug else 100,
        imgsz=640,
        
        # === VSA Config (Memory-Optimized) ===
        batch=4,        # Global Batch=2 (1 per GPU)
        time_steps=4,   # REDUCED: 4 frames per clip (was 8)
                        # This halves memory for temporal processing!
        vid_stride=100,  # Fixed back to 10 (100 was too extreme, causing DDP issues)
        
        project='runs/train-video-vsa',
        name='yolo10-vsa-lite-T4-stride10',
        device='0,3',   # Use single GPU to avoid DDP deadlock issues
        workers=2,
        val=False if fast_debug else True,
        
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
        
        # === Validation: Lower resolution to save memory ===
        val_imgsz=1280,  # REDUCED from 1920 (saves ~40% VRAM during val)
        
        # Optimize
        amp=True,
        
        # === Optional: Gradient Accumulation ===
        # accumulate=4,  # Uncomment to accumulate 4 steps (effective batch=8)
    )
    
    # Initialize VSA Trainer
    trainer = VSAVideoTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    train()
