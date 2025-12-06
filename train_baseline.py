import os
# 禁用 OpenCV 的 OpenCL 以避免在错误的 GPU 上创建上下文
# os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
import cv2
# cv2.setUseOptimized(False)  # 可选：禁用特定优化以确保纯 CPU 运行

import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")
print(f"[LAUNCH] train_yolo12n_baseline.py pid={os.getpid()}")

def train():
    # 使用相同的参数，但使用标准的 YOLOv8/12 Trainer (非 Video)
    args = dict(
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=100,
        imgsz=1280, 
        batch=4,   
        project='runs/train-baseline',
        name='yolo12n-baseline',
        device='0,3',
        workers=2,  # Reduce workers to avoid "ancdata" error (file descriptor limit)
        
        # 保持与 Video 训练完全一致的增强参数
        # mosaic=0.0,
        # mixup=0.0,
        # scale=0.0,
        # degrees=0.0,
        # translate=0.0,
        # shear=0.0,
        # perspective=0.0, 
    )
    
    # 加载官方 YOLO12n 模型 (假设存在，或者使用 yolo12n.yaml)
    # 注意：如果 yolo12n.pt 不存在，会自动下载。
    # 如果你想从 yaml 重新初始化，请使用 'yolo12n.yaml'
    model = YOLO('/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/models/v10/yolov10l-p2-no-p5.yaml') 
    
    model.train(**args)

if __name__ == '__main__':
    train()

