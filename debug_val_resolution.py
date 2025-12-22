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
print(f"[LAUNCH] debug_val_resolution.py pid={os.getpid()}")

def validate():
    args = dict(
        model='ultralytics/cfg/models/12/yolo12-video-p2-p5.yaml', 
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        imgsz=640,  # Train size
        batch=4,   
        project='runs/debug-val-resolution',
        name='yolo12-sam2-video-debug',
        device='0',
        workers=4,
        fraction=0.005,
        use_homography=False,
        random_crop_prob=0.8,
        
        # Validation resolution
        val_imgsz=1920,
        
        # Validation settings
        val=True,
        plots=True,
    )
    
    # Initialize trainer
    trainer = SAM2VideoTrainer(overrides=args)
    
    # Run validation only
    # We need to setup the model first. Since we are debugging resolution, 
    # we can use a random model or a pre-trained one if available.
    # The config loads yolo12-video-p2-p5.yaml.
    
    # We call trainer.train() for 1 epoch or use internal validator
    # Using validator directly might be tricky due to setup.
    # Easiest is to run train with epochs=1 and get the validation at the end, 
    # OR if trainer has a validator method exposed.
    
    # DetectionTrainer has .validate() method but it usually requires a trained model.
    # Let's try to just run validation on the initialized model (random weights).
    # We need to ensure the trainer is set up correctly.
    
    # trainer.train() calls .validate() at the end of epoch.
    # Let's try explicitly calling validator.
    
    # Setup trainer (usually done in .train())
    # trainer._setup_train(world_size=1) # Private method, risky.
    
    # Let's just run training for 1 epoch with small batch.
    trainer.args.epochs = 1
    trainer.train()

if __name__ == '__main__':
    validate()
