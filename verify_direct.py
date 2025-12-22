import os
import torch
from ultralytics.models.yolo.video.train import SAM2VideoTrainer
import warnings

warnings.filterwarnings("ignore")

def verify():
    args = dict(
        model='ultralytics/cfg/models/12/yolo12-video-p2-p5.yaml', 
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        imgsz=640,
        batch=1,   
        project='runs/verify-direct',
        name='direct_val_test',
        device='0',
        workers=4,
        val_imgsz=1920,
        val=True,
        plots=True,
    )
    
    print("Initializing Trainer...")
    trainer = SAM2VideoTrainer(overrides=args)
    
    print("Setting up train (creating model, data)...")
    # This might be needed to populate trainer.data and trainer.model
    
    # We need to setup dataset. 
    # trainer.train() does: self._setup_train(world_size)
    trainer._setup_train()
    
    print("Getting Validator...")
    validator = trainer.get_validator()
    
    print("Running Validation...")
    # validate() usually runs on self.validator
    # validator(trainer=trainer) should work
    validator(model=trainer.model)

if __name__ == "__main__":
    verify()
