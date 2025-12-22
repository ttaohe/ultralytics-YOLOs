
import sys
from unittest.mock import MagicMock

# Mock ultralytics.nn.tasks to break circular import
# tasks imports nn.modules, which imports video_block
sys.modules["ultralytics.nn.tasks"] = MagicMock()

import torch
# Now we can import video_block. 
# It will import models.sam..., which imports models..., which imports engine..., which imports tasks (now mocked).
from ultralytics.nn.modules.video_block import YOLOMemoryAttention

def verify_yolo_video_block():
    print("Verifying YOLO Video Block...")
    
    # Setup
    B, C, H, W = 2, 64, 16, 32 # Non-square shape
    d_model = 64
    model = YOLOMemoryAttention(d_model=d_model, max_memory=4)
    model.eval() # Inference mode
    
    x = torch.randn(B, C, H, W)
    
    print(f"Input shape: {x.shape}")
    
    # Run forward pass (Frame 1)
    out1 = model(x)
    print(f"Frame 1 Output shape: {out1.shape}")
    assert out1.shape == x.shape
    assert len(model.memory_bank) == 1
    print("Frame 1 passed.")
    
    # Run forward pass (Frame 2)
    x2 = torch.randn(B, C, H, W)
    out2 = model(x2)
    print(f"Frame 2 Output shape: {out2.shape}")
    assert out2.shape == x.shape
    assert len(model.memory_bank) == 2
    print("Frame 2 passed.")
    
    # Test Training Mode
    model.train()
    model.time_steps = 2
    # Input for training: (B*T, C, H, W)
    x_train = torch.randn(B*2, C, H, W)
    out_train = model(x_train)
    print(f"Training Output shape: {out_train.shape}")
    assert out_train.shape == x_train.shape
    print("Training mode passed.")
    
    print("All verifications passed!")

if __name__ == "__main__":
    verify_yolo_video_block()
