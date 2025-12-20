import torch
import torch.nn as nn
import ultralytics.nn.tasks # Pre-import to break circular dependency
from ultralytics.nn.modules.video_attention import VSAMemoryAttention

def test_vsa_training():
    print("\n=== Testing VSA Training Mode ===")
    c1 = 64
    d_model = 64
    model = VSAMemoryAttention(c1=c1, d_model=d_model, max_memory=5, global_topk=2, block_size=4)
    model.train()
    
    # Simulate a batch of clips
    # B=4, T=3 clips, C=64, H=16, W=16
    # Batch Total = 12
    B_total = 12
    C = 64
    H = 16
    W = 16
    x = torch.randn(B_total, C, H, W).cuda()
    model.cuda()
    
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape
    print("Training forward pass successful.")

def test_vsa_inference():
    print("\n=== Testing VSA Inference Mode ===")
    c1 = 64
    d_model = 64
    model = VSAMemoryAttention(c1=c1, d_model=d_model, max_memory=10, global_topk=5, block_size=4)
    model.eval()
    model.cuda()
    
    # Simulate streaming frames
    B = 2
    C = 64
    H = 16
    W = 16
    
    print("Streaming 15 frames (max_memory=10)...")
    for i in range(15):
        x = torch.randn(B, C, H, W).cuda()
        out = model(x)
        
        # Verify internal banks
        if i == 0:
            assert len(model.key_bank) == 1, "After first frame, bank should have 1 element"
        else:
            assert len(model.key_bank) == min(i + 1, 10), f"Bank size mismatch at frame {i}: {len(model.key_bank)}"
            
        assert out.shape == x.shape
        
    print(f"Final Bank Size: {len(model.key_bank)}")
    assert len(model.key_bank) == 10
    print("Inference streaming successful.")
    
    # Reset
    model.reset_memory()
    assert len(model.key_bank) == 0
    print("Memory reset successful.")

if __name__ == "__main__":
    try:
        test_vsa_training()
        test_vsa_inference()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
