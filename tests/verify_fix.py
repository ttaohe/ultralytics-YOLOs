
import torch
from ultralytics.models.sam.modules.memory_attention import MemoryAttention, MemoryAttentionLayer

def verify_memory_attention_fix():
    d_model = 64
    layer = MemoryAttentionLayer(d_model=d_model)
    # batch_first=True is default
    attention = MemoryAttention(d_model=d_model, pos_enc_at_input=True, layer=layer, num_layers=1, batch_first=True)
    
    B, L, D = 2, 10, d_model
    curr = torch.randn(B, L, D)
    memory = torch.randn(B, L, D)
    curr_pos = torch.randn(B, L, D)
    memory_pos = torch.randn(B, L, D)
    
    print(f"Input shape: {curr.shape}")
    
    # We expect the output to be (B, L, D)
    # And inside, RoPEAttention should receive (B, L, D)
    
    # Let's monkey patch RoPEAttention.forward to print shapes
    original_forward = layer.self_attn.forward
    
    def mocked_forward(q, k, v, num_k_exclude_rope=0):
        print(f"RoPEAttention input q shape: {q.shape}")
        return original_forward(q, k, v, num_k_exclude_rope)
        
    # Apply mock to the layer instance inside attention
    attention.layers[0].self_attn.forward = mocked_forward
    
    try:
        output = attention(curr, memory, curr_pos, memory_pos)
        print(f"Output shape: {output.shape}")
        assert output.shape == (B, L, D), f"Expected {(B, L, D)}, got {output.shape}"
        print("SUCCESS: Shapes match and execution completed.")
    except Exception as e:
        print(f"Error: {e}")
        raise e

if __name__ == "__main__":
    verify_memory_attention_fix()
