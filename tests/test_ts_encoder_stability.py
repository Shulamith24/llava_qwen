
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.ts_encoder import SimplePatchTSTEncoder

def test_encoder_stability():
    print("Testing SimplePatchTSTEncoder stability...")
    
    # Configuration
    context_window = 256
    patch_len = 16
    stride = 8
    d_model = 128
    
    # Initialize encoder
    encoder = SimplePatchTSTEncoder(
        context_window=context_window,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.1
    )
    
    # Move to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    print(f"Device: {device}")
    
    # Create dummy input
    batch_size = 2
    n_vars = 3
    x = torch.randn(n_vars, context_window, device=device)
    
    # Test 1: FP32 Input
    print("\nTest 1: FP32 Input")
    try:
        out = encoder(x)
        print(f"Output shape: {out.shape}")
        print(f"Output dtype: {out.dtype}")
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("FAILED: Output contains NaN or Inf")
        else:
            print("PASSED: Output is finite")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 2: BF16 Input (Simulating Trainer)
    print("\nTest 2: BF16 Input")
    try:
        x_bf16 = x.to(dtype=torch.bfloat16)
        # Simulate mixed precision environment
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = encoder(x_bf16)
        
        print(f"Output shape: {out.shape}")
        print(f"Output dtype: {out.dtype}")
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("FAILED: Output contains NaN or Inf")
        else:
            print("PASSED: Output is finite")
            
    except Exception as e:
        print(f"FAILED: {e}")

    # Test 3: Check Gradients
    print("\nTest 3: Gradient Check")
    try:
        x.requires_grad = True
        out = encoder(x)
        loss = out.mean()
        loss.backward()
        
        # Check if pos_embed is a buffer (no grad)
        if encoder.pos_embed.requires_grad:
             print("FAILED: pos_embed should not require grad (should be buffer)")
        else:
             print("PASSED: pos_embed is fixed buffer")
             
        # Check if gradients flow to projection
        if encoder.patch_projection.weight.grad is None:
             print("FAILED: patch_projection.weight.grad is None")
        else:
             print("PASSED: Gradients flow to projection")
             
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_encoder_stability()
