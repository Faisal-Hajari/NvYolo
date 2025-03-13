import torch
from custom_add_cuda import custom_add

# Create tensor
x = torch.randn(5, device="cuda")

# Create a larger tensor for more meaningful timing
x_large = torch.randn(1000000, device="cuda")  # 1M elements

# Timing function using CUDA events (more accurate for GPU operations)
def benchmark(func, x, iterations=100):
    # Warmup
    for _ in range(10):
        _ = func(x)
        
    # Create CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Timing loop
    start.record()
    for _ in range(iterations):
        result = func(x)
    end.record()
    
    # Synchronize and get elapsed time
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / iterations
    
    return result, elapsed_ms

# Small tensor benchmark
print("Small tensor (5 elements):")
custom_result, custom_time = benchmark(custom_add, x)
torch_result, torch_time = benchmark(torch.sum, x)

print(f"Custom sum : {custom_result.item()}")
print(f"PyTorch sum: {torch_result.item()}")
print(f"Custom implementation time : {custom_time:.4f} ms")
print(f"PyTorch implementation time: {torch_time:.4f} ms")
print(f"Speed ratio: {torch_time/custom_time:.2f}x")

# Large tensor benchmark (more realistic)
print("\nLarge tensor (1M elements):")
custom_result_large, custom_time_large = benchmark(custom_add, x_large)
torch_result_large, torch_time_large = benchmark(torch.sum, x_large)

print(f"Custom sum : {custom_result_large.item()}")
print(f"PyTorch sum: {torch_result_large.item()}")
print(f"Custom implementation time : {custom_time_large:.4f} ms")
print(f"PyTorch implementation time: {torch_time_large:.4f} ms")
print(f"Speed ratio: {torch_time_large/custom_time_large:.2f}x")