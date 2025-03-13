#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel for tensor sum
template <typename scalar_t>
__global__ void custom_add_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Use shared memory for efficient reduction
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_mem);
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    scalar_t val = 0;
    if (idx < size) {
        val = input[idx];
    }
    shared[tid] = val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// Host function to launch the kernel
torch::Tensor custom_add_cuda(torch::Tensor input) {
    auto input_contiguous = input.contiguous();
    const int size = input_contiguous.numel();
    
    // Create output tensor
    auto output = torch::zeros({1}, input.options());
    
    // Set up CUDA execution parameters
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Launch kernel with appropriate type
    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_add_cuda", ([&] {
        const int shared_memory = threads * sizeof(scalar_t);
        custom_add_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
            input_contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));
    
    return output;
}