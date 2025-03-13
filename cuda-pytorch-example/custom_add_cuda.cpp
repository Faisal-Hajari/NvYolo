#include <torch/extension.h>

// Declaration of the CUDA function
torch::Tensor custom_add_cuda(torch::Tensor input);

// CPU implementation (fallback)
torch::Tensor custom_add_cpu(torch::Tensor input) {
    return torch::sum(input);
}

// Interface exposed to Python
torch::Tensor custom_add(torch::Tensor input) {
    if (input.device().is_cuda()) {
        return custom_add_cuda(input);
    } else {
        return custom_add_cpu(input);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add", &custom_add, "Custom addition operation");
}