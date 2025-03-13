# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_add",
    ext_modules=[
        CUDAExtension(
            name="custom_add_cuda",
            sources=[
                "custom_add_cuda.cpp",
                "custom_add_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

# custom_add.py
import torch
from torch.utils.cpp_extension import load
import os

class CustomAdd(torch.nn.Module):
    def __init__(self):
        super(CustomAdd, self).__init__()
        
    def forward(self, input):
        # Use the JIT-loaded module or the installed extension
        if not hasattr(self, '_custom_add_cuda'):
            self._custom_add_cuda = torch.ops.load_library("custom_add_cuda")
        return self._custom_add_cuda.custom_add(input)

# Alternative dynamic loading approach:
"""
# Load the C++ extension at runtime
current_path = os.path.dirname(os.path.realpath(__file__))
custom_add_cuda = load(
    name="custom_add_cuda",
    sources=[
        os.path.join(current_path, "custom_add_cuda.cpp"),
        os.path.join(current_path, "custom_add_cuda_kernel.cu"),
    ],
    verbose=True
)

class CustomAdd(torch.nn.Module):
    def __init__(self):
        super(CustomAdd, self).__init__()

    def forward(self, input):
        return custom_add_cuda.custom_add(input)
"""