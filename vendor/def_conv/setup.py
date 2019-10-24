from setuptools import setup
from setuptools import find_namespace_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="def_conv",
    packages=find_namespace_packages(),
    ext_modules=[
        CUDAExtension(
            "deform_conv_cuda",
            ["src/deform_conv_cuda.cpp", "src/deform_conv_cuda_kernel.cu"],
        ),
        CUDAExtension(
            "deform_pool_cuda",
            ["src/deform_pool_cuda.cpp", "src/deform_pool_cuda_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
