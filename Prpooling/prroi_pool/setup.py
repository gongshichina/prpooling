from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='prpooling',
    ext_modules=[
        CUDAExtension('prpooling', [
            'src/prroi_pooling_gpu.cpp',
            'src/prroi_pooling_gpu_impl.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})