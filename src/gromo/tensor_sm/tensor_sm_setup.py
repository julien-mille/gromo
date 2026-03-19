from os.path import join

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


project_root = '.'
sources = [join(project_root, file) for file in ['tensor_sm.cpp',
                                                 'tensor_sm_cuda_kernel.cu']]

setup(
    name='tensor_sm',
    description="tensor_sm module for pytorch",
    long_description_content_type="text/markdown",
    ext_modules=[
        CUDAExtension('tensor_sm',
                      sources,
                      extra_compile_args={'cxx': ['-fopenmp'], 'nvcc':[]})
    ],
    package_dir={'': project_root},
    # packages=['PatchMatch_Module'],
    cmdclass={
        'build_ext': BuildExtension
    },
    )
