from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='pydeform',
    ext_modules=[
        CppExtension(
        	name='pydeform',
        	sources=['pydeform.cc'],
        	extra_compile_args=['-I../lib','-L../../build'],
        	extra_ldflags=['-ldeform']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
