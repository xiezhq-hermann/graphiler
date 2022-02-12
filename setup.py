from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="graphiler",
    version="0.0.1",
    url="https://github.com/xiezhq-hermann/graphiler",
    package_dir={'': 'python'},
    packages=['graphiler', 'graphiler.utils'],
    ext_modules=[
        CppExtension('graphiler.mpdfg', [
            'src/pybind.cpp',
            'src/builder.cpp'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires=">=3.7"
)
