from setuptools import find_packages, setup
from setuptools_rust import RustExtension

setup(
    name='strsim',
    version='0.0.1',
    packages=find_packages(
        where='strsim',
    ),
    include_package_data=True,
    rust_extensions=[
        RustExtension(
            "strsim._py_strsim",
        )
    ],
)