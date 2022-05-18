"""PyPI setup script.  To use it, run `python setup.py sdist bdist_wheel` from this directory."""

import setuptools
from setuptools_rust import RustExtension, Binding

import os
import sys
develop_mode = os.environ.get("DEVELOP_MODE", False)
if develop_mode:
    rust_extensions = []
    print("make sure to install the rust extensions manually\n cd rust; maturin develop;")
else:
    rust_extensions = [
        RustExtension(
            "fastsimrust",
            "rust/Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
        ),
    ]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastsim",
    version="2.0.5",
    author="MBAP Group",
    author_email="fastsim@nrel.gov",
    description="Tool for modeling vehicle powertrains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nrel.gov/transportation/fastsim.html",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7,<3.9',
    include_package_data=True,
    package_data={
        "fastsim.resources": ["*"],
        "fastsim.docs": ["*"],
    },
    install_requires=[
        "pandas>=1",
        "matplotlib>=3.3",
        "numpy>=1.18",
        "seaborn>=0.10",
    ],
    # rust extension
    zip_safe=False,
    rust_extensions=rust_extensions,
)
