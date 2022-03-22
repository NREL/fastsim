"""PyPI setup script.  To use it, run `python setup.py sdist bdist_wheel` from this directory."""

import setuptools
# from setuptools_rust import Binding, RustExtension

# TODO: uncomment lines for rust stuff to work after figuring out how to properly configure it
# TODO: put the folliwing in MANIFEST.in: 
# include Cargo.toml
# recursive include src *
# put this in pyproject.toml:
# [build-system]
# requires = ["maturin>=0.12,<0.13", "setuptools", "wheel", "setuptools-rust"]
# build-backend = "maturin"


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastsim",  
    version="0.0.1",
    # rust_extensions=[RustExtension("fastsimrust.fastsimrust", binding=Binding.PyO3)],
    # rust extensions are not zip safe, just like C-extensions.
    # zip_safe=False,    
    author="MBAP Group",
    author_email="fastsim@nrel.gov",
    description="Tool for modeling vehicle powertrains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nrel.gov/transportation/fastsim.html",
    packages=setuptools.find_packages().append("fastsimrust"),
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
        # "setuptools_rust>1",
    ],
)
