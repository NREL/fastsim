"""PyPI setup script.  To use it, run `python setup.py sdist bdist_wheel` from this directory."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastsim",  
    version="1.2.1",
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
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        "fastsim.resources": ["*"],
        "fastsim.docs": ["*"],
    },
    install_requires=[
        "pandas>=1",
        "matplotlib>=3.3",
        "numpy>=1.18",
        "numba>=0.52",
        "seaborn>=0.10",
    ],
)
