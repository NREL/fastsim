import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastsim",  # Replace with your own username
    version="0.0.1",
    author="MBAP Group",
    author_email="fastsim@nrel.gov",
    description="Tool for modeling vehicle powertrains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nrel.gov/transportation/fastsim.html",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        "fastsim.resources": ["*"]
    },
    install_requires=[
        "pandas>=1",
        "matplotlib",
        "numpy>=1.18",
        "numba",
        "scipy>=1.5",
        "seaborn>=0.10",
        "bleach>=2.1",
        "docutils>=0.13",
        "Pygments>=2.5",
        "jupyter>=1.0",
        "jupyter_client>=6.1",
        "jupyter_console>=6.1",
        "jupyter_core>=4.6",
        "jupyterlab>=2.1",
        "jupyterlab_server>=1.2",
        "keyring>=15.1",
        "pkginfo>=1.4",
        "tqdm>=4.14",
    ],
)
