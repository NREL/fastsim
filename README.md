![FASTSim Logo](https://www.nrel.gov/transportation/assets/images/icon-fastsim.jpg)

[![homepage](https://img.shields.io/badge/homepage-fastsim-blue)](https://www.nrel.gov/transportation/fastsim.html)
[![tests](https://github.com/NREL/fastsim/actions/workflows/tests.yaml/badge.svg)](https://github.com/NREL/fastsim/actions/workflows/tests.yaml)
<!-- TODO: replace wheels -->
[![wheels](https://github.com/NREL/fastsim/actions/workflows/wheels.yaml/badge.svg)](https://github.com/NREL/fastsim/actions/workflows/wheels.yaml?event=release)
[![python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/fastsim/)
[![documentation](https://img.shields.io/badge/documentation-book-blue.svg)](https://nrel.github.io/fastsim/)
[![github](https://img.shields.io/badge/github-fastsim-blue.svg)](https://github.com/NREL/fastsim)

# Description
This is the `fastim-3` version of [NREL's FASTSim](https://www.nrel.gov/transportation/fastsim.html).
It introduces numerous new enhancements and features, including:
- ~10x faster! -- when setting `save_interval` to `None`, which means only the state at the last 
  time step, which includes fuel consumption and/or battery depletion, among other useful 
  cumulative state variables.  
- Roughly ~60% reduction in memory consumption (~160 mb in [`fastsim-2`](https://github.com/NREL/fastsim) 
  v. 60 mb in [`fastsim-3`](https://github.com/NREL/fastsim/tree/fastsim-3)
- object-oriented, hierarchical model structure
- ability to control granularity of time-resolved data -- e.g. save at every time step, save at 
  every _n_th time step, or never save at all (saving only cumulative trip-level results)
- component-specific vehicle models -- i.e. the conventional vehicle contains only data for the fuel
  converter and other relevant components but does contain any dummy battery or motor parameters as
  is the case in `fastsim-2`
- file formats that are more robust and more human readable
- backwards compatibility with `fastsim-2`
- flexible data structures to allow for future model types
- ability to simulate standalone component models
- flexible model interfacing (e.g. multiple folder/file formats for reading and/or writing data)
- more accurate interpolation methods

# Installation

## Python 
Set up and activate a python environment (compatible with Python 3.8 - 3.10; we recommend Python 3.10) with the following steps.

### [Anaconda](https://www.anaconda.com/) 
1. Create: `conda create -n fastsim python=3.10`
1. Activate: `conda activate fastsim`

### [venv](https://docs.python.org/3/library/venv.html)
There is some variation based on your Operating System:  

- PowerShell (windows):
    1. Create: `python -m venv fastsim-venv` -- name is user decision
    1. Activate: `fastsim-venv/Scripts/Activate.ps1`

- Bash (i.e. unix/linux/mac):
    1. Create: `python -m venv fastsim-venv` -- name is user decision
    1. Activate: `source fastsim-venv/bin/activate`

- Command Prompt (windows):
    1. Create: `python -m venv fastsim-venv` -- name is user decision
    1. Activate: `fastsim-venv/Scripts/activate.bat`

### [Pixi](https://pixi.sh/latest/) (recommended for developers)
1. `pixi shell -e dev` -- this creates the environment, installs all the dependencies,
    builds fastsim, and activates the environment

## FASTSim-3
### Via PyPI
In an active Python environment created above, run `pip install fastsim`.

### Building from Scratch
Developers might want to install the code in place so that FASTSim files can be editable (the `-e` flag for 
pip provides this behavior). This option can be handy since FASTSim will be installed in place from the 
installation location and any updates will be propagated each time FASTSim is freshly imported.  To do 
this, you'll need to have the [Rust toolchain](https://www.rust-lang.org/tools/install) installed.

- Option 1: run `sh build_and_test.sh` in root folder.  
- Option 2:  
    1. Run `pip install -e ".[dev]"`  
    Optional testing steps:
    1. Run `cargo test`
    1. Run `pytest -v python/fastsim/tests/`

# Usage
To see and run examples, download the `fastsim-3` demo files using the following code (with your Python environment activated and `fastsim-3` installed):
```python
from fastsim import utils  
utils.copy_demo_files()
```
This code downloads demo files into a specified local directory (if no directory is specified, it will create a `\demos` folder in the current working directory). WARNING: If you download the demo files to a location where files of the same name already exist, the original files will be overwritten. 


# Contributors  
Chad Baker -- Chad.Baker@nrel.gov  
Aaron Brooker -- Aaron.Brooker@nrel.gov  
Kyle Carow -- Kyle.Carow@nrel.gov  
Jeffrey Gonder -- Jeff.Gonder@nrel.gov  
Jacob Holden -- Jacob.Holden@nrel.gov  
Jinghu Hu -- Jinghu.Hu@nrel.gov  
Jason Lustbader -- Jason.Lustbader@nrel.gov  
Sean Lopp -- sean@rstudio.com  
Matthew Moniot -- Matthew.Moniot@nrel.gov  
Grant Payne -- Grant.Payne@nrel.gov  
Laurie Ramroth -- lramroth@ford.com  
Eric Wood -- Eric.Wood@nrel.gov
Robin Steuteville -- Robin.Steuteville@nrel.gov
