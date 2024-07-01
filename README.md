# FASTSim

![FASTSim Logo](https://www.nrel.gov/transportation/assets/images/icon-fastsim.jpg)

[![homepage](https://img.shields.io/badge/homepage-fastsim-blue)](https://www.nrel.gov/transportation/fastsim.html) [![tests](https://github.com/NREL/fastsim/actions/workflows/tests.yaml/badge.svg)](https://github.com/NREL/fastsim/actions/workflows/tests.yaml) [![wheels](https://github.com/NREL/fastsim/actions/workflows/wheels.yaml/badge.svg)](https://github.com/NREL/fastsim/actions/workflows/wheels.yaml?event=release) [![python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/fastsim/) [![documentation](https://img.shields.io/badge/documentation-book-blue.svg)](https://nrel.github.io/fastsim/) [![github](https://img.shields.io/badge/github-fastsim-blue.svg)](https://github.com/NREL/fastsim)

## Description

This is the python/rust flavor of [NREL's FASTSim<sup>TM</sup>](https://www.nrel.gov/transportation/fastsim.html), which is based on the original Excel implementation. Effort will be made to keep the core methodology between this software and the Excel flavor in line with one another.

All classes and methods are self-documented.  

## Installation

### Python

Set up and activate a python environment (compatible with Python 3.8 - 3.10; we recommend Python 3.10) with the following steps.

#### [Anaconda](https://www.anaconda.com/)

1. Create: `conda create -n fastsim python=3.10`
1. Activate: `conda activate fastsim`

#### [venv](https://docs.python.org/3/library/venv.html)

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

### FASTSim

#### Via PyPI

In an active Python environment created above, run `pip install fastsim`.

#### Building from Scratch

Developers might want to install the code in place so that FASTSim files can be editable (the `-e` flag for pip provides this behavior). This option can be handy since FASTSim will be installed in place from the installation location and any updates will be propagated each time FASTSim is freshly imported.  To do this, you'll need to have the [Rust toolchain](https://www.rust-lang.org/tools/install) installed.

- Option 1: run `sh build_and_test.sh` in root folder.  
- Option 2:  
    1. Run `pip install -e ".[dev]"`  
    Optional testing steps:
    1. Run `cd rust/ && cargo test`
    1. Run `pytest -v python/fastsim/tests/`

## Usage

To see and run examples, download the FASTSim demo files using the following code (with your Python environment activated and FASTSim installed):
```python
from fastsim import utils  
utils.copy_demo_files()
```
This code downloads demo files into a specified local directory (if no directory is specified, it will create a `\demos` folder in the current working directory). WARNING: If you download the demo files to a location where files of the same name already exist, the original files will be overwritten.  

## Adding FASTSim as a Dependency in Rust

### Via GitHub

Add this line:

```
fastsim-core = { git = "https://github.com/NREL/fastsim/", branch = "fastsim-2" }
```

to your Cargo.toml file, modifying the `branch` key as appropriate.  

### Via Cargo

FASTSim is [available as a Rust crate](https://crates.io/crates/fastsim-core), which can be added to your dependencies via the following command:

```
cargo add fastsim-core
```

## List of Abbreviations

cur = current time step  
prev = previous time step  
cyc = drive cycle  
secs = seconds  
mps = meters per second  
mph = miles per hour  
kw = kilowatts, unit of power  
kwh = kilowatt-hour, unit of energy  
kg = kilograms, unit of mass  
max = maximum  
min = minimum  
avg = average  
fs = fuel storage (eg. gasoline/diesel tank, pressurized hydrogen tank)  
fc = fuel converter (eg. internal combustion engine, fuel cell)  
mc = electric motor/generator and controller  
ess = energy storage system (eg. high voltage traction battery)  
chg = charging of a component  
dis = discharging of a component  
lim = limit of a component  
regen = associated with regenerative braking  
des = desired value  
ach = achieved value  
in = component input  
out = component output  

## Contributors  

Chad Baker -- <Chad.Baker@nrel.gov>  
Aaron Brooker -- <Aaron.Brooker@nrel.gov>  
Kyle Carow -- <Kyle.Carow@nrel.gov>  
Robin Steuteville -- <Robin.Steuteville@nrel.gov>  
Jeffrey Gonder -- <Jeff.Gonder@nrel.gov>  
Jacob Holden -- <Jacob.Holden@nrel.gov>  
Jinghu Hu -- <Jinghu.Hu@nrel.gov>  
Jason Lustbader -- <Jason.Lustbader@nrel.gov>  
Sean Lopp -- <sean@rstudio.com>  
Matthew Moniot -- <Matthew.Moniot@nrel.gov>  
Grant Payne -- <Grant.Payne@nrel.gov>  
Laurie Ramroth -- <lramroth@ford.com>  
Eric Wood -- <Eric.Wood@nrel.gov>  
