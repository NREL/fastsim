![FASTSim Logo](fastsim-icon-web-131x172.jpg)

# Description
This is the python/rust flavor of FASTSim which is based on the original Excel implementation. Effort will be made to keep the core methodology between this software and the Excel flavor in line with one another. Other FASTSim flavors may spin off as variations on this core functionality, but these should integrated back into `main` if there is any intent of persistence.

All classes and methods are self-documented.  

# Installation

## PyPI Installation

In an active Python environment (either [venv](https://docs.python.org/3/library/venv.html) or [Conda](https://www.anaconda.com/)), run `pip install fastsim`.

## Installing from NREL's internal GitHub

1. Clone the repository from GitHub:

       git clone git@github.nrel.gov:MBAP/fastsim.git

2. Set up an environment for the project. There are two ways to do this:

   - Using Anaconda (with Python 3.8 or newer):

         conda create -n fastsim python=3.8
         conda activate fastsim
         conda install -c conda-forge "rust>=1.64"

   - Manually:
     - Install [Rust](https://www.rust-lang.org/tools/install). 
     - Then, using a virtual environment, and making sure to use Python >= 3.8:
    
        PowerShell:

            python -m venv .venv
            .venv/Scripts/Activate.ps1

        Bash:

            python -m venv .venv
            source .venv/bin/activate

        Command Prompt:

            python -m venv .venv
            .venv/Scripts/activate.bat
    
3. From within the top level of the FASTSim repository, install using pip:

       pip install .

## Developers
Developers might want to install the code in place so that FASTSim files can be editable (the `-e` flag for pip provides this behavior). This option can be handy since FASTSim will be installed in place from the installation location and any updates will be propagated each time FASTSim is freshly imported.  

To do this, a couple of extra steps are required:

1. First install the python code in place `DEVELOP_MODE=True pip install -e ".[dev]"` if on Mac OS, Linux, or Windows Bash (e.g. git bash, VSCode bash).  On Windows in Power Shell or Command Prompt, run `set DEVELOP_MODE=True` then `pip install -e ".[dev]"`.
1. Within the same python environment, navigate to `fastsim/rust/` and run `pip install maturin`.
1. _Optional_: Within the `rust/` folder (which contains the rust `src/` folder), run `cargo test --release` to build and run the tests.
1. In `fastsim/rust/fastsim-py`, you should now be able to run `maturin develop --release`, which will enable the tests that use rust to run.  You should also now be able to run `fastsim/fastsim/docs/demo.py`.

After FASTSim has been installed as editable per the above instructions, you can rebuild and test everything with `sh build_and_test.sh` in Windows bash or `./build_and_test.sh` in Linux/Unix in the `fastsim/` dir.  

# Users with NREL VPN Access
## Installation as PyPi Package
Note: the following instructions work only if you are inside NREL VPN:  
To install and/or update, run
```
pip install fastsim --upgrade --extra-index-url=https://github.nrel.gov/pages/MBAP/mbap-pypi/
```

## Adding FASTSim as a Depency in Rust
Add this line:  
`fastsim-core = { git = "https://github.nrel.gov/MBAP/fastsim", branch = "rust-port" }`  
to your Cargo.toml file, modifying the `branch` key as appropriate.  

# Usage
To see and run examples, navigate to fastsim/docs and run the various *demo.py files to see fastsim use cases. There are other examples in fastsim/tests.  

To get help in an interactive ipython or jupyter session:  
```
import fastsim
fastsim.simdrive.SimDriveClassic? # or
help(fastsim.simdrive.SimDriveClassic)
```

Help can be used in this manner on any FASTSim object.

# Testing

The `unittest` package has been implemented such that you can run `python -m unittest discover` from within the fastsim folder, and all tests will be automatically discovered and run.  

## Against Previous Python Version

To run tests, first run the command `from fastim import tests`.  
To compare FASTSim back to the master branch version from 17 December 2019, run `tests.test26veh3cyc.main()`.  For timing comparisons, run `tests.test26veh3cyc_CPUtime.main()`.  

## Against Excel FASTSim
To compare Python FASTSim results to Excel FASTSim, you can run `tests.test_vs_excel.main()` to do an experimental (i.e. beta) comparison against saved Excel results. If you have the Excel version (obtainable here: [https://www.nrel.gov/transportation/fastsim.html](https://www.nrel.gov/transportation/fastsim.html)) of FASTSim open, you can specify `rerun_excel=True` to do a live run of the Excel version.

# List of Abbreviations
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

# Release Notes
2.0.13 -- yet another PyPI-related change
2.0.12 -- more PyPI-related changes
2.0.11 -- fixes related to PyPI web rendering
2.0.10 -- logging fixes, proc macro reorganization, some CAVs performance fixes
2.0.9 -- support for mac ARM/RISC architecture
2.0.8 -- performance improvements
2.0.6 -- `dist_v2_m` fixes and preliminary CAV functionality
2.0.5 -- added `to_rust` method for cycle  
2.0.4 -- exposed `veh.set_veh_mass`  
2.0.3 -- exposed `veh.__post_init__`  
2.0.2 -- provisioned for non-default vehdb path  
2.0.1 -- bug fix  
2.0.0 -- All second-by-second calculations are now implemented in both rust and python.  Rust provides a ~30x speedup  
1.3.1 -- `fastsim.simdrive.copy_sim_drive` function can deepcopy jit to non-jit (and back) for pickling  
1.2.6 -- time dilation bug fix for zero speed  
1.2.4 -- bug fix changing `==` to `=`  
1.2.3 -- `veh_file` can be passed as standalone argument.  `fcEffType` can be anything if `fcEffMap` is provided, but typing is otherwise enforced.  
1.2.2 -- added checks for some conflicting vehicle parameters.  Vehicle parameters `fcEffType` and `vehPtType` must now be str type.  
1.2.1 -- improved time dilation and added test for it  
1.1.7 -- get_numba_veh() and get_numba_cyc() can now be called from already jitted objects  
1.1.6 -- another bug fix for numba compatibility with corresponding unit test  
1.1.5 -- bug fix for numba compatibility of fcPeakEffOverride and mcPeakEffOverride  
1.1.4 -- nan bug fix for fcPeakEffOverride and mcPeakEffOverride  
1.1.3 -- provisioned for optional load time motor and engine peak overrides  
1.1.2 -- made vehicle loading _more_ more robust  
1.1.1 -- made vehicle loading more robust  
1.1.0 -- separated jitclasses into own module, made vehicle engine and motor efficiency setting more robust  
1.0.4 -- bug fix with custom engine curve  
1.0.3 -- bug fixes, faster testing  
1.0.2 -- forced type np.float64 on vehicle mass attributes  
1.0.1 -- Added `vehYear` attribute to vehicle and other minor changes.    
1.0.0 -- Implemented unittest package.  Fixed energy audit calculations to be based on achieved speed.  Updated this file.  Improved documentation.  Vehicle can be instantiated as dict.   
0.1.5 -- Updated to be compatible with ADOPT    
0.1.4 -- Bug fix: `mcEffMap` is now robust to having zero as first element    
0.1.3 -- Bug fix: `fastsim.vehicle.Vehicle` method `set_init_calcs` no longer overrides `fcEffMap`.    
0.1.2 -- Fixes os-dependency of xlwings by not running stuff that needs xlwings.  Improvements in functional test.  Refinment utomated typying of jitclass objects.    
0.1.1 -- Now includes label fuel economy and/or battery kW-hr/mi values that match excel and test for benchmarking against Excel values and CPU time.   

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
