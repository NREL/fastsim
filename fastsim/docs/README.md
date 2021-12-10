![FASTSim Logo](fastsim-icon-web-131x172.jpg)

# Description
This is the pythonic flavor of FASTSim which is based on the original Excel implementation. Effort will be made to keep the core methodology between this software and the Excel flavor in line with one another. Other FASTSim flavors may spin off as variations on this core functionality, but these should integrated back into master if there is any intent of persistence.

All classes and methods are self-documented.  

# Installation
First, clone the repository from GitHub if you don't already have a local copy of the FASTSim package files:

    git clone git@github.nrel.gov:MBAP/fastsim.git  
    
FASTSim depends on python 3.8. One way to satisfy this is to use conda (we recommend Anaconda Powershell Prompt for Windows OS):

    conda create -n fastsim python=3.8
    conda activate fastsim
    
Then, from within the top level of the FASTSim folder, run a pip install:

    pip install -e .
    
This will install FASTSim with minimal dependencies in place so that FASTSim files can be editable (`-e` provides this behavior). Developers will find this option handy since FASTSim will be installed in place from the installation location and any updates will be propagated each time FASTSim is freshly imported.  

For users who are not developers, FASTSim can also be installed without the `-e` option (i.e. `pip install .`), and package files will be copied to the python site-packages folder.   

# Update
Note: the following instructions work only if you are inside NREL VPN:  
To update, run
```
pip install fastsim --upgrade --extra-index-url=https://github.nrel.gov/pages/MBAP/mbap-pypi/
```

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

# numba
To significantly speed up the simulations `numba` has been used extensively to augment every class in `fastsim.simdrive`, `fastsim.cycle`, and `fastsim/vehicle`. Classes that are "just in time compiled", as well as variables needed for datatype declaration, are preceeded by the `numba` decorator `@jitclass` or defined by numba types `float64, int32, bool_, types`, respectively.

*notes on numba*
- `numba` caches compiled classes for you in `__pycache__`
- should usually automatically register source code changes and recompile, even if `__pycache__` isn't deleted first

## numba pitfalls
- `numba` does not always work well with `numpy`, although this happens in rare occassions and has completely been resolved in this code base, as far as we know.
- Some users have reported Python __zombie__ processes that crop up when using the `numba` extended code. This has been a difficult to reproduce bug and may have been user platform specific, it also involved heavy use of `xlwings` calling the code via Excel. These zombies can be seen in the Task Manager as latent Pythonw processes, they will prevent `numba` from recompiling, even if you delete the `__pycache__` folders.

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
Aaron Brooker -- Aaron.Brooker@nrel.gov  
Jeffrey Gonder -- Jeff.Gonder@nrel.gov  
Chad Baker -- Chad.Baker@nrel.gov  
Eric Wood -- Eric.Wood@nrel.gov  
Jacob Holden -- Jacob.Holden@nrel.gov  
Grant Payne -- Grant.Payne@nrel.gov  
Matthew Moniot -- Matthew.Moniot@nrel.gov  
Jason Lustbader -- Jason.Lustbader@nrel.gov  
Sean Lopp -- sean@rstudio.com  
Laurie Ramroth -- lramroth@ford.com  
