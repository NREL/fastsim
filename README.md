# Description
This repo houses the pythonic flavor of FASTSim which is based on the original Excel implementation. Effort will be made to keep the core methodology between this software and the Excel flavor in line with one another. Other FASTSim flavors may spin off as variations on this core functionality, but these should integrated back into master if there is any intent of persistence.

All classes and methods are self-documented.  

# Installation
First, clone the repository from GitHub:

    git clone git@github.nrel.gov:MBAP/fastsim.git
    
FASTSim depends on python 3.7. One way to satisfy this is to use conda:

    conda create -n fastsim python=3.7
    conda activate fastsim
    
Then, from within the top level of the folder created by the `git clone` command, just run a pip install:

    pip install .
    
This will install FASTSim with minimal dependencies.

FASTSim can also be installed with the `-e` option (i.e. `pip install -e fastsim`) so that FASTSim files can be editable. Developers will find this option handy since FASTSim will be installed in place from the installation location, and any updates will be propagated each time FASTSim is freshly imported.  

# Usage
To run an example, navigate to fastsim/docs and run `jupyter lab demo.ipynb` to see a demo of fastsim use cases. There are other examples in fastsim/docs and fastsim/tests.  

To get help in an interactive ipython or jupyter session:  
```
import fastsim
fastsim.simdrive.SimDriveClassic? # or
help(fastsim.simdrive.SimDriveClassic)
```

# Testing

## Against Previous Python Version
Run the file 'tests/test26veh3cyc.py' to compare FASTSim back to the master branch version from 17 December 2019.  For timing comparisons, run 'tests/test26veh3cyc_CPUtime.py' from within 'tests/'.  

## Against Excel FASTSim
If you have the Excel version (obtainable here: [https://www.nrel.gov/transportation/fastsim.html](https://www.nrel.gov/transportation/fastsim.html)) of FASTSim open, you can run 'tests/test_vs_excel.py' from within 'tests/' to do an experimental (i.e. beta) comparison against Excel results.  

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
0.1.0 -- pip install without using the `-e` option now copies all fastsim files and folder hierarchy into the python site-packages folder.  Tests are now included as modules inside of a sub-package.   

# Known Bugs

tests/accel_test.py will print "Warning: There is a problem with conservation of energy." for some vehicles.  This will be resolved in a future release.   

