# Description
This repo houses the pythonic flavor of FASTSim which is based on the original Excel implementation. Effort will be made to keep the core methodology between this software and the Excel flavor in line with one another. Other FASTSim flavors may spin off as variations on this core functionality, but these should integrated back into master if there is any intent of persistence.

All classes and methods are self-documented.  

# Usage
To run an example:
1. Assuming you have the latest version of [Anaconda Python Distribution](https://www.anaconda.com/products/individual), open the anaconda prompt, navigate to the top level fastsim directory, and run `conda env create -f environment.yml`.
2. Activate the environment with `$ conda activate fastsim_py`.
3. Navigate to .../fastsim/docs and run `jupter lab demo.ipynb` to see a demo of fastsim use cases. 

To use FASTSim as an external library, make sure the fastsim package is on your path and then import `fastsim.simdrive`.  

# Testing

## Against Previous Python Version
Run the file 'tests/test26veh3cyc.py' to compare FASTSim back to the master branch version from 17 December 2019.  For timing comparisons, run 'tests/test26veh3cyc_CPUtime.py'.  

## Against Excel FASTSim
This has not been implemented yet.

# numba
To significantly speed up the simulations `numba` has been used extensively to augment every class in `fastsim.simdrive`, `fastsim.cycle`, and `fastsim/vehicle`. Classes that are "just in time compiled", as well as variables needed for datatype declaration, are preceeded by the `numba` decorator `@jitclass` or defined by numba types `float64, int32, bool_, types`, respectively.

*notes on numba*
- `numba` caches compiled classes for you in `__pycache__`
- should usually automatically register source code changes and recompile, even if `__pycache__` isn't deleted first

## numba pitfalls
- `numba` does not always work well with `numpy`, although this happens in rare occassions and has completely been resolved in this code base, as far as we know.
- Some users have reported Python __zombie__ processes that crop up when using the `numba` extended code. This has been a difficult to reproduce bug and may have been user platform specific, it also involved heavy use of `xlwings` calling the code via Excel. These zombies can be seen in the Task Manager as latent Pythonw processes, they will prevent `numba` from recompiling, even if you delete the `__pycache__` folder

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

# Known Bugs

When using `sim_drive_params = simdrive.SimDriveParams(missed_trace_correction=True)`, an instance of `cycle.TypedCycle()` (obtainable by using `cycle.Cycle().get_numba_cyc()`) must be used to get the right behavior. For some reason, when using an instance of `cycle.Cycle()` the `secs` array elements are not being overwritten as they should be.  

