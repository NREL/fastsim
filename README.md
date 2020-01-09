# Description
This repo houses the pythonic flavor of FASTSim which is based on the original Excel implementation. Effort will be made to keep the core methodology between this software and the Excel flavor in line with one another. Other FASTSim flavors may spin off as variations on this core functionality, but these should integrated back into master if there is any intent of persistence.

All classes and methods are self-documented.  

# Usage
To run the code:
1. Install environment.yml per https://github.nrel.gov/MBAP/arnaud/wiki/Conda-Environments and activate the new environment
2. Using a Jupyter Notebook or Jupyter lab, run docs/demo.ipynb.

# Testing

## Against Previous Python Version
Run the file 'tests/test26veh3cyc.py' to compare FASTSim back to the master branch version from 17 December 2019.  For timing comparisons, run 'tests/test26veh3cyc_CPUtime.py'.  

## Against Excel FASTSim
This has not been implemented yet

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

