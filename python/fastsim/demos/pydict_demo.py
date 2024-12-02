"""
Script demonstrating how to use to_pydict() and from_pydict() for FASTSim
classes.
"""
import fastsim as fsim
from fastsim import fastsimrust as fsr

# load 2012 Ford F-150 Ecoboost from file
veh = fsim.vehicle.Vehicle.from_file("2017_Ford_F-150_Ecoboost.csv").to_rust()
print(veh)

# saving vehicle as pydict
veh_pydict = veh.to_pydict()
print(veh_pydict)

# getting fastsim vehicle from pydict
from_pydict_veh = fsr.RustVehicle.from_pydict(veh_pydict)
print(from_pydict_veh)

# spot checking to see if the pre-pydict and post-pydict Rust Vehicles are the same
assert(veh.scenario_name==from_pydict_veh.scenario_name)
assert(veh.drag_coef==from_pydict_veh.drag_coef)