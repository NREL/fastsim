from numba.experimental import jitclass
import numpy as np

from .simdrive import SimDriveClassic, SimAccelTest, SimDriveParamsClassic
from .cyclejit import Cycle
from .vehiclejit import Vehicle
from .buildspec import build_spec
from . import parametersjit as paramsjit
from . import params


param_spec = build_spec(SimDriveParamsClassic())

@jitclass(param_spec)
class SimDriveParams(SimDriveParamsClassic):
    pass

sim_drive_spec = build_spec(SimDriveClassic(Cycle('udds'), Vehicle(1, verbose=False)))

@jitclass(sim_drive_spec)
class SimDriveJit(SimDriveClassic):
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle fuel economy simulations. This class will be 
    faster for large batch runs.
    Arguments:
    ----------
    cyc: cycle.CycleJit instance. Can come from cycle.Cycle.get_numba_cyc
    veh: vehicle.VehicleJit instance. Can come from vehicle.Vehicle.get_numba_veh"""

    def __init_objects__(self, cyc, veh):        
        self.veh = veh
        self.cyc = cyc.copy() # this cycle may be manipulated
        self.cyc0 = cyc.copy() # this cycle is not to be manipulated
        self.sim_params = SimDriveParams()
        self.props = paramsjit.PhysicalPropertiesJit()
                
@jitclass(sim_drive_spec)
class SimAccelTestJit(SimAccelTest):
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle acceleration simulation. This class will be 
    faster for large batch runs."""

    def __init_objects__(self, cyc, veh):        
        self.veh = veh
        self.cyc = cyc.copy() # this cycle may be manipulated
        self.cyc0 = cyc.copy() # this cycle is not to be manipulated
        self.sim_params = SimDriveParams()
        self.props = paramsjit.PhysicalPropertiesJit()