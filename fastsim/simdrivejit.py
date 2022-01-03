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

    def sim_drive(self, initSoc=-1, auxInKwOverride=np.zeros(1, dtype=np.float64)):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        initSoc: initial SOC for electrified vehicles.  
            Leave empty for default value.  Otherwise, must be between 0 and 1.
            Numba's jitclass does not support keyword args so this allows for optionally
            passing initSoc as positional argument.
            auxInKw: auxInKw override.  Array of same length as cyc.cycSecs.  
                Default of np.zeros(1) causes veh.auxKw to be used. If zero is actually
                desired as an override, either set veh.auxKw = 0 before instantiaton of
                SimDrive*, or use `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting
                the final value to non-zero prevents override mechanism.  
        """

        if (auxInKwOverride == 0).all():
            auxInKwOverride = self.auxInKw

        self.hev_sim_count = 0

        if (initSoc != -1):
            if (initSoc > 1.0 or initSoc < 0.0):
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = -1 # override initSoc if invalid value is used
            else:
                self.sim_drive_walk(initSoc, auxInKwOverride)
    
        elif self.veh.vehPtType == 1: # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0 # this initSoc has no impact on results
            
            self.sim_drive_walk(initSoc, auxInKwOverride)

        elif self.veh.vehPtType == 2 and initSoc == -1:  # HEV 

            #####################################
            ### Charge Balancing Vehicle SOC ###
            #####################################

            # Charge balancing SOC for HEV vehicle types. Iterating initsoc and comparing to final SOC.
            # Iterating until tolerance met or 30 attempts made.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            ess2fuelKwh = 1.0
            while ess2fuelKwh > self.veh.essToFuelOkError and self.hev_sim_count < self.sim_params.sim_count_max:
                self.hev_sim_count += 1
                self.sim_drive_walk(initSoc, auxInKwOverride)
                fuelKj = np.sum(self.fsKwOutAch * self.cyc.secs)
                roadwayChgKj = np.sum(self.roadwayChgKwOutAch * self.cyc.secs)
                if (fuelKj + roadwayChgKj) > 0:
                    ess2fuelKwh = np.abs((self.soc[0] - self.soc[-1]) *
                                         self.veh.maxEssKwh * 3600 / (fuelKj + roadwayChgKj))
                else:
                    ess2fuelKwh = 0.0
                initSoc = min(1.0, max(0.0, self.soc[-1]))
                        
            self.sim_drive_walk(initSoc, auxInKwOverride)

        elif (self.veh.vehPtType == 3 and initSoc == -1) or (self.veh.vehPtType == 4 and initSoc == -1): # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = self.veh.maxSoc
            
            self.sim_drive_walk(initSoc, auxInKwOverride)

        else:
            
            self.sim_drive_walk(initSoc, auxInKwOverride)
        
        self.set_post_scalars()     
                
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

    def sim_drive(self):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType."""

        if self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_walk(initSoc)

        elif self.veh.vehPtType == 2:  # HEV

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_walk(initSoc)

        else:

            # If EV, initializing initial SOC to maximum SOC.
            initSoc = self.veh.maxSoc
            self.sim_drive_walk(initSoc)

        self.set_post_scalars()
