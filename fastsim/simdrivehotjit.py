# python modules
import numpy as np
from numba.experimental import jitclass

# fastsim modules
from .buildspec import build_spec
from .simdrivehot import SimDriveHot
from .vehicle import Vehicle
from .cycle import Cycle
from .simdrivejit import SimDriveParams
from .simdrivehot import AirProperties, ConvectionCalcs, VehicleThermal
from . import parametersjit


@jitclass(build_spec(VehicleThermal()))
class VehicleThermalJit(VehicleThermal):
    """Numba jitclass version of VehicleThermal"""
    pass

@jitclass(build_spec(ConvectionCalcs()))
class ConvectionCalcsJit(ConvectionCalcs):
    "Numba JIT version of ConvectionCalcs."
    pass

@jitclass(build_spec(AirProperties()))
class AirPropertiesJit(AirProperties):
    """Numba jitclass version of FluidProperties"""
    pass

_hotspec = build_spec(
    SimDriveHot(Cycle('udds'), 
    Vehicle(1, verbose=False), 
    teAmbDegC=np.ones(len(Cycle('udds').cycSecs)))
)
@jitclass(_hotspec)
class SimDriveHotJit(SimDriveHot):
    """JTI-compiled class containing methods for running FASTSim vehicle 
    fuel economy simulations with thermal behavior. 

    Inherits everything from SimDriveHot
    
    This class is not compiled and will run slower for large batch runs."""
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle fuel economy simulations with thermal behavior. 
    This class will be faster for large batch runs. 
    Arguments:
    ----------
    cyc: cycle.TypedCycle instance. Can come from cycle.Cycle.get_numba_cyc
    veh: vehicle.TypedVehicle instance. Can come from vehicle.Vehicle.get_numba_veh"""

    def __init_objects__(self, cyc, veh):
        """Override of method in super class (SimDriveHot)."""
        self.veh = veh
        self.cyc = cyc.copy()  # this cycle may be manipulated
        self.cyc0 = cyc.copy()  # this cycle is not to be manipulated
        self.sim_params = SimDriveParams()
        self.props = parametersjit.PhysicalPropertiesJit()
        self.air = AirPropertiesJit()
        self.conv_calcs = ConvectionCalcsJit()
        self.vehthrm = VehicleThermalJit()


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

        self.hev_sim_count = 0 # probably not necassary since numba initializes int vars as 0, but adds clarity

        if (initSoc != -1):
            if (initSoc > 1.0 or initSoc < 0.0):
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = -1  # override initSoc if invalid value is used
            else:
                self.sim_drive_walk(initSoc, auxInKwOverride)

        elif self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0  # this initSoc has no impact on results

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

        elif (self.veh.vehPtType == 3 and initSoc == -1) or (self.veh.vehPtType == 4 and initSoc == -1):  # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = self.veh.maxSoc

            self.sim_drive_walk(initSoc, auxInKwOverride)

        else:

            self.sim_drive_walk(initSoc, auxInKwOverride)

        self.set_post_scalars()

