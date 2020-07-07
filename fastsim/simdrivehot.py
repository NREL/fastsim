"""Module containing classes and methods for simulating vehicle drive cycle.
For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
import pandas as pd
import re
import sys
from numba import jitclass                 # import the decorator
from numba import float64, int32, bool_, types    # import the types
import warnings
warnings.simplefilter('ignore')

# local modules
from fastsim import globalvars as gl
from fastsim.simdrive import SimDriveCore, spec

# things to model:
# cabin temperature
# power associated with maintaining cabin temperature as variable aux load for both cooling and heating
# solar heat load
# cabin interaction with convection (flat plate?)
# engine temperature
# engine interaction with ambient via convection
# radiator behavior for hot engine -- impose steady temperature by whatever heat loss is needed
# radiator aux load -- maybe constant times heat loss via radiator (e.g. 0.1?)
# impact of engine temperature on engine efficiency

hotspec = spec + [('teAmbDegC', float64), # ambient temperature
                    ('teFcDegC', float64[:]), # fuel converter temperature
                    ('fcEffAdj', float64[:]), # fuel converter temperature efficiency correction
                    ('fcHeatGenKw', float64[:]), # fuel converter heat generation
                    ('fcConvToAmbKw', float64[:]), # fuel converter convection to ambient
                    ('fcToHtrKw', float64[:]), # fuel converter heat loss to heater core
                    ('fcThrmMass', float64), # fuel converter thermal mass (kJ/(kg*K))
                    ('teCabDegC', float64[:]), # cabin temperature
                    ('cabSolarKw', float64[:]), # cabin solar load
                    ('cabConvToAmbKw', float64[:]), # cabin convection to ambient
                    ('cabFromHtrKw', float64[:]), # cabin heat from heater 
                    ('cabThrmMass', float64)  # cabin thermal mass (kJ/(kg*K))
                    ]

@jitclass(hotspec)
class SimDriveHotJit(SimDriveCore):
    """Class containing methods for running FASTSim vehicle 
    fuel economy simulations with thermal behavior. 
    
    This class is not compiled and will run slower for large batch runs."""
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle fuel economy simulations with thermal behavior. 
    This class will be faster for large batch runs. 
    Arguments:
    ----------
    cyc: cycle.TypedCycle instance. Can come from cycle.Cycle.get_numba_cyc
    veh: vehicle.TypedVehicle instance. Can come from vehicle.Vehicle.get_numba_veh"""

    __init_core = SimDriveCore.__init__ # workaround for initializing super class within jitclass
    
    def __init__(self, cyc, veh, teAmbDegC=22, teFcInitDegC=90, fcThrmMass=100, teCabInitDegC=22, cabThmMass=5):
        """Initialize time array variables that are not used in base SimDrive."""
        self.__init_core(cyc, veh)
        
        len_cyc = len(self.cyc.cycSecs)
        self.teFcDegC = np.zeros(len_cyc, dtype=np.float64)
        self.fcEffAdj = np.zeros(len_cyc, dtype=np.float64)
        self.fcHeatGenKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcConvToAmbKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcToHtrKw = np.zeros(len_cyc, dtype=np.float64)
        self.teCabDegC = np.zeros(len_cyc, dtype=np.float64)
        self.cabSolarKw = np.zeros(len_cyc, dtype=np.float64)
        self.cabConvToAmbKw = np.zeros(len_cyc, dtype=np.float64)
        self.cabFromHtrKw = np.zeros(len_cyc, dtype=np.float64)
        
        # scalars
        self.teFcDegC[0] = teFcInitDegC
        self.fcThrmMass = fcThrmMass

    def sim_drive(self, *args):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        args[0]: first argument in *args is initial SOC for electrified vehicles.  
            Leave empty for default value.  Otherwise, must be between 0 and 1.
            Numba's jitclass does not support keyword args so this is allows for optionally
            passing initSoc."""

        if len(args) > 0:
            initSoc = args[0] # set initSoc
            if (initSoc != -1) and (initSoc > 1.0 or initSoc < 0.0):
                    print('Must enter a valid initial SOC between 0.0 and 1.0')
                    print('Running standard initial SOC controls')
                    initSoc = -1 # override initSoc if invalid value is used
            elif initSoc == -1:
                print('initSoc = -1 passed to drive default SOC behavior.')
        else:
            initSoc = -1 # -1 enforces the default SOC behavior
    
        if self.veh.vehPtType == 1: # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0 # this initSoc has no impact on results
            
            self.sim_drive_walk(initSoc)

        elif self.veh.vehPtType == 2 and initSoc == -1:  # HEV 

            #####################################
            ### Charge Balancing Vehicle SOC ###
            #####################################

            # Charge balancing SOC for HEV vehicle types. Iterating initsoc and comparing to final SOC.
            # Iterating until tolerance met or 30 attempts made.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            ess2fuelKwh = 1.0
            sim_count = 0
            while ess2fuelKwh > self.veh.essToFuelOkError and sim_count < 30:
                sim_count += 1
                self.sim_drive_walk(initSoc)
                fuelKj = np.sum(self.fsKwOutAch * self.cyc.secs)
                roadwayChgKj = np.sum(self.roadwayChgKwOutAch * self.cyc.secs)
                ess2fuelKwh = np.abs((self.soc[0] - self.soc[-1]) * 
                    self.veh.maxEssKwh * 3600 / (fuelKj + roadwayChgKj))
                initSoc = min(1.0, max(0.0, self.soc[-1]))
                        
            self.sim_drive_walk(initSoc)

        elif (self.veh.vehPtType == 3 and initSoc == -1) or (self.veh.vehPtType == 4 and initSoc == -1): # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = self.veh.maxSoc
            
            self.sim_drive_walk(initSoc)

        else:
            
            self.sim_drive_walk(initSoc)
        
        self.set_post_scalars()            
   
    def sim_drive_step(self, *args):
        """
        Override of sim_drive_step for thermal modeling.  
        Step through 1 time step.
        Arguments:
        ----------
        *args: variables to be overridden outside of sim_drive_step (experimental)"""

        self.set_thermal_calcs(self.i)
        self.set_misc_calcs(self.i, *args)
        self.set_comp_lims(self.i)
        self.set_power_calcs(self.i)
        self.set_ach_speed(self.i)
        self.set_hybrid_cont_calcs(self.i)
        self.set_fc_forced_state(self.i) # can probably be *mostly* done with list comprehension in post processing
        self.set_hybrid_cont_decisions(self.i)
        self.set_fc_power(self.i)

        self.i += 1 # increment time step counter

    def set_thermal_calcs(self, i):
        """Sets temperature calculations at time step 'i'
        Arguments:
        ----------
        i: index of time step"""

        # Constitutive equations for fuel converter
        self.fcHeatGenKw[i-1] = self.fcKwInAch[i-1] - self.fcKwOutAch[i-1]
        self.fcConvToAmbKw[i-1] = 666e-3 # placeholder
        self.fcToHtrKw[i-1] = 666e-3  # placeholder
        # Energy balance for fuel converter
        self.teFcDegC[i] = self.teFcDegC[i-1] + (
            self.fcHeatGenKw[i-1] - self.fcConvToAmbKw[i-1] - self.fcToHtrKw[i-1]
        ) / self.fcThrmMass * self.cyc.secs[i-1]
        
