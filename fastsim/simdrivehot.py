"""Module containing classes and methods for simulating vehicle drive cycle.
For example usage, see ../README.md"""

### Import necessary python modules
import os
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import re
import sys
from numba import jitclass                 # import the decorator
from numba import float64, int32, bool_, types   # import the types
import warnings
warnings.simplefilter('ignore')

# local modules
from fastsim.simdrive import SimDriveClassic, sim_drive_spec

# Fluid Properties for calculations
teAirForPropsDegC = np.arange(-20, 140, 20, dtype=np.float64) # deg C
rhoAirArray = np.array([1.38990154, 1.28813317, 1.20025098, 1.12359437, 
                        1.05614161, 0.99632897, 0.94292798, 0.89496013], 
                        dtype=np.float64)
                        # density of air [kg / m ** 3]
kAirArray = np.array([0.02262832, 0.02416948, 0.02567436, 0.02714545, 0.02858511,
                      0.02999558, 0.03137898, 0.03273731], dtype=np.float64)
                      # thermal conductivity of air [W / (m * K)]
cpAirArray = np.array([1003.15536494, 1003.71709112, 1004.49073603, 1005.51659149,
                       1006.82540109, 1008.43752504, 1010.36365822, 1012.60611422], 
                       dtype=np.float64) / 1e3
                       # specific heat of air [kJ / (kg * K)]
hAirArray = np.array([253436.58748754, 273504.99629716, 293586.68523714, 313686.30935277,
                      333809.23760193, 353961.34965289, 374148.83386308, 394378.00634841], 
                      dtype=np.float64) / 1e3
                      # specific enthalpy of air [kJ / kg]
PrAirArray = np.array([0.71884378, 0.7159169, 0.71334768, 0.71112444, 0.70923784,
                       0.7076777, 0.70643177, 0.70548553], 
                       dtype=np.float64)
                        # Prandtl number of air
muAirArray = np.array([1.62150624e-05, 1.72392601e-05, 1.82328655e-05, 1.91978833e-05,
                       2.01362020e-05, 2.10495969e-05, 2.19397343e-05, 2.28081749e-05], 
                       dtype=np.float64)
                       # Dynamic viscosity of air [Pa * s]

re_array = np.array([0, 4, 40, 4e3, 40e3], dtype=np.float64)

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

# thinking about cabin model and interaction with heater:
# cabin needs to leak!
# treat it as a flat plate with vehicle dimensions and thermal mass
# heater core: error relative to target, --nominal coolant flow rate--, no recirc, 
# nominal air flow rate (cabin exchanges per min?) at some assumed effectiveness -- tunable?

hotspec = sim_drive_spec + [('teAmbDegC', float64[:]), # ambient temperature
                    ('teFcDegC', float64[:]), # fuel converter temperature
                    ('fcEffAdj', float64[:]), # fuel converter temperature efficiency correction
                    ('fcHeatGenKw', float64[:]), # fuel converter heat generation
                    ('fcConvToAmbKw', float64[:]), # fuel converter convection to ambient
                    ('fcToHtrKw', float64[:]), # fuel converter heat loss to heater core
                    ('fcThrmMass', float64), # fuel converter thermal mass [kJ/K]
                    ('teCabDegC', float64[:]), # cabin temperature
                    ('cabSolarKw', float64[:]), # cabin solar load
                    ('cabConvToAmbKw', float64[:]), # cabin convection to ambient
                    ('cabFromHtrKw', float64[:]), # cabin heat from heater 
                    ('cabThrmMass', float64),  # cabin thermal mass [kJ/K]
                    ('fcDiam', float64), # engine characteristic length [m] for heat transfer calcs 
                    ('fcSurfArea', float64), # engine surface area for heat transfer calcs
                    ('hFcToAmbStop', float64), # heat transfer coeff [W / (m ** 2 * K)] from eng to ambient during stop
                    ('hFcToAmbRad', float64), # heat transfer coeff [W / (m ** 2 * K)] from eng to ambient if radiator is active
                    ('hFcToAmb', float64[:]), # heat transfer coeff [W / (m ** 2 * K)] to amb after arbitration
                    ('fcCombToThrmlMassKw', float64), # fraction of combustion heat that goes to FC thermal mass
                    # remainder goes to environment (e.g. via tailpipe)
                    ('teFcInitDegC', float64), # fuel converter initial temperature [deg C]
                    ('teCabInitDegC', float64), # cabin inital temperature [deg C]
                    ('teTStatSTODegC', float64), # temperature [ºC] at which thermostat starts to open 
                    ('teTStatFODegC', float64), # temperature [ºC] at which thermostat is fully open 
                    ('teTStatDeltaDegC', float64), # temperature delta [ºC] over which thermostat is partially open
                    ('radiator_eff', float64), # radiator effectiveness -- ratio of active heat rejection from radiator to passive heat rejection
                    ('fcTempEffOffset', float64), # offset for scaling FC efficiency w.r.t. to temperature
                    ('fcTempEffSlope', float64), # slope for scaling FC efficiency w.r.t. to temperature
                    ]

class SimDriveHot(SimDriveClassic):
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

    def __init__(self, cyc, veh, teAmbDegC, teFcInitDegC=90, teCabInitDegC=22):
        """Initialize time array variables that are not used in base SimDrive.
        Arguments:
        ----------
        teAmbDegC: array of ambient temperatures [C].  Must be declared with 
            dtype=np.float64, e.g. np.zeros(len(cyc.cycSecs, dtype=np.float64)).
        teFcInitDegC: (optional) fuel converter initial temperature [C]
        teCabInitDegC: (optional) cabin initial temperature [C]"""
        self.__init_objects__(cyc, veh)
        self.init_arrays()
        self.init_thermal_scalars(teFcInitDegC, teCabInitDegC)
        self.init_thermal_arrays(teAmbDegC)

    def init_thermal_scalars(self, teFcInitDegC, teCabInitDegC):
        # scalars
        self.teFcInitDegC = teFcInitDegC # for persistence through iteration
        self.teCabInitDegC = teCabInitDegC # for persistence through iteration
        self.fcThrmMass = 100 
        self.fcDiam = 1 
        self.fcSurfArea = np.pi * self.fcDiam ** 2 / 4
        self.cabThrmMass = 5
        self.hFcToAmbStop = 50
        self.fcCombToThrmlMassKw = 0.5 
        self.teTStatSTODegC = 85
        self.teTStatDeltaDegC = 5
        self.teTStatFODegC = self.teTStatSTODegC + self.teTStatDeltaDegC
        self.radiator_eff = 5
        self.fcTempEffOffset = 0.1
        self.fcTempEffSlope = 0.01

    def init_thermal_arrays(self, teAmbDegC):
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
        self.hFcToAmb = np.zeros(len_cyc, dtype=np.float64)
        # this block ~should~ prevent the __init__ call in sim_drive_walk from 
        # overriding the prescribed ambient temperature
        self.teFcDegC[0] = self.teFcInitDegC
        self.teCabDegC[0] = self.teCabInitDegC

        self.teAmbDegC = teAmbDegC

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
   
    def sim_drive_walk(self, initSoc):
        """Receives second-by-second cycle information, vehicle properties, 
        and an initial state of charge and runs sim_drive_step to perform a 
        backward facing powertrain simulation. Method 'sim_drive' runs this
        iteratively to achieve correct SOC initial and final conditions, as 
        needed.

        Arguments
        ------------
        initSoc (optional): initial battery state-of-charge (SOC) for electrified vehicles"""

        ############################
        ###   Loop Through Time  ###
        ############################

        ###  Assign First Values  ###
        ### Drive Train
        # reinitialize arrays for each new run
        self.init_arrays() # reinitialize arrays for each new run
        self.init_thermal_arrays(self.teAmbDegC)
        self.cycMet[0] = 1
        self.curSocTarget[0] = self.veh.maxSoc
        self.essCurKwh[0] = initSoc * self.veh.maxEssKwh
        self.soc[0] = initSoc

        self.i = 1  # time step counter
        while self.i < len(self.cyc.cycSecs):
            self.sim_drive_step()

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

        # most of the thermal equations are at [i-1] because the various thermally 
        # sensitive component efficiencies dependent on the [i] temperatures, but 
        # these are in turn dependent on [i-1] heat transfer processes  
        # Constitutive equations for fuel converter
        self.fcHeatGenKw[i] = self.fcCombToThrmlMassKw * (self.fcKwInAch[i-1] - self.fcKwOutAch[i-1])
        teFcFilmDegC = 0.5 * (self.teFcDegC[i-1] + self.teAmbDegC[i-1])
        Re_fc = np.interp(teFcFilmDegC, teAirForPropsDegC, rhoAirArray) \
            * self.mpsAch[i-1] * self.fcDiam / \
            np.interp(teFcFilmDegC, teAirForPropsDegC, muAirArray) 
        # density * speed * diameter / dynamic viscosity

        def get_sphere_conv_params(Re):
            """Given Reynolds number, return C and m.
            Nusselt number coefficients from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44"""
            if Re < 4:
                C = 0.989
                m = 0.330
            elif Re < 40:
                C = 0.911
                m = 0.385
            elif Re < 4e3:
                C = 0.683
                m = 0.466
            elif Re < 40e3:
                C = 0.193
                m = 0.618
            else:
                C = 0.027
                m = 0.805
            return [C, m]

        # calculate heat transfer coeff. from engine to ambient [W / (m ** 2 * K)]
        if self.mpsAch[i-1] < 1:
            # if stopped, scale based on thermostat opening and constant convection
            self.hFcToAmb[i] = np.interp(self.teFcDegC[i-1], 
                [self.teTStatSTODegC, self.teTStatFODegC],
                [self.hFcToAmbStop, self.hFcToAmbStop * self.radiator_eff])
        else:
            # if moving, scale based on speed dependent convection and thermostat opening
            self.hFcToAmb[i] = np.interp(self.teFcDegC[i-1],
                [self.teTStatSTODegC, self.teTStatFODegC],
                [self.hFcToAmbStop, self.hFcToAmbStop * self.radiator_eff])
            # Nusselt number coefficients from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            hFcToAmbSphere = (get_sphere_conv_params(Re_fc)[0] * Re_fc ** get_sphere_conv_params(Re_fc)[1]) * \
                np.interp(teFcFilmDegC, teAirForPropsDegC, PrAirArray) ** (1 / 3) * \
                np.interp(teFcFilmDegC, teAirForPropsDegC, kAirArray) / self.fcDiam
            self.hFcToAmb[i] = np.interp(self.teFcDegC[i-1],
                [self.teTStatSTODegC, self.teTStatFODegC],
                [hFcToAmbSphere, hFcToAmbSphere * self.radiator_eff])

        self.fcConvToAmbKw[i] = self.hFcToAmb[i] * 1e-3 * self.fcSurfArea * (self.teFcDegC[i-1] - self.teAmbDegC[i-1])
        self.fcToHtrKw[i] = 0  # placeholder
        # Energy balance for fuel converter
        self.teFcDegC[i] = self.teFcDegC[i-1] + (
            self.fcHeatGenKw[i] - self.fcConvToAmbKw[i] - self.fcToHtrKw[i]
            ) / self.fcThrmMass * self.cyc.secs[i]
        
    def set_fc_power(self, i):
        """Sets fcKwOutAch and fcKwInAch.
        Arguments
        ------------
        i: index of time step"""

        if self.veh.maxFuelConvKw == 0:
            self.fcKwOutAch[i] = 0

        elif self.veh.fcEffType == 4:
            self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(
                0, self.mcElecKwInAch[i] + self.auxInKw[i] - self.essKwOutAch[i] - self.roadwayChgKwOutAch[i]))

        elif self.veh.noElecSys == True or self.veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
            self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(
                0, self.transKwInAch[i] - self.mcMechKwOutAch[i] + self.auxInKw[i]))

        else:
            self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(
                0, self.transKwInAch[i] - self.mcMechKwOutAch[i]))

        if self.veh.maxFuelConvKw == 0:
            self.fcKwOutAch_pct[i] = 0
        else:
            self.fcKwOutAch_pct[i] = self.fcKwOutAch[i] / \
                self.veh.maxFuelConvKw

        if self.fcKwOutAch[i] == 0:
            self.fcKwInAch[i] = 0
            self.fcKwOutAch_pct[i] = 0

        else:
            # 0 to 1 scaling for multiplying efficiency to be dependent on temperature.
            self.fcEffAdj[i] = max(0.1, # assuming that 90% is max temp-dependent efficiency reduction
                                min(1, 
                                     self.fcTempEffOffset + self.fcTempEffSlope * self.teFcDegC[i]
                                    )
                                )
            if self.fcKwOutAch[i] == self.veh.fcMaxOutkW:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / (self.veh.fcEffArray[-1] * self.fcEffAdj[i])
            else:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / \
                    (self.veh.fcEffArray[max(1, np.argmax(self.veh.fcKwOutArray > min(self.fcKwOutAch[i], self.veh.fcMaxOutkW - 0.001)) - 1)]) \
                        / self.fcEffAdj[i]

        self.fsKwOutAch[i] = self.fcKwInAch[i]

        self.fsKwhOutAch[i] = self.fsKwOutAch[i] * \
            self.cyc.secs[i] * (1 / 3600.0)


@jitclass(hotspec)
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

    
