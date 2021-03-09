"""Module containing classes and methods for simulating vehicle drive cycle.
For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
import pandas as pd
import re
import sys
from numba.experimental import jitclass                 # import the decorator
from numba import float64, int32, bool_, types   # import the types
import warnings
warnings.simplefilter('ignore')

# local modules
from fastsim import params, utils, simdrive
from fastsim.simdrive import SimDriveClassic, SimDriveParams
from fastsim.cycle import Cycle
from fastsim.vehicle import Vehicle
from fastsim.buildspec import build_spec


class AirProperties(object):
    """Fluid Properties for calculations"""
    def __init__(self):
        # array at of temperatures at which properties are evaluated ()
        self._te_array_degC = np.arange(-20, 140, 20, dtype=np.float64) 
        # thermal conductivity of air [W / (m * K)]
        self._k_Array = np.array([0.02262832, 0.02416948, 0.02567436, 0.02714545, 0.02858511,
                            0.02999558, 0.03137898, 0.03273731], dtype=np.float64)
        # specific heat of air [kJ / (kg * K)]
        self._cp_Array = np.array([1003.15536494, 1003.71709112, 1004.49073603, 1005.51659149,
                            1006.82540109, 1008.43752504, 1010.36365822, 1012.60611422], 
                            dtype=np.float64) / 1e3
        # specific enthalpy of air [kJ / kg]
        self._h_Array = np.array([253436.58748754, 273504.99629716, 293586.68523714, 313686.30935277,
                            333809.23760193, 353961.34965289, 374148.83386308, 394378.00634841], 
                            dtype=np.float64) / 1e3
        # Prandtl number of air
        self._Pr_Array = np.array([0.71884378, 0.7159169, 0.71334768, 0.71112444, 0.70923784,
                            0.7076777, 0.70643177, 0.70548553], 
                            dtype=np.float64)
        # Dynamic viscosity of air [Pa * s]
        self._mu_Array = np.array([1.62150624e-05, 1.72392601e-05, 1.82328655e-05, 1.91978833e-05,
                            2.01362020e-05, 2.10495969e-05, 2.19397343e-05, 2.28081749e-05], 
                            dtype=np.float64)

    def get_rho(self, T, h=180):
        return utils.get_rho_air(T, h)

    def get_k(self, T):
        return np.interp(T, self._te_array_degC, self._k_Array)

    def get_cp(self, T):
        return np.interp(T, self._te_array_degC, self._cp_Array)

    def get_h(self, T):
        return np.interp(T, self._te_array_degC, self._h_Array)

    def get_Pr(self, T):
        return np.interp(T, self._te_array_degC, self._Pr_Array)

    def get_mu(self, T):
        return np.interp(T, self._te_array_degC, self._mu_Array)


@jitclass(build_spec(AirProperties()))
class AirPropertiesJit(AirProperties):
    """Numba jitclass version of FluidProperties"""
    pass

class VehicleThermal(object):
    """Class for containing vehicle thermal (and related) parameters."""
    def __init__(self):
        """Initial default values for vehicle thermal parameters."""
        # parameter fuel converter thermal mass [kJ/K]
        self.fcThrmMass = 100.0
        # parameter for ngine characteristic length [m] for heat transfer calcs
        self.fcDiam = 1.0
        # parameter for ngine surface area for heat transfer calcs
        self.fcSurfArea = np.pi * self.fcDiam ** 2.0 / 4.0
        # parameter for cabin thermal mass [kJ/K]
        self.cabThrmMass = 5.0
        # cabin length [m], modeled as a flat plate
        self.cabLength = 2.5 
        # cabin width [m], modeled as a flat plate
        self.cabWidth = 1.75
        # parameter for eat transfer coeff [W / (m ** 2 * K)] from eng to ambient during stop
        self.hFcToAmbStop = 50.0
        # parameter for fraction of combustion heat that goes to fuel converter (engine) 
        # thermal mass. Remainder goes to environment (e.g. via tailpipe)
        self.fcCombToThrmlMassFrac = 0.5 
        # parameter for temperature [ºC] at which thermostat starts to open
        self.teTStatSTODegC = 85.0
        # temperature delta [ºC] over which thermostat is partially open
        self.teTStatDeltaDegC = 5.0
        # derived temperature [ºC] at which thermostat is fully open
        self.teTStatFODegC = self.teTStatSTODegC + self.teTStatDeltaDegC
        # radiator effectiveness -- ratio of active heat rejection from 
        # radiator to passive heat rejection
        self.radiator_eff = 5.0
        # offset for scaling FC efficiency w.r.t. to temperature
        self.fcTempEffOffset = 0.1
        # slope for scaling FC efficiency w.r.t. to temperature
        self.fcTempEffSlope = 0.01
        # HVAC model 'internal' or 'external' w.r.t. fastsim
        self.hvac_model = 'external'
        # cabin model 'internal' or 'external' w.r.t. fastsim
        self.cabin_model = 'internal'


@jitclass(build_spec(VehicleThermal()))
class VehicleThermalJit(VehicleThermal):
    """Numba jitclass version of VehicleThermal"""
    pass

# things to model:
# cabin temperature
    # use cabin volume (usually readily available) for thermal mass
    # use wheel base and some nominal width (probably a constant mostly???) to calculate flat plate heat transfer coefficient
# power associated with maintaining cabin temperature as variable aux load for both cooling and heating
# solar heat load
# cabin interaction with convection (flat plate?)
# engine temperature
# engine interaction with ambient via convection
# radiator behavior for hot engine -- impose steady temperature by whatever heat loss is needed
# radiator aux load -- maybe constant times heat loss via radiator (e.g. 0.1?) 
# ... aka aux load penalty per radiator heat removal 
# impact of engine temperature on engine efficiency -- check

# thinking about cabin model and interaction with heater:
# cabin needs to have "leak" as heat exchange with ambient
# treat it as a flat plate with vehicle dimensions and thermal mass
# heater core: error relative to target, --nominal coolant flow rate--, no recirc, 
# nominal air flow rate (cabin exchanges per min?) at some assumed effectiveness -- tunable?
# aux load penalty per battery heat removal

# battery model 
# *** parameters: specific heat (c_p, [J/(kg-K)]) (since there is already a mass)
# *** heat transfer coefficient per ESS volume (need density if not already included) that maybe is active only
# when vehicle is stationary/off
# *** similar heat tranfer coefficient for when battery is hot (maybe ramp up w.r.t. temperature like thermostat behavior)
# *** solve for heat gen in battery (should just be simple subtraction)
# *** aux load penalty per battery heat removal

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

    def __init__(self, cyc, veh, teAmbDegC, teFcInitDegC=90.0, teCabInitDegC=22.0):
        """Initialize time array variables that are not used in base SimDrive.
        Arguments:
        ----------
        teAmbDegC: array of ambient temperatures [C].  Must be declared with 
            dtype=np.float64, e.g. np.zeros(len(cyc.cycSecs, dtype=np.float64)).
        teFcInitDegC: (optional) fuel converter initial temperature [C]
        teCabInitDegC: (optional) cabin initial temperature [C]"""
        self.__init_objects__(cyc, veh)
        self.teFcInitDegC = teFcInitDegC # for persistence through iteration
        self.teCabInitDegC = teCabInitDegC # for persistence through iteration
        self.init_arrays()
        self.init_thermal_arrays(teAmbDegC)

    def __init_objects__(self, cyc, veh):
        self.veh = veh
        self.cyc = cyc.copy()  # this cycle may be manipulated
        self.cyc0 = cyc.copy()  # this cycle is not to be manipulated
        self.sim_params = simdrive.SimDriveParamsClassic()
        self.props = params.PhysicalProperties()
        self.air = AirProperties()
        self.vehthrm = VehicleThermal()

    def init_thermal_arrays(self, teAmbDegC):
        """Arguments:
        teAmbDegC : Float, ambient temperature array for cycle"""
        len_cyc = len(self.cyc.cycSecs)
        # fuel converter (engine) temperature
        self.teFcDegC = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter temperature efficiency correction
        self.fcEffAdj = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter heat generation
        self.fcHeatGenKw = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter convection to ambient
        self.fcConvToAmbKw = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter heat loss to heater core
        self.fcToHtrKw = np.zeros(len_cyc, dtype=np.float64)
        # cabin temperature
        self.teCabDegC = np.zeros(len_cyc, dtype=np.float64)
        # cabin solar load
        self.cabSolarKw = np.zeros(len_cyc, dtype=np.float64)
        # cabin convection to ambient
        self.cabConvToAmbKw = np.zeros(len_cyc, dtype=np.float64)
        # heat transfer coeff [W / (m ** 2 * K)] to amb after arbitration
        self.hFcToAmb = np.zeros(len_cyc, dtype=np.float64)
        
        # this block ~should~ prevent the __init__ call in sim_drive_walk from 
        # overriding the prescribed ambient temperature
        self.teFcDegC[0] = self.teFcInitDegC
        self.teCabDegC[0] = self.teCabInitDegC

        self.teAmbDegC = teAmbDegC

    def sim_drive_walk(self, initSoc, auxInKwOverride=np.zeros(1, dtype=np.float64)):
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
        self.fcHeatGenKw[i] = self.vehthrm.fcCombToThrmlMassFrac * (self.fcKwInAch[i-1] - self.fcKwOutAch[i-1])
        teFcFilmDegC = 0.5 * (self.teFcDegC[i-1] + self.teAmbDegC[i-1])
        Re_fc = self.air.get_rho(teFcFilmDegC) * self.mpsAch[i-1] * self.vehthrm.fcDiam / \
            self.air.get_mu(teFcFilmDegC) 
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
                [self.vehthrm.teTStatSTODegC, self.vehthrm.teTStatFODegC],
                [self.vehthrm.hFcToAmbStop, self.vehthrm.hFcToAmbStop * self.vehthrm.radiator_eff])
        else:
            # if moving, scale based on speed dependent convection and thermostat opening
            # Nusselt number coefficients from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            hFcToAmbSphere = (get_sphere_conv_params(Re_fc)[0] * Re_fc ** get_sphere_conv_params(Re_fc)[1]) * \
                self.air.get_Pr(teFcFilmDegC) ** (1 / 3) * \
                self.air.get_k(teFcFilmDegC) / self.vehthrm.fcDiam
            self.hFcToAmb[i] = np.interp(self.teFcDegC[i-1],
                [self.vehthrm.teTStatSTODegC, self.vehthrm.teTStatFODegC],
                [hFcToAmbSphere, hFcToAmbSphere * self.vehthrm.radiator_eff])

        self.fcConvToAmbKw[i] = self.hFcToAmb[i] * 1e-3 * self.vehthrm.fcSurfArea * (self.teFcDegC[i-1] - self.teAmbDegC[i-1])
        
        if self.vehthrm.hvac_model == 'internal':
            self.fcToHtrKw[i] = 0.0 # placeholder

        if self.vehthrm.cabin_model == 'internal':
            # flat plate model for isothermal, mixed-flow from Incropera and deWitt, Fundamentals of Heat and Mass
            # Transfer, 7th Edition
            teCabFilmDegC = 0.5 * (self.teCabDegC[i-1] + self.teAmbDegC[i-1])
            Re_L = self.air.get_rho(teCabFilmDegC) * self.mpsAch[i-1] * self.vehthrm.cabLength / self.air.get_mu(teCabFilmDegC)
            Re_L_crit = 5.0e5 # critical Re for transition to turbulence
            if Re_L < Re_L_crit:
                # equation 7.30
                Nu_L_bar = 0.664 * Re_L ** 0.5 * self.air.get_Pr(teCabFilmDegC) ** (1 / 3)
            else:
                # equation 7.38
                A = 871.0 # equation 7.39
                Nu_L_bar = (0.037 * Re_L ** 0.8 - A) * self.air.get_Pr(teCabFilmDegC)
            
            self.cabConvToAmbKw[i] = 1e-3 * (self.vehthrm.cabLength * self.vehthrm.cabWidth) * Nu_L_bar * \
                self.air.get_k(teFcFilmDegC) / self.vehthrm.cabLength * (self.teCabDegC[i-1] - self.teAmbDegC[i-1]) 
            
            self.teCabDegC[i] = self.teCabDegC[i-1] + \
                (self.fcToHtrKw[i] - self.cabConvToAmbKw[i]) / \
                    self.vehthrm.cabThrmMass * self.cyc.secs[i]
        
        # Energy balance for fuel converter
        self.teFcDegC[i] = self.teFcDegC[i-1] + (
            self.fcHeatGenKw[i] - self.fcConvToAmbKw[i] - self.fcToHtrKw[i]
            ) / self.vehthrm.fcThrmMass * self.cyc.secs[i]
        
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
                    self.vehthrm.fcTempEffOffset + self.vehthrm.fcTempEffSlope * self.teFcDegC[i]
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

hotspec = build_spec(SimDriveHot(Cycle('udds'), 
                    Vehicle(1), 
                    teAmbDegC=np.ones(len(Cycle('udds').cycSecs))))

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

    def __init_objects__(self, cyc, veh):
        """Override of method in super class (SimDriveHot)."""
        self.veh = veh
        self.cyc = cyc.copy()  # this cycle may be manipulated
        self.cyc0 = cyc.copy()  # this cycle is not to be manipulated
        self.sim_params = SimDriveParams()
        self.props = params.PhysicalPropertiesJit()
        self.air = AirPropertiesJit()
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


    
