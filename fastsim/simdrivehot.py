"""Module containing classes and methods for simulating vehicle drive cycle.
For example usage, see ../README.md"""

### Import necessary python modules
from itertools import cycle
import os
import numpy as np
import pandas as pd
import re
import sys

# local modules
from fastsim import params, utils, simdrive
from fastsim.simdrive import SimDriveClassic
from fastsim import cycle
from fastsim import vehicle

class AirProperties(object):
    """Fluid Properties for calculations.  
    
    Values obtained via:
    >>> from CoolProp.CoolProp import PropsSI
    >>> import numpy as np
    >>> import pandas as pd
    >>> T_degC = np.logspace(1, np.log10(5e3 + 70), 25) - 70
    >>> T = T_degC + 273.15
    >>> prop_dict = {
    >>>     'T [°C]': T_degC,
    >>>     'h [J/kg]': [0] * len(T),
    >>>     'k [W/(m*K)]': [0] * len(T),
    >>>     'rho [kg/m^3]': [0] * len(T),
    >>>     'c_p [J/(kg*K)]': [0] * len(T),
    >>>     'mu [Pa*s]': [0] * len(T),
    >>> }

    >>> for i, _ in enumerate(T_degC):
    >>>     prop_dict['h [J/kg]'][i] = f"{PropsSI('H', 'P', 101325, 'T', T[i], 'Air'):.5g}" # specific enthalpy [J/(kg*K)]
    >>>     prop_dict['k [W/(m*K)]'][i] = f"{PropsSI('L', 'P', 101325, 'T', T[i], 'Air'):.5g}" # thermal conductivity [W/(m*K)]
    >>>     prop_dict['rho [kg/m^3]'][i] = f"{PropsSI('D', 'P', 101325, 'T', T[i], 'Air'):.5g}" # density [kg/m^3]
    >>>     prop_dict['c_p [J/(kg*K)]'][i] = f"{PropsSI('C', 'P', 101325, 'T', T[i], 'Air'):.5g}" # density [kg/m^3]
    >>>     prop_dict['mu [Pa*s]'][i] = f"{PropsSI('V', 'P', 101325, 'T', T[i], 'Air'):.5g}" # viscosity [Pa*s]

    >>> prop_df = pd.DataFrame(data=prop_dict)
    >>> pd.set_option('display.float_format', lambda x: '%.3g' % x)
    >>> prop_df = prop_df.apply(np.float64)

    """
    def __init__(self):
        # array at of temperatures [°C] at which properties are evaluated ()
        self._te_array_degC = np.array([ 
            -60.        ,  -57.03690616,  -53.1958198 ,  -48.21658352,
            -41.7619528 ,  -33.39475442,  -22.54827664,   -8.48788571,
            9.73873099,   33.36606527,   63.99440042,  103.69819869,
            155.16660498,  221.88558305,  308.37402042,  420.48979341,
            565.82652205,  754.22788725,  998.45434496, 1315.04739396,
            1725.44993435, 2257.45859876, 2947.10642291, 3841.10336915,
            5000.        
        ], dtype=np.float64)
        # thermal conductivity of air [W / (m * K)]
        self._k_Array = np.array([ 
            0.019597, 0.019841, 0.020156, 0.020561, 0.021083, 0.021753,
            0.022612, 0.023708, 0.025102, 0.026867, 0.02909 , 0.031875,
            0.035342, 0.039633, 0.044917, 0.051398, 0.059334, 0.069059,
            0.081025, 0.095855, 0.11442 , 0.13797 , 0.16828 , 0.20795 ,
            0.26081 
        ], dtype=np.float64)
        # specific heat of air [J / (kg * K)]
        self._cp_Array = np.array([
            1006.2, 1006.1, 1006. , 1005.9, 1005.7, 1005.6, 1005.5, 1005.6,
            1005.9, 1006.6, 1008.3, 1011.6, 1017.9, 1028.9, 1047. , 1073.4,
            1107.6, 1146.1, 1184.5, 1219.5, 1250.1, 1277.1, 1301.7, 1324.5,
            1347.
        ], dtype=np.float64)
        # specific enthalpy of air [J / kg]
        # w.r.t. 0K reference
        self._h_Array = np.array([
            338940.,  341930.,  345790.,  350800.,  357290.,  365710.,
            376610.,  390750.,  409080.,  432860.,  463710.,  503800.,
            556020.,  624280.,  714030.,  832880.,  991400., 1203800.,
            1488700., 1869600., 2376700., 3049400., 3939100., 5113600.,
            6662000.
        ], dtype=np.float64)
        # Dynamic viscosity of air [Pa * s]
        self._mu_Array = np.array([
            1.4067e-05, 1.4230e-05, 1.4440e-05, 1.4711e-05, 1.5058e-05,
            1.5502e-05, 1.6069e-05, 1.6791e-05, 1.7703e-05, 1.8850e-05,
            2.0283e-05, 2.2058e-05, 2.4240e-05, 2.6899e-05, 3.0112e-05,
            3.3966e-05, 3.8567e-05, 4.4049e-05, 5.0595e-05, 5.8464e-05,
            6.8036e-05, 7.9878e-05, 9.4840e-05, 1.1423e-04, 1.4006e-04
        ], dtype=np.float64)
        # Prandtl number of air
        self._Pr_Array = self._mu_Array * self._cp_Array / self._k_Array

    def get_rho(self, T, h=180):
        """"
        Returns density [kg/m^3] of air 
        Arguments:
        ----------
        T: Float
            temperature [°C] of air 
        h=180: Float
            evelation [m] above sea level 
        """
        return utils.get_rho_air(T, h)

    def get_k(self, T):
        """"
        Returns thermal conductivity [W/(m*K)] of air 
        Arguments:
        ----------
        T: Float
            temperature [°C] of air 
        """
        return np.interp(T, self._te_array_degC, self._k_Array)

    def get_cp(self, T):
        """Returns specific heat [J/(kg*K)] of air 
        Arguments:
        ----------
        T: Float
            temperature [°C] of air 
        """
        return np.interp(T, self._te_array_degC, self._cp_Array)

    def get_h(self, T):
        """"
        Returns specific enthalpy [J/kg] of air  
        Arguments:  
        ----------
        T: Float
            temperature [°C] of air 
        """
        return np.interp(T, self._te_array_degC, self._h_Array)

    def get_Pr(self, T):
        """"
        Returns thermal Prandtl number of air 
        Arguments:
        ----------
        T: Float
            temperature [°C] of air 
        """
        return np.interp(T, self._te_array_degC, self._Pr_Array)

    def get_mu(self, T):
        """"
        Returns dynamic viscosity [Pa*s] of air 
        Arguments:
        ----------
        T: Float
            temperature [°C] of air 
        """
        return np.interp(T, self._te_array_degC, self._mu_Array)

    def get_T_from_h(self, h):
        """"
        Returns temperature [°C] of air 
        Arguments:
        ----------
        h: Float
            specific enthalpy [J/kg] of air 
        """
        return np.interp(h, self._h_Array, self._te_array_degC)


class VehicleThermal(object):
    """Class for containing vehicle thermal (and related) parameters."""
    def __init__(self):
        """Initial default values for vehicle thermal parameters."""
        # fuel converter / engine
        # parameter fuel converter thermal mass [kJ/K]
        self.fcThrmMass = 100.0
        # parameter for ngine characteristic length [m] for heat transfer calcs
        self.fcDiam = 1.0
        # parameter for heat transfer coeff [W / (m ** 2 * K)] from eng to ambient during vehicle stop
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
        # 'linear' or 'exponential'
        # if 'linear', temparature has linear impact on efficiency.  If 'exponential',
        # temperature has exponential impact on temperature
        self.fc_temp_eff_model = 'linear'  
        # offset for scaling FC efficiency w.r.t. to temperature in linear or exponential model
        self.fcTempEffOffset = 0.1
        # slope for scaling FC efficiency w.r.t. to temperature in linear model
        # exponential decay constant for exponential model
        self.fcTempEffSlope = 0.01
        # minimum coefficient for scaling FC efficiency as a function of temperature
        self.fcTempEffMin = 0.1

        # cabin
        # parameter for cabin thermal mass [kJ/K]
        self.cabThrmMass = 5.0
        # cabin length [m], modeled as a flat plate
        self.cabLength = 2.5
        # cabin width [m], modeled as a flat plate
        self.cabWidth = 1.75
        # cabin shell thermal resistance [m **2 * K / W]
        self.RCabToAmb = 0.5
        # parameter for heat transfer coeff [W / (m ** 2 * K)] from cabin to ambient during
        # vehicle stop
        self.hCabToAmbStop = 50.0

        # catalytic converter (catalyst)
        # diameter [m] of catalyst as sphere
        self.catDiam = 0.3
        # catalyst thermal capacitance [kJ/K]
        self.catThrmMass = 20.0 
        # parameter for heat transfer coeff [W / (m ** 2 * K)] from catalyst to ambient
        # during vehicle stop
        self.hCatToAmbStop = 50.0

        # model choices
        # HVAC model 'internal' or 'external' w.r.t. fastsim
        self.hvac_model = 'external'
        # cabin model 'internal' or 'external' w.r.t. fastsim
        self.cabin_model = 'internal'
        # fuel converter (engine or fuel cell) model 'internal' or 'external' w.r.t. fastsim
        self.fc_model = 'internal'
        # catalyst model 'internal' or 'external' w.r.t. fastsim 
        # 'external' (effectively no model) is default
        self.cat_model = 'external'

    def get_numba_vehthrm(self):
        """Load numba JIT-compiled vehicle."""
        from fastsim.buildspec import build_spec
        numba_vehthrm = VehicleThermalJit()
        for item in build_spec(VehicleThermal()):
            numba_vehthrm.__setattr__(item[0], self.__getattribute__(item[0]))
            
        return numba_vehthrm

    # parameter for engine surface area [m**2] for heat transfer calcs
    @property
    def fcSurfArea(self):
        return np.pi * self.fcDiam ** 2.0 / 4.0
    # parameter for catalyst surface area [m**2] for heat transfer calcs
    @property
    def catSurfArea(self):
        return np.pi * self.catDiam ** 2.0 / 4.0

def copy_vehthrm(vehthrm, use_jit=None):
    """
    Return non-jit version of numba JIT-compiled vehicle.
    Arguments:
    vehthrm: simdrivehot.VehicleThermal
    use_jit: (Boolean)
        default -- infer from arg
        True -- use numba
        False -- don't use numba

    """
    from fastsim.buildspec import build_spec
    vehthrm = VehicleThermalJit() if use_jit else VehicleThermal() 
    for item in build_spec(VehicleThermal()):
        vehthrm.__setattr__(item[0], vehthrm.__getattribute__(item[0]))
        
    return vehthrm

class ConvectionCalcs(object):
    "Class with methods for convection calculations"
    def __init__(self):
        self.__dummy = 0.0

    def get_sphere_conv_params(self, Re):
        """
        Given Reynolds number, return C and m to calculate Nusselt number for 
        sphere, from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
        """
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
        self.conv_calcs = ConvectionCalcs()
        self.vehthrm = VehicleThermal()

    def init_thermal_arrays(self, teAmbDegC):
        """Arguments:
        teAmbDegC : Float, ambient temperature array for cycle"""
        len_cyc = len(self.cyc.cycSecs)
        # fuel converter (engine) temperature [°C]
        self.teFcDegC = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter temperature efficiency correction
        self.fcEffAdj = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter heat generation [kW]
        self.fcHeatGenKw = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter convection to ambient [kW]
        self.fcConvToAmbKw = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter heat loss to heater core [kW]
        self.fcToHtrKw = np.zeros(len_cyc, dtype=np.float64)
        # cabin temperature [°C]
        self.teCabDegC = np.zeros(len_cyc, dtype=np.float64)
        # cabin solar load [kW]
        self.cabSolarKw = np.zeros(len_cyc, dtype=np.float64)
        # cabin convection to ambient [kW]
        self.cabConvToAmbKw = np.zeros(len_cyc, dtype=np.float64) 
        # heat transfer coeff [W / (m ** 2 * K)] to amb after arbitration
        self.hFcToAmb = np.zeros(len_cyc, dtype=np.float64)
        # catalyst heat generation [kW]
        self.catHeatGenKw = np.zeros(len_cyc, dtype=np.float64)
        # catalytic converter convection coefficient to ambient [W / (m ** 2 * K)]
        self.hCatToAmb = np.zeros(len_cyc, dtype=np.float64)
        # heat transfer from catalyst to ambient [kW]
        self.catConvToAmbKw = np.zeros(len_cyc, dtype=np.float64)
        # catalyst temperature [°C]
        self.teCatDegC = np.zeros(len_cyc, dtype=np.float64)
        
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

        if self.vehthrm.fc_model != 'external':
            # external or internal model handling fuel converter (engine) thermal behavior
            # only internal model choice is sphere-based model 

            # Constitutive equations for fuel converter
            self.fcHeatGenKw[i] = self.vehthrm.fcCombToThrmlMassFrac * (self.fcKwInAch[i-1] - self.fcKwOutAch[i-1])
            teFcFilmDegC = 0.5 * (self.teFcDegC[i-1] + self.teAmbDegC[i-1])
            # density * speed * diameter / dynamic viscosity
            Re_fc = self.air.get_rho(teFcFilmDegC) * self.mpsAch[i-1] * self.vehthrm.fcDiam / \
                self.air.get_mu(teFcFilmDegC) 

            # calculate heat transfer coeff. from engine to ambient [W / (m ** 2 * K)]
            if self.mpsAch[i-1] < 1:
                # if stopped, scale based on thermostat opening and constant convection
                self.hFcToAmb[i] = np.interp(self.teFcDegC[i-1], 
                    [self.vehthrm.teTStatSTODegC, self.vehthrm.teTStatFODegC],
                    [self.vehthrm.hFcToAmbStop, self.vehthrm.hFcToAmbStop * self.vehthrm.radiator_eff])
            else:
                # Calculate heat transfer coefficient for sphere, 
                # from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44

                fc_sphere_conv_params = self.conv_calcs.get_sphere_conv_params(Re_fc)
                hFcToAmbSphere = (fc_sphere_conv_params[0] * Re_fc ** fc_sphere_conv_params[1]) * \
                    self.air.get_Pr(teFcFilmDegC) ** (1 / 3) * \
                    self.air.get_k(teFcFilmDegC) / self.vehthrm.fcDiam
                self.hFcToAmb[i] = np.interp(self.teFcDegC[i-1],
                    [self.vehthrm.teTStatSTODegC, self.vehthrm.teTStatFODegC],
                    [hFcToAmbSphere, hFcToAmbSphere * self.vehthrm.radiator_eff])

            self.fcConvToAmbKw[i] = self.hFcToAmb[i] * 1e-3 * self.vehthrm.fcSurfArea * (self.teFcDegC[i-1] - self.teAmbDegC[i-1])
        
        if self.vehthrm.hvac_model != 'external':
            self.fcToHtrKw[i] = 0.0 # placeholder

        if self.vehthrm.cabin_model != 'external':
            # if self.vehthrm.cabin_model == 'flat plate':
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
            
            if self.mphAch[i-1] > 2.0:                
                self.cabConvToAmbKw[i] = 1e-3 * (self.vehthrm.cabLength * self.vehthrm.cabWidth) / (
                    1 / (Nu_L_bar * self.air.get_k(teFcFilmDegC) / self.vehthrm.cabLength) + self.vehthrm.RCabToAmb
                    ) * (self.teCabDegC[i-1] - self.teAmbDegC[i-1]) 
            else:
                self.cabConvToAmbKw[i] = 1e-3 * (self.vehthrm.cabLength * self.vehthrm.cabWidth) / (
                    1 / self.vehthrm.hCabToAmbStop + self.vehthrm.RCabToAmb
                    ) * (self.teCabDegC[i-1] - self.teAmbDegC[i-1])
            
            self.teCabDegC[i] = self.teCabDegC[i-1] + \
                (self.fcToHtrKw[i] - self.cabConvToAmbKw[i]) / \
                    self.vehthrm.cabThrmMass * self.cyc.secs[i]
        
        if self.vehthrm.cat_model != 'external':
            # external or internal model handling catalyst thermal behavior
            # only internal model choice is sphere-based model 

            # Constitutive equations for catalyst
            self.catHeatGenKw[i] = 666 # TODO: put something substantive here
            teCatFilmDegC = 0.5 * (self.teCatDegC[i-1] + self.teAmbDegC[i-1])
            # density * speed * diameter / dynamic viscosity
            Re_cat = self.air.get_rho(teCatFilmDegC) * self.mpsAch[i-1] * self.vehthrm.catDiam / \
                self.air.get_mu(teCatFilmDegC) 

            # calculate heat transfer coeff. from cat to ambient [W / (m ** 2 * K)]
            if self.mpsAch[i-1] < 1:
                # if stopped, scale based on constant convection
                self.hCatToAmb[i] = self.vehthrm.hCatToAmbStop
            else:
                # if moving, scale based on speed dependent convection and thermostat opening
                # Nusselt number coefficients from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
                cat_sphere_conv_params = self.conv_calcs.get_sphere_conv_params(Re_cat)
                hCatToAmbSphere = (cat_sphere_conv_params[0] * Re_cat ** cat_sphere_conv_params[1]) * \
                    self.air.get_Pr(teFcFilmDegC) ** (1 / 3) * \
                    self.air.get_k(teFcFilmDegC) / self.vehthrm.catDiam
                self.hFcToAmb[i] = hCatToAmbSphere

            self.catConvToAmbKw[i] = self.hCatToAmb[i] * 1e-3 * self.vehthrm.catSurfArea * (self.teCatDegC[i-1] - self.teCatDegC[i-1])
            self.teCatDegC[i] = self.teCatDegC[i-1] + (
                self.catHeatGenKw[i] - self.catConvToAmbKw[i]) / self.vehthrm.catThrmMass * self.cyc.secs[i]

        if self.vehthrm.fc_model != 'external':
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
            if self.vehthrm.fc_temp_eff_model == 'linear':
                # scaling for multiplying efficiency to be dependent on temperature.
                self.fcEffAdj[i] = max(self.vehthrm.fcTempEffMin,  
                    min(1, 
                        self.vehthrm.fcTempEffOffset + self.vehthrm.fcTempEffSlope * self.teFcDegC[i]
                        )
                )
            elif self.vehthrm.fc_temp_eff_model == 'exponential':
                self.fcEffAdj[i] = max(self.vehthrm.fcTempEffMin, 
                    1 - np.exp(-max(
                        self.teFcDegC[i] - self.vehthrm.fcTempEffOffset, 0) / self.vehthrm.fcTempEffSlope)
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

# Wrappers to enable apparent jitclasses in this module:

def VehicleThermalJit():
    """Wrapper for Numba jitclass version of VehicleThermal"""
    from . import simdrivehotjit 
    VehicleThermalJit.__doc__ += simdrivehotjit.VehicleThermalJit().__doc__ 

    return simdrivehotjit.VehicleThermalJit()

def ConvectionCalcsJit():
    "Wrapper for Numba JIT version of ConvectionCalcs."
    from . import simdrivehotjit 
    ConvectionCalcs.__doc__ += simdrivehotjit.ConvectionCalcsJit().__doc__

    return simdrivehotjit.ConvectionCalcsJit()

def AirPropertiesJit():
    """Wrapper for Numba jitclass version of FluidProperties"""
    from . import simdrivehotjit
    AirProperties.__doc__ += simdrivehotjit.AirPropertiesJit().__doc__

    return simdrivehotjit.AirPropertiesJit()

def SimDriveHotJit(cyc, veh, teAmbDegC, teFcInitDegC=90.0, teCabInitDegC=22.0):
    """Wrapper for Numba jitclass version of SimDriveHot"""
    from . import simdrivehotjit
    SimDriveHotJit.__doc__ += simdrivehotjit.SimDriveHotJit.__doc__

    return simdrivehotjit.SimDriveHotJit(cyc, veh, teAmbDegC, teFcInitDegC, teCabInitDegC)

def copy_sim_drive_hot(sdhotjit:SimDriveHotJit):
    """
    Given fully solved SimDriveHotJit, returns identical SimDriveHot()
    """
    from . import simdrivehotjit

    # create dummy instance
    sdhot = SimDriveHot(
        cycle.Cycle('udds'), 
        vehicle.Vehicle(1, verbose=False), 
        teAmbDegC=np.ones(len(cycle.Cycle('udds').cycSecs))
    )

    # overwrite
    for keytup in simdrivehotjit._hotspec:
        key = keytup[0]
        if key not in ['cyc', 'veh']:
            sdhot.__setattr__(
                key, 
                sdhotjit.__getattribute__(key)
            )
        elif key == 'cyc':
            # make sure it's not jitted
            sdhot.cyc = cycle.copy_cycle(sdhotjit.cyc, use_jit=False)

        elif key == 'veh':
            # make sure it's not jitted
            sdhot.veh = vehicle.copy_vehicle(sdhotjit.veh, use_jit=False)

        elif key == 'vehthrm':
            # make sure it's not jitted
            sdhot.vehthrm = copy_vehthrm(sdhotjit.vehthrm, use_jit=False)
    
    return sdhot

