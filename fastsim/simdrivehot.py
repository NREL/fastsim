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
        self.fc_C_kJ__K = 100.0
        # parameter for ngine characteristic length [m] for heat transfer calcs
        self.fc_L = 1.0
        # parameter for heat transfer coeff [W / (m ** 2 * K)] from eng to ambient during vehicle stop
        self.fc_htc_to_amb_stop = 50.0
        # coeff. for fraction of combustion heat that goes to fuel converter (engine) 
        # thermal mass. Remainder goes to environment (e.g. via tailpipe)
        self.fc_coeff_from_comb = 1e-4 
        # parameter for temperature [ºC] at which thermostat starts to open
        self.tstat_te_sto_degC = 85.0
        # temperature delta [ºC] over which thermostat is partially open
        self.tstat_te_delta_degC = 5.0
        # radiator effectiveness -- ratio of active heat rejection from 
        # radiator to passive heat rejection
        self.rad_eps = 5.0
        
        # temperature-dependent efficiency
        # fuel converter (engine or fuel cell) thermal model 'internal' or 'external' w.r.t. fastsim
        self.fc_model = 'internal'
        # 'linear' or 'exponential'
        # if 'linear', temparature has linear impact on efficiency.  If 'exponential',
        # temperature has exponential impact on temperature
        self.fc_temp_eff_model = 'exponential'  
        # offset for scaling FC efficiency w.r.t. to temperature in linear or exponential model
        self.fc_temp_eff_offset = 0.1
        # slope for scaling FC efficiency w.r.t. to temperature in linear model
        # exponential decay constant for exponential model
        self.fc_temp_eff_slope = 0.01
        # minimum coefficient for scaling FC efficiency as a function of temperature
        self.fc_temp_eff_min = 0.1

        # cabin
        # cabin model 'internal' or 'external' w.r.t. fastsim
        self.cabin_model = 'external'
        # parameter for cabin thermal mass [kJ/K]
        self.cab_C_kJ__K = 5.0
        # cabin length [m], modeled as a flat plate
        self.cab_L_length = 2.5
        # cabin width [m], modeled as a flat plate
        self.cab_L_width = 1.75
        # cabin shell thermal resistance [m **2 * K / W]
        self.cab_r_to_amb = 0.5
        # parameter for heat transfer coeff [W / (m ** 2 * K)] from cabin to ambient during
        # vehicle stop
        self.cab_h_to_amb_stop = 50.0

        # exhaust port
        # 'external' (effectively no model) is default
        # exhaust port (exhport) model 'internal' or 'external' w.r.t. fastsim 
        self.exhport_model = 'external'
        # thermal conductance [W/K] for heat transfer to ambient
        self.exhport_hA_ext = 0.05 
        # thermal conductance [W/K] for heat transfer from exhaust
        self.exhport_hA_int = 0.05 
        # exhaust port thermal capacitance [kJ/K]
        self.exhport_C_kJ__K = 20.0 

        # catalytic converter (catalyst)
        # catalyst model 'internal' or 'external' w.r.t. fastsim 
        # 'external' (effectively no model) is default
        self.cat_model = 'external'
        # diameter [m] of catalyst as sphere for thermal model
        self.cat_L = 0.3
        # catalyst thermal capacitance [kJ/K]
        self.cat_C_kJ__K = 20.0 
        # parameter for heat transfer coeff [W / (m ** 2 * K)] from catalyst to ambient
        # during vehicle stop
        self.cat_h_to_amb_stop = 50.0

        # model choices
        # HVAC model 'internal' or 'external' w.r.t. fastsim
        self.hvac_model = 'external'

    def get_numba_vehthrm(self):
        """Load numba JIT-compiled vehicle."""
        from fastsim.buildspec import build_spec
        numba_vehthrm = VehicleThermalJit()
        for item in build_spec(VehicleThermal()):
            numba_vehthrm.__setattr__(item[0], self.__getattribute__(item[0]))
            
        return numba_vehthrm

    # derived temperature [ºC] at which thermostat is fully open
    @property
    def tstat_te_fo_degC(self):
        return self.tstat_te_sto_degC + self.tstat_te_delta_degC

    # parameter for engine surface area [m**2] for heat transfer calcs
    @property
    def fc_area_ext(self):
        return np.pi * self.fc_L ** 2.0 / 4.0
    # parameter for catalyst surface area [m**2] for heat transfer calcs
    @property
    def cat_area_ext(self):
        return np.pi * self.cat_L ** 2.0 / 4.0

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
    vehthrm_copy = VehicleThermalJit() if use_jit else VehicleThermal() 
    for item in build_spec(VehicleThermal()):
        vehthrm_copy.__setattr__(item[0], vehthrm.__getattribute__(item[0]))
        
    return vehthrm_copy

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

    def __init__(self, cyc, veh, amb_te_degC, fc_te_init_degC=90.0, cab_te_init_degC=22.0, exhport_te_init_degC=22.0, cat_te_init_degC=22.0):
        """Initialize time array variables that are not used in base SimDrive.
        Arguments:
        ----------
        amb_te_degC: array of ambient temperatures [C].  Must be declared with 
            dtype=np.float64, e.g. np.zeros(len(cyc.cycSecs, dtype=np.float64)).
        fc_te_init_degC: (optional) fuel converter initial temperature [C]
        cab_te_init_degC: (optional) cabin initial temperature [C]"""
        self.__init_objects__(cyc, veh)
        
        # for persistence through iteration        
        self.fc_te_init_degC = fc_te_init_degC 
        self.cab_te_init_degC = cab_te_init_degC 
        self.exhport_te_init_degC = exhport_te_init_degC 
        self.cat_te_init_degC = cat_te_init_degC 
        
        self.hev_sim_count = 0
        self.init_arrays()
        self.init_thermal_arrays(amb_te_degC)

    def __init_objects__(self, cyc: cycle.Cycle, veh: vehicle.Vehicle):
        self.veh = veh
        self.cyc = cyc.copy()  # this cycle may be manipulated
        self.cyc0 = cyc.copy()  # this cycle is not to be manipulated
        self.sim_params = simdrive.SimDriveParamsClassic()
        self.props = params.PhysicalProperties()
        self.air = AirProperties()
        self.conv_calcs = ConvectionCalcs()
        self.vehthrm = VehicleThermal()

    def init_thermal_arrays(self, amb_te_degC):
        """Arguments:
        amb_te_degC : Float, ambient temperature array for cycle"""
        len_cyc = len(self.cyc.cycSecs)
        # fuel converter (engine) variables
        # fuel converter (engine) temperature [°C]
        self.fc_te_degC = np.zeros(len_cyc, dtype=np.float64) 
        # fuel converter temperature efficiency correction
        self.fc_L = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter heat generation [kW]
        self.fc_qdot_kW = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter convection to ambient [kW]
        self.fc_qdot_to_amb_kW = np.zeros(len_cyc, dtype=np.float64)
        # fuel converter heat loss to heater core [kW]
        self.fc_clnt_to_htr_kW = np.zeros(len_cyc, dtype=np.float64)
        # heat transfer coeff [W / (m ** 2 * K)] to amb after arbitration
        self.fc_htc_to_amb = np.zeros(len_cyc, dtype=np.float64)
        # lambda (air/fuel ratio normalized w.r.t. stoich air/fuel ratio) -- 1 is reasonable default
        self.fc_lambda = np.ones(len_cyc, dtype=np.float64)
        # lambda-dependent adiabatic flame temperature
        self.fc_te_adiabatic_degC = np.zeros(len_cyc, dtype=np.float64)

        # cabin (cab) variables
        # cabin temperature [°C]
        self.cab_te_degC = np.zeros(len_cyc, dtype=np.float64)
        # cabin solar load [kW]
        self.cab_qdot_solar_kW = np.zeros(len_cyc, dtype=np.float64)
        # cabin convection to ambient [kW]
        self.cab_qdot_to_amb_kW = np.zeros(len_cyc, dtype=np.float64) 

        # exhaust mass flow rate
        self.exh_mdot = np.zeros(len_cyc, dtype=np.float64)

        # exhaust port (exhport) variables
        # exhaust temperature at exhaust port inlet 
        self.exhport_exh_te_in_degC = np.append(
            self.exhport_te_init_degC, np.zeros(len_cyc - 1, dtype=np.float64)
        )
        # heat transfer from exhport to ambient [kW]
        self.exhport_qdot_to_amb = np.zeros(len_cyc, dtype=np.float64)
        # catalyst temperature [°C]
        self.exhport_te_degC = np.append(
            self.exhport_te_init_degC, np.zeros(len_cyc - 1, dtype=np.float64)
        )
        # convection from exhaust to exhport [W] 
        # positive means exhport is receiving heat
        self.exhport_qdot_from_exh = np.zeros(len_cyc)
        # net heat generation in cat [W]
        self.exhport_qdot_net = np.zeros(len_cyc)

        # catalyst (cat) variables
        # catalyst heat generation [W]
        self.cat_qdot = np.zeros(len_cyc, dtype=np.float64)
        # catalytic converter convection coefficient to ambient [W / (m ** 2 * K)]
        self.cat_htc_to_amb = np.zeros(len_cyc, dtype=np.float64)
        # heat transfer from catalyst to ambient [W]
        self.cat_qdot_to_amb = np.zeros(len_cyc, dtype=np.float64)
        # catalyst temperature [°C]
        self.cat_te_degC = np.append(
            self.cat_te_init_degC, np.zeros(len_cyc - 1, dtype=np.float64)
        )
        # catalyst external reynolds number
        self.cat_Re_ext = np.zeros(len_cyc)
        # convection from exhaust to cat [W] 
        # positive means cat is receiving heat
        self.cat_qdot_from_exh = np.zeros(len_cyc)
        # net heat generation in cat [W]
        self.cat_qdot_net = np.zeros(len_cyc)

        
        # this block ~should~ prevent the __init__ call in sim_drive_walk from 
        # overriding the prescribed ambient temperature
        self.fc_te_degC[0] = self.fc_te_init_degC
        self.cab_te_degC[0] = self.cab_te_init_degC

        self.amb_te_degC = amb_te_degC

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
        self.init_thermal_arrays(self.amb_te_degC)
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

        # verify that valid option is specified
        assert self.vehthrm.fc_model in ['external', 'internal'], "Invalid option."

        if self.vehthrm.fc_model == 'internal':
            self.set_fc_thermal_calcs(i)

        # verify that valid option is specified
        assert self.vehthrm.hvac_model in ['external', 'internal'], "Invalid option: " + self.vehthrm.hvac_model

        if self.vehthrm.hvac_model == 'internal':
            self.fc_clnt_to_htr_kW[i] = 0.0 # placeholder

        # verify that valid option is specified
        assert self.vehthrm.cabin_model in ['external', 'internal'], "Invalid option: " + self.vehthrm.cabin_model

        if self.vehthrm.cabin_model == 'internal':
            self.set_cab_thermal_calcs(i)

        # verify that valid option is specified
        assert self.vehthrm.exhport_model in ['external', 'internal'], "Invalid option: " + self.vehthrm.exhport_model 

        if self.vehthrm.exhport_model == 'internal':
            self.set_exhport_thermal_calcs(i)

        # verify that valid option is specified
        assert self.vehthrm.cat_model in ['external', 'internal'], "Invalid option: " + self.vehthrm.cat_model

        if self.vehthrm.cat_model == 'internal':
            self.set_cat_thermal_calcs(i)

        if self.vehthrm.fc_model == 'internal':
            # Energy balance for fuel converter
            self.fc_te_degC[i] = self.fc_te_degC[i-1] + (
               self.fc_qdot_kW[i] - self.fc_qdot_to_amb_kW[i] - self.fc_clnt_to_htr_kW[i]
            ) / self.vehthrm.fc_C_kJ__K * self.cyc.dt_s[i]
       
    def set_fc_thermal_calcs(self, i):
        """
        Solve fuel converter thermal behavior assuming convection parameters of sphere.
        """
        # Constitutive equations for fuel converter
        # calculation of adiabatic flame temperature
        self.fc_te_adiabatic_degC[i] = self.air.get_T_from_h(
            ((1 + self.fc_lambda[i] * self.props.fuel_afr_stoich) * self.air.get_h(self.amb_te_degC[i]) + 
                self.props.fuel_lhv_kJ__kg * 1_000 * min(1, self.fc_lambda[i])
            ) / (1 + self.fc_lambda[i] * self.props.fuel_afr_stoich)
        )

        # heat generation 
        self.fc_qdot_kW[i] = self.vehthrm.fc_coeff_from_comb * (
            self.fc_te_adiabatic_degC[i] - self.fc_te_degC[i-1]) * (self.fcKwInAch[i-1] - self.fcKwOutAch[i-1])
    
        # density * speed * diameter / dynamic viscosity
        fc_air_film_Re = self.air.get_rho(fc_air_film_te_degC) * self.mpsAch[i-1] * self.vehthrm.fc_L / \
            self.air.get_mu(fc_air_film_te_degC) 

        # calculate heat transfer coeff. from engine to ambient [W / (m ** 2 * K)]
        if self.mpsAch[i-1] < 1:
            # if stopped, scale based on thermostat opening and constant convection
            self.fc_htc_to_amb[i] = np.interp(self.fc_te_degC[i-1], 
                [self.vehthrm.tstat_te_sto_degC, self.vehthrm.tstat_te_fo_degC],
                [self.vehthrm.fc_htc_to_amb_stop, self.vehthrm.fc_htc_to_amb_stop * self.vehthrm.rad_eps])
        else:
            # Calculate heat transfer coefficient for sphere, 
            # from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44

            fc_sphere_conv_params = self.conv_calcs.get_sphere_conv_params(fc_air_film_Re)
            fc_htc_to_ambSphere = (fc_sphere_conv_params[0] * fc_air_film_Re ** fc_sphere_conv_params[1]) * \
                self.air.get_Pr(fc_air_film_te_degC) ** (1 / 3) * \
                self.air.get_k(fc_air_film_te_degC) / self.vehthrm.fc_L
            self.fc_htc_to_amb[i] = np.interp(self.fc_te_degC[i-1],
                [self.vehthrm.tstat_te_sto_degC, self.vehthrm.tstat_te_fo_degC],
                [fc_htc_to_ambSphere, fc_htc_to_ambSphere * self.vehthrm.rad_eps])

        self.fc_qdot_to_amb_kW[i] = self.fc_htc_to_amb[i] * 1e-3 * self.vehthrm.fc_area_ext * (self.fc_te_degC[i-1] - self.amb_te_degC[i-1])

    def set_cab_thermal_calcs(self, i):
        """
        Solve cabin thermal behavior.
        """

        # if self.vehthrm.cabin_model == 'flat plate':
        # flat plate model for isothermal, mixed-flow from Incropera and deWitt, Fundamentals of Heat and Mass
        # Transfer, 7th Edition
        teCabFilmDegC = 0.5 * (self.cab_te_degC[i-1] + self.amb_te_degC[i-1])
        Re_L = self.air.get_rho(teCabFilmDegC) * self.mpsAch[i-1] * self.vehthrm.cab_L_length / self.air.get_mu(teCabFilmDegC)
        Re_L_crit = 5.0e5 # critical Re for transition to turbulence
        if Re_L < Re_L_crit:
            # equation 7.30
            Nu_L_bar = 0.664 * Re_L ** 0.5 * self.air.get_Pr(teCabFilmDegC) ** (1 / 3)
        else:
            # equation 7.38
            A = 871.0 # equation 7.39
            Nu_L_bar = (0.037 * Re_L ** 0.8 - A) * self.air.get_Pr(teCabFilmDegC)
        
        if self.mphAch[i-1] > 2.0:                
            self.cab_qdot_to_amb_kW[i] = 1e-3 * (self.vehthrm.cab_L_length * self.vehthrm.cab_L_width) / (
                1 / (Nu_L_bar * self.air.get_k(fc_air_film_te_degC) / self.vehthrm.cab_L_length) + self.vehthrm.cab_r_to_amb
                ) * (self.cab_te_degC[i-1] - self.amb_te_degC[i-1]) 
        else:
            self.cab_qdot_to_amb_kW[i] = 1e-3 * (self.vehthrm.cab_L_length * self.vehthrm.cab_L_width) / (
                1 / self.vehthrm.cab_h_to_amb_stop + self.vehthrm.cab_r_to_amb
                ) * (self.cab_te_degC[i-1] - self.amb_te_degC[i-1])
        
        self.cab_te_degC[i] = self.cab_te_degC[i-1] + \
            (self.fc_clnt_to_htr_kW[i] - self.cab_qdot_to_amb_kW[i]) / \
                self.vehthrm.cab_C_kJ__K * self.cyc.dt_s[i]

    def set_exhport_thermal_calcs(self, i):
        """
        Solve exhport thermal behavior.
        """
        self.exh_mdot[i] = self.fsKwOutAch[i] / self.props.fuel_lhv_kJ__kg * (1 + self.props.fuel_afr_stoich)

        if self.exh_mdot[i] > 5e-4:
            self.exhport_exh_te_in_degC[i] = self.air.get_T_from_h(
                (self.fsKwOutAch[i] * (1 - self.vehthrm.comb_h_to_thrml_mass_frac) - self.fcKwOutAch[i]) / self.exh_mdot[i] * 1e3
            )
        else:
            # when flow is small, assume inlet temperature is temporally constant
            self.exhport_exh_te_in_degC[i] = self.exhport_exh_te_in_degC[i-1]

        # calculate heat transfer coeff. from exhaust port to ambient [W / (m ** 2 * K)]

        if (self.exhport_te_degC[i-1] - self.amb_te_degC[i-1]) > 0:
            # if exhaust port is hotter than ambient, make sure heat transfer cannot violate the first law
            self.exhport_qdot_to_amb[i] = min(
                # nominal heat transfer to ambient
                self.vehthrm.exhport_hA_ext * (self.exhport_te_degC[i-1] - self.amb_te_degC[i-1]), 
                # max possible heat transfer to ambient
                self.vehthrm.exhport_C_kJ__K * (self.exhport_te_degC[i-1] - self.amb_te_degC[i-1]) / self.cyc.dt_s[i] 
            )
        else:
            self.exhport_qdot_to_amb[i] = max(
                # nominal heat transfer to ambient
                self.vehthrm.exhport_hA_ext * (self.exhport_te_degC[i-1] - self.amb_te_degC[i-1]), 
                # max possible heat transfer to ambient
                self.vehthrm.exhport_C_kJ__K * (self.exhport_te_degC[i-1] - self.amb_te_degC[i-1]) / self.cyc.dt_s[i] 
            )                

        if (self.exhport_exh_te_in_degC[i-1] - self.exhport_te_degC[i-1]) > 0:
            self.exhport_qdot_from_exh[i] = min(
                self.exh_mdot[i-1] * (self.air.get_h(self.exhport_exh_te_in_degC[i-1]) - self.air.get_h(self.exhport_te_degC[i-1])),
                self.vehthrm.exhport_C_kJ__K * (self.exhport_exh_te_in_degC[i-1] - self.exhport_te_degC[i-1]) / self.cyc.dt_s[i]
            )
        else:
            self.exhport_qdot_from_exh[i] = max(
                self.exh_mdot[i-1] * (self.air.get_h(self.exhport_exh_te_in_degC[i-1]) - self.air.get_h(self.exhport_te_degC[i-1])),
                self.vehthrm.exhport_C_kJ__K * (self.exhport_exh_te_in_degC[i-1] - self.exhport_te_degC[i-1]) / self.cyc.dt_s[i]
            )

        self.exhport_qdot_net[i] = self.exhport_qdot_from_exh[i] - self.exhport_qdot_to_amb[i]
        self.exhport_te_degC[i] = (
            self.exhport_te_degC[i-1] + self.exhport_qdot_net * 1e-3 / self.vehthrm.exhport_C_kJ__K * self.cyc.dt_s[i]
        )

    def set_cat_thermal_calcs(self, i):
        """
        Solve catalyst thermal behavior.
        """
        # external or internal model handling catalyst thermal behavior

        # Constitutive equations for catalyst
        # catalyst film temperature for property calculation
        cat_te_ext_film_degC = 0.5 * (self.cat_te_degC[i-1] + self.amb_te_degC[i-1])
        # density * speed * diameter / dynamic viscosity
        self.cat_Re_ext[i] = (
            self.air.get_rho(cat_te_ext_film_degC) * self.mpsAch[i-1] * self.vehthrm.cat_L 
            / self.air.get_mu(cat_te_ext_film_degC) 
        )

        # calculate heat transfer coeff. from cat to ambient [W / (m ** 2 * K)]
        if self.mpsAch[i-1] < 1:
            # if stopped, scale based on constant convection
            self.cat_htc_to_amb[i] = self.vehthrm.cat_h_to_amb_stop
        else:
            # if moving, scale based on speed dependent convection and thermostat opening
            # Nusselt number coefficients from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            cat_sphere_conv_params = self.conv_calcs.get_sphere_conv_params(self.cat_Re_ext)
            cat_htc_to_ambSphere = (cat_sphere_conv_params[0] * self.cat_Re_ext ** cat_sphere_conv_params[1]) * \
                self.air.get_Pr(fc_air_film_te_degC) ** (1 / 3) * \
                self.air.get_k(fc_air_film_te_degC) / self.vehthrm.cat_L
            self.fc_htc_to_amb[i] = cat_htc_to_ambSphere

        if (self.cat_te_degC[i-1] - self.amb_te_degC[i-1]) > 0:
            self.cat_qdot_to_amb_kW[i] = min(
                # nominal heat transfer to ambient
                self.cat_htc_to_amb[i] * self.vehthrm.cat_area_ext * (self.cat_te_degC[i-1] - self.amb_te_degC[i-1]), 
                # max possible heat transfer to ambient
                self.vehthrm.cat_C_kJ__K * (self.cat_te_degC[i-1] - self.amb_te_degC[i-1]) / self.cyc.dt_s[i] 
            )
        else:
            self.cat_qdot_to_amb_kW[i] = max(
                # nominal heat transfer to ambient
                self.cat_htc_to_amb[i] * self.vehthrm.cat_area_ext * (self.cat_te_degC[i-1] - self.amb_te_degC[i-1]), 
                # max possible heat transfer to ambient
                self.vehthrm.cat_C_kJ__K * (self.cat_te_degC[i-1] - self.amb_te_degC[i-1]) / self.cyc.dt_s[i] 
            )                

        if (self.te_exh_in[i-1] - self.te_comp[i-1]) > 0:
            self.qd_exh_to_comp[i] = min(
                self.md_exh[i-1] * (self.air.get_h(self.te_exh_in[i-1]) - self.air.get_h(self.te_comp[i-1])),
                self.thrml_mass * (self.te_exh_in[i-1] - self.te_comp[i-1]) / self.dtime_s[i]
            )
        else:
            self.qd_exh_to_comp[i] = max(
                self.md_exh[i-1] * (self.air.get_h(self.te_exh_in[i-1]) - self.air.get_h(self.te_comp[i-1])),
                self.thrml_mass * (self.te_exh_in[i-1] - self.te_comp[i-1]) / self.dtime_s[i]
            )

        # catalyst heat generation
        self.cat_qdot[i] = 0.0 # TODO: put something substantive here eventually

        # net heat generetion/transfer in cat
        self.cat_qdot_net[i] = self.cat_qdot + self.cat_qdot_from_exh - self.cat_qdot_to_amb

        self.cat_te_degC[i] = self.cat_te_degC[i-1] + self.cat_qdot_net[i] * 1e-3 / self.vehthrm.cat_C_kJ__K * self.cyc.dt_s[i]

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
            # verify that valid option is specified
            assert self.vehthrm.fc_temp_eff_model in ['linear', 'exponential'], "Invalid option: " + self.vehthrm.fc_temp_eff_model

            if self.vehthrm.fc_temp_eff_model == 'linear':
                # scaling for multiplying efficiency to be dependent on temperature.
                self.fc_L[i] = max(self.vehthrm.fc_temp_eff_min,  
                    min(1, 
                        self.vehthrm.fc_temp_eff_offset + self.vehthrm.fc_temp_eff_slope * self.fc_te_degC[i]
                        )
                )
            elif self.vehthrm.fc_temp_eff_model == 'exponential':
                self.fc_L[i] = max(self.vehthrm.fc_temp_eff_min, 
                    1 - np.exp(-max(
                        self.fc_te_degC[i] - self.vehthrm.fc_temp_eff_offset, 0) / self.vehthrm.fc_temp_eff_slope)
                )
                
            if self.fcKwOutAch[i] == self.veh.fcMaxOutkW:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / (self.veh.fcEffArray[-1] * self.fc_L[i])
            else:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / \
                    (self.veh.fcEffArray[max(1, np.argmax(self.veh.fcKwOutArray > min(self.fcKwOutAch[i], self.veh.fcMaxOutkW - 0.001)) - 1)]) \
                        / self.fc_L[i]

        self.fsKwOutAch[i] = self.fcKwInAch[i]

        self.fsKwhOutAch[i] = self.fsKwOutAch[i] * self.cyc.dt_s[i] * (1 / 3600.0)

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

def SimDriveHotJit(cyc, veh, amb_te_degC, fc_te_init_degC=90.0, cab_te_init_degC=22.0):
    """Wrapper for Numba jitclass version of SimDriveHot"""
    from . import simdrivehotjit
    SimDriveHotJit.__doc__ += simdrivehotjit.SimDriveHotJit.__doc__

    return simdrivehotjit.SimDriveHotJit(cyc, veh, amb_te_degC, fc_te_init_degC, cab_te_init_degC)

