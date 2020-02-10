"""Module containing class and methods for simulating vehicle drive cycle.
For example usage, see ../README.md"""

### Import necessary python modules
import numpy as np
import pandas as pd
import re
from Globals import *
from numba import jitclass                 # import the decorator
from numba import float64, int32, bool_, types    # import the types
import warnings
warnings.simplefilter('ignore')

class Cycle(object):
    """Object for containing time, speed, road grade, and road charging vectors 
    for drive cycle."""
    def __init__(self, std_cyc_name=None, cyc_dict=None):
        """Runs other methods, depending on provided keyword argument. Only one keyword
        argument should be provided.  Keyword arguments are identical to 
        arguments required by corresponding methods.  The argument 'std_cyc_name' can be
        optionally passed as a positional argument."""
        super().__init__()
        if std_cyc_name:
            self.set_standard_cycle(std_cyc_name)
        if cyc_dict:
            self.set_from_dict(cyc_dict)
        
    def get_numba_cyc(self):
        """Returns numba jitclass version of Cycle object."""
        numba_cyc = TypedCycle(len(self.cycSecs))
        for key in self.__dict__.keys():
            numba_cyc.__setattr__(key, self.__getattribute__(key).astype(np.float64))
        return numba_cyc

    def set_standard_cycle(self, std_cyc_name):
        """Load time trace of speed, grade, and road type in a pandas dataframe.
        Argument:
        ---------
        std_cyc_name: cycle name string (e.g. 'udds', 'us06', 'hwfet')"""
        csv_path = '..//cycles//' + std_cyc_name + '.csv'
        cyc = pd.read_csv(csv_path)
        for column in cyc.columns:
            self.__setattr__(column, cyc[column].to_numpy())
        self.set_dependents()

    def set_from_dict(self, cyc_dict):
        """Set cycle attributes from dict with keys 'cycGrade', 'cycMps', 'cycSecs', 'cycRoadType'
        and numpy arrays of equal length for values.
        Arguments
        ---------
        cyc_dict: dict containing cycle data
        """

        for key in cyc_dict.keys():
            self.__setattr__(key, cyc_dict[key])
        self.set_dependents()
    
    def set_dependents(self):
        """Sets values dependent on cycle info loaded from file."""
        self.cycMph = self.cycMps * mphPerMps
        self.secs = np.insert(np.diff(self.cycSecs), 0, 0)

# type specifications for attributes of Cycle class
cyc_spec = [('cycSecs', float64[:]),
            ('cycMps', float64[:]),
            ('cycGrade', float64[:]),
            ('cycRoadType', float64[:]),
            ('cycMph', float64[:]),
            ('secs', float64[:])
]

@jitclass(cyc_spec)
class TypedCycle(object):
    """Just-in-time compiled version of Cycle using numba."""
    def __init__(self, len_cyc):
        """This method initialized type numpy arrays as required by 
        numba jitclass."""
        self.cycSecs = np.zeros(len_cyc, dtype=np.float64)
        self.cycMps = np.zeros(len_cyc, dtype=np.float64)
        self.cycGrade = np.zeros(len_cyc, dtype=np.float64)
        self.cycRoadType = np.zeros(len_cyc, dtype=np.float64)
        self.cycMph = np.zeros(len_cyc, dtype=np.float64)
        self.secs = np.zeros(len_cyc, dtype=np.float64)

class Vehicle(object):
    """Class for loading and contaning vehicle attributes
    Optional Argument:
    ---------
    vnum: row number of vehicle to simulate in 'FASTSim_py_veh_db.csv'"""

    def __init__(self, vnum=None):
        super().__init__()
        if vnum:
            self.load_vnum(vnum)

    def get_numba_veh(self):
        if 'numba_veh' not in self.__dict__:
            self.numba_veh = TypedVehicle()
        for item in veh_spec:
            if (type(self.__getattribute__(item[0])) == np.ndarray) | (type(self.__getattribute__(item[0])) == np.float64):
                self.numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.float64))
            elif type(self.__getattribute__(item[0])) == np.int64:
                self.numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.int32))
            else:
                self.numba_veh.__setattr__(
                    item[0], self.__getattribute__(item[0]))
            
        return self.numba_veh
    
    def load_vnum(self, vnum):
        """Load vehicle parameters based on vnum and assign to self.
        Argument:
        ---------
        vnum: row number of vehicle to simulate in 'FASTSim_py_veh_db.csv'"""

        vehdf = pd.read_csv('..//docs//FASTSim_py_veh_db.csv')
        vehdf.set_index('Selection', inplace=True, drop=False)

        def clean_data(raw_data):
            """Cleans up data formatting.
            Argument:
            ------------
            raw_data: cell of vehicle dataframe
            
            Output:
            clean_data: cleaned up data"""
            
            # convert data to string types
            data = str(raw_data)
            # remove percent signs if any are found
            if '%' in data:
                data = data.replace('%', '')
                data = float(data)
                data = data / 100.0
            # replace string for TRUE with Boolean True
            elif re.search('(?i)true', data) != None:
                data = True
            # replace string for FALSE with Boolean False
            elif re.search('(?i)false', data) != None:
                data = False
            else:
                try:
                    data = float(data)
                except:
                    pass
            
            return data
        
        vehdf.loc[vnum].apply(clean_data)

        ### selects specified vnum from vehdf
        for col in vehdf.columns:
            col1 = col.replace(' ', '_')
            
            # assign dataframe columns 
            self.__setattr__(col1, vehdf.loc[vnum, col])
        
        self.set_init_calcs()
        self.set_veh_mass()

    def set_init_calcs(self):
        """Set parameters that can be calculated after loading vehicle data"""
        ### Build roadway power lookup table
        self.MaxRoadwayChgKw = np.zeros(6)
        self.chargingOn = False

        # Checking if a vehicle has any hybrid components
        if self.maxEssKwh == 0 or self.maxEssKw == 0 or self.maxMotorKw == 0:
            self.noElecSys = True

        else:
            self.noElecSys = False

        # Checking if aux loads go through an alternator
        if self.noElecSys == True or self.maxMotorKw <= self.auxKw or self.forceAuxOnFC == True:
            self.noElecAux = True

        else:
            self.noElecAux = False

        # Copying vehPtType to additional key
        self.vehTypeSelection = self.vehPtType
        # to be consistent with Excel version but not used in Python version

        ### Defining Fuel Converter efficiency curve as lookup table for %power_in vs power_out
        ### see "FC Model" tab in FASTSim for Excel

        # Power and efficiency arrays are defined in Globals.py
        
        if self.fcEffType == 1:  # SI engine
            eff = eff_si + self.fcAbsEffImpr

        elif self.fcEffType == 2:  # Atkinson cycle SI engine -- greater expansion
            eff = eff_atk + self.fcAbsEffImpr

        elif self.fcEffType == 3:  # Diesel (compression ignition) engine
            eff = eff_diesel + self.fcAbsEffImpr

        elif self.fcEffType == 4:  # H2 fuel cell
            eff = eff_fuel_cell + self.fcAbsEffImpr

        elif self.fcEffType == 5:  # heavy duty Diesel engine
            eff = eff_hd_diesel + self.fcAbsEffImpr

        # discrete array of possible engine power outputs
        inputKwOutArray = fcPwrOutPerc * self.maxFuelConvKw
        # Relatively continuous array of possible engine power outputs
        fcKwOutArray = self.maxFuelConvKw * fcPercOutArray
        # Initializes relatively continuous array for fcEFF
        fcEffArray = np.zeros(len(fcPercOutArray))

        # the following for loop populates fcEffArray
        for j in range(0, len(fcPercOutArray) - 1):
            low_index = np.argmax(inputKwOutArray >= fcKwOutArray[j])
            fcinterp_x_1 = inputKwOutArray[low_index-1]
            fcinterp_x_2 = inputKwOutArray[low_index]
            fcinterp_y_1 = eff[low_index-1]
            fcinterp_y_2 = eff[low_index]
            fcEffArray[j] = (fcKwOutArray[j] - fcinterp_x_1)/(fcinterp_x_2 -
                                fcinterp_x_1) * (fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        # populate final value
        fcEffArray[-1] = eff[-1]

        # assign corresponding values in veh dict
        self.fcEffArray = fcEffArray
        self.fcKwOutArray = fcKwOutArray
        self.maxFcEffKw = self.fcKwOutArray[np.argmax(fcEffArray)]
        self.fcMaxOutkW = np.max(inputKwOutArray)
            
        ### Defining MC efficiency curve as lookup table for %power_in vs power_out
        ### see "Motor" tab in FASTSim for Excel

        maxMotorKw = self.maxMotorKw
        
        # Power and efficiency arrays are defined in Globals.py

        modern_diff = modern_max - max(large_baseline_eff)

        large_baseline_eff_adj = large_baseline_eff + modern_diff

        mcKwAdjPerc = max(0.0, min((maxMotorKw - 7.5)/(75.0 - 7.5), 1.0))
        mcEffArray = np.zeros(len(mcPwrOutPerc))

        for k in range(0, len(mcPwrOutPerc)):
            mcEffArray[k] = mcKwAdjPerc * large_baseline_eff_adj[k] + \
                (1 - mcKwAdjPerc)*(small_baseline_eff[k])

        mcInputKwOutArray = mcPwrOutPerc * maxMotorKw
        mcFullEffArray = np.zeros(len(mcPercOutArray))
        mcKwOutArray = np.linspace(0, 1, len(mcPercOutArray)) * maxMotorKw

        for m in range(1, len(mcPercOutArray) - 1):
            low_index = np.argmax(mcInputKwOutArray >= mcKwOutArray[m])

            fcinterp_x_1 = mcInputKwOutArray[low_index-1]
            fcinterp_x_2 = mcInputKwOutArray[low_index]
            fcinterp_y_1 = mcEffArray[low_index-1]
            fcinterp_y_2 = mcEffArray[low_index]

            mcFullEffArray[m] = (mcKwOutArray[m] - fcinterp_x_1)/(
                fcinterp_x_2 - fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        mcFullEffArray[0] = 0
        mcFullEffArray[-1] = mcEffArray[-1]

        mcKwInArray = mcKwOutArray / mcFullEffArray
        mcKwInArray[0] = 0

        self.mcKwInArray = mcKwInArray
        self.mcKwOutArray = mcKwOutArray
        self.mcMaxElecInKw = max(mcKwInArray)
        self.mcFullEffArray = mcFullEffArray
        self.mcEffArray = mcEffArray

        if 'stopStart' in self.__dir__() and np.isnan(self.__getattribute__('stopStart')):
            self.stopStart = False

        self.mcMaxElecInKw = max(self.mcKwInArray)

        ### Specify shape of mc regen efficiency curve
        ### see "Regen" tab in FASTSim for Excel
        self.regenA = 500.0  # hardcoded
        self.regenB = 0.99  # hardcoded

    def set_veh_mass(self):
        """Calculate total vehicle mass.  Sum up component masses if 
        positive real number is not specified for self.vehOverrideKg"""
        if not(self.vehOverrideKg > 0):
            if self.maxEssKwh == 0 or self.maxEssKw == 0:
                ess_mass_kg = 0.0
            else:
                ess_mass_kg = ((self.maxEssKwh * self.essKgPerKwh) +
                            self.essBaseKg) * self.compMassMultiplier
            if self.maxMotorKw == 0:
                mc_mass_kg = 0.0
            else:
                mc_mass_kg = (self.mcPeBaseKg+(self.mcPeKgPerKw
                                                * self.maxMotorKw)) * self.compMassMultiplier
            if self.maxFuelConvKw == 0:
                fc_mass_kg = 0.0
            else:
                fc_mass_kg = (((1 / self.fuelConvKwPerKg) * self.maxFuelConvKw +
                            self.fuelConvBaseKg)) * self.compMassMultiplier
            if self.maxFuelStorKw == 0:
                fs_mass_kg = 0.0
            else:
                fs_mass_kg = ((1 / self.fuelStorKwhPerKg) *
                            self.fuelStorKwh) * self.compMassMultiplier
            self.vehKg = self.cargoKg + self.gliderKg + self.transKg * \
                self.compMassMultiplier + ess_mass_kg + \
                mc_mass_kg + fc_mass_kg + fs_mass_kg
        # if positive real number is specified for vehOverrideKg, use that
        else:
            self.vehKg = self.vehOverrideKg

        self.maxTracMps2 = ((((self.wheelCoefOfFric * self.driveAxleWeightFrac * self.vehKg * gravityMPerSec2) /
                              (1+((self.vehCgM * self.wheelCoefOfFric) / self.wheelBaseM))))/(self.vehKg * gravityMPerSec2)) * gravityMPerSec2
        self.maxRegenKwh = 0.5 * self.vehKg * (27**2) / (3600 * 1000)

# type specifications for attributions of Vehcile class
veh_spec = [('Selection', int32),
    ('Scenario_name', types.unicode_type),
    ('vehPtType', int32),
    ('dragCoef', float64),
    ('frontalAreaM2', float64),
    ('gliderKg', float64),
    ('vehCgM', float64),
    ('driveAxleWeightFrac', float64),
    ('wheelBaseM', float64),
    ('cargoKg', float64),
    ('vehOverrideKg', float64),
    ('maxFuelStorKw', float64),
    ('fuelStorSecsToPeakPwr', float64),
    ('fuelStorKwh', float64),
    ('fuelStorKwhPerKg', float64),
    ('maxFuelConvKw', float64),
    ('fcEffType', int32),
    ('fcAbsEffImpr', float64),
    ('fuelConvSecsToPeakPwr', float64),
    ('fuelConvBaseKg', float64),
    ('fuelConvKwPerKg', float64),
    ('maxMotorKw', float64),
    ('motorPeakEff', float64),
    ('motorSecsToPeakPwr', float64),
    ('stopStart', bool_),
    ('mcPeKgPerKw', float64),
    ('mcPeBaseKg', float64),
    ('maxEssKw', float64),
    ('maxEssKwh', float64),
    ('essKgPerKwh', float64),
    ('essBaseKg', float64),
    ('essRoundTripEff', float64),
    ('essLifeCoefA', float64),
    ('essLifeCoefB', float64),
    ('wheelInertiaKgM2', float64),
    ('numWheels', float64),
    ('wheelRrCoef', float64),
    ('wheelRadiusM', float64),
    ('wheelCoefOfFric', float64),
    ('minSoc', float64),
    ('maxSoc', float64),
    ('essDischgToFcMaxEffPerc', float64),
    ('essChgToFcMaxEffPerc', float64),
    ('maxAccelBufferMph', float64),
    ('maxAccelBufferPercOfUseableSoc', float64),
    ('percHighAccBuf', float64),
    ('mphFcOn', float64),
    ('kwDemandFcOn', float64),
    ('altEff', float64),
    ('chgEff', float64),
    ('auxKw', float64),
    ('forceAuxOnFC', bool_),
    ('transKg', float64),
    ('transEff', float64),
    ('compMassMultiplier', float64),
    ('essToFuelOkError', float64),
    ('maxRegen', float64),
    ('valUddsMpgge', float64),
    ('valHwyMpgge', float64),
    ('valCombMpgge', float64),
    ('valUddsKwhPerMile', float64),
    ('valHwyKwhPerMile', float64),
    ('valCombKwhPerMile', float64),
    ('valCdRangeMi', float64),
    ('valConst65MphKwhPerMile', float64),
    ('valConst60MphKwhPerMile', float64),
    ('valConst55MphKwhPerMile', float64),
    ('valConst45MphKwhPerMile', float64),
    ('valUnadjUddsKwhPerMile', float64),
    ('valUnadjHwyKwhPerMile', float64),
    ('val0To60Mph', float64),
    ('valEssLifeMiles', float64),
    ('valRangeMiles', float64),
    ('valVehBaseCost', float64),
    ('valMsrp', float64),
    ('minFcTimeOn', float64),
    ('idleFcKw', float64),
    ('MaxRoadwayChgKw', float64[:]),
    ('chargingOn', bool_),
    ('noElecSys', bool_),
    ('noElecAux', bool_),
    ('vehTypeSelection', int32),
    ('fcEffArray', float64[:]),
    ('fcKwOutArray', float64[:]),
    ('maxFcEffKw', float64),
    ('fcMaxOutkW', float64),
    ('mcKwInArray', float64[:]),
    ('mcKwOutArray', float64[:]),
    ('mcMaxElecInKw', float64),
    ('mcFullEffArray', float64[:]),
    ('mcEffArray', float64[:]),
    ('regenA', float64),
    ('regenB', float64),
    ('vehKg', float64),
    ('maxTracMps2', float64),
    ('maxRegenKwh', float64)
]

@jitclass(veh_spec)
class TypedVehicle(object):
    """Just-in-time compiled version of Vehicle using numba."""
    
    def __init__(self):
        """This method initialized type numpy arrays as required by
        numba jitclass."""
        self.MaxRoadwayChgKw = np.zeros(6, dtype=np.float64)
        self.fcEffArray = np.zeros(100, dtype=np.float64)
        self.fcKwOutArray = np.zeros(100, dtype=np.float64)
        self.mcKwInArray = np.zeros(101, dtype=np.float64)
        self.mcKwOutArray = np.zeros(101, dtype=np.float64)
        self.mcFullEffArray = np.zeros(101, dtype=np.float64)
        self.mcEffArray = np.zeros(11, dtype=np.float64)

class SimDriveCore(object):
    """Class containing methods for running FASTSim iteration.  This class needs to be extended 
    by a class with an init method before being runnable."""
   
    def sim_drive_sub(self, initSoc=None):
        """Receives second-by-second cycle information, vehicle properties, 
        and an initial state of charge and performs a backward facing 
        powertrain simulation. Method 'sim_drive' runs this to 
        iterate through the time steps of 'cyc'.

        Arguments
        ------------
        initSoc: initial battery state-of-charge (SOC) for electrified vehicles"""
        
        ############################
        ###   Loop Through Time  ###
        ############################

        ###  Assign First ValueS  ###
        ### Drive Train
        self.cycMet[0] = 1
        self.curSocTarget[0] = self.veh.maxSoc
        self.essCurKwh[0] = initSoc * self.veh.maxEssKwh
        self.soc[0] = initSoc


        for i in range(1, len(self.cyc.cycSecs)):
            ### Misc calcs
            # If noElecAux, then the HV electrical system is not used to power aux loads 
            # and it must all come from the alternator.  This apparently assumes no belt-driven aux 
            # loads
            # *** 

            self.set_misc_calcs(i)
            self.set_comp_lims(i)
            self.set_power_calcs(i)
            self.set_speed_dist_calcs(i)
            self.set_hybrid_cont_calcs(i)
            self.set_fc_forced_state(i) # can probably be *mostly* done with list comprehension in post processing
            self.set_hybrid_cont_decisions(i)

    def set_misc_calcs(self, i):
        """Sets misc. calculations at time step 'i'
        Arguments
        ------------
        i: index of time step"""

        if self.veh.noElecAux == True:
            self.auxInKw[i] = self.veh.auxKw / self.veh.altEff
        else:
            self.auxInKw[i] = self.veh.auxKw

        # Is SOC below min threshold?
        if self.soc[i-1] < (self.veh.minSoc + self.veh.percHighAccBuf):
            self.reachedBuff[i] = 0
        else:
            self.reachedBuff[i] = 1

        # Does the engine need to be on for low SOC or high acceleration
        if self.soc[i-1] < self.veh.minSoc or (self.highAccFcOnTag[i-1] == 1 and self.reachedBuff[i] == 0):
            self.highAccFcOnTag[i] = 1
        else:
            self.highAccFcOnTag[i] = 0
        self.maxTracMps[i] = self.mpsAch[i-1] + (self.veh.maxTracMps2 * self.cyc.secs[i])

    def set_comp_lims(self, i):
        """Sets component limits for time step 'i'
        Arguments
        ------------
        i: index of time step
        initSoc: initial SOC for electrified vehicles"""

        # max fuel storage power output
        self.curMaxFsKwOut[i] = min(self.veh.maxFuelStorKw, self.fsKwOutAch[i-1] + (
            (self.veh.maxFuelStorKw / self.veh.fuelStorSecsToPeakPwr) * (self.cyc.secs[i])))
        # maximum fuel storage power output rate of change
        self.fcTransLimKw[i] = self.fcKwOutAch[i-1] + \
            ((self.veh.maxFuelConvKw / self.veh.fuelConvSecsToPeakPwr) * (self.cyc.secs[i]))

        self.fcMaxKwIn[i] = min(self.curMaxFsKwOut[i], self.veh.maxFuelStorKw)
        self.fcFsLimKw[i] = self.veh.fcMaxOutkW
        self.curMaxFcKwOut[i] = min(
            self.veh.maxFuelConvKw, self.fcFsLimKw[i], self.fcTransLimKw[i])

        # *** I think self.veh.maxEssKw should also be in the following
        # boolean condition
        if self.veh.maxEssKwh == 0 or self.soc[i-1] < self.veh.minSoc:
            self.essCapLimDischgKw[i] = 0.0

        else:
            self.essCapLimDischgKw[i] = (
                self.veh.maxEssKwh * np.sqrt(self.veh.essRoundTripEff)) * 3600.0 * (self.soc[i-1] - self.veh.minSoc) / (self.cyc.secs[i])
        self.curMaxEssKwOut[i] = min(
            self.veh.maxEssKw, self.essCapLimDischgKw[i])

        if self.veh.maxEssKwh == 0 or self.veh.maxEssKw == 0:
            self.essCapLimChgKw[i] = 0

        else:
            self.essCapLimChgKw[i] = max(((self.veh.maxSoc - self.soc[i-1]) * self.veh.maxEssKwh * (1 /
                                                                                            np.sqrt(self.veh.essRoundTripEff))) / ((self.cyc.secs[i]) * (1 / 3600.0)), 0)

        self.curMaxEssChgKw[i] = min(self.essCapLimChgKw[i], self.veh.maxEssKw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if self.veh.fcEffType == 4:
            self.curMaxElecKw[i] = self.curMaxFcKwOut[i] + self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        else:
            self.curMaxElecKw[i] = self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        self.curMaxAvailElecKw[i] = min(
            self.curMaxElecKw[i], self.veh.mcMaxElecInKw)

        if self.curMaxElecKw[i] > 0:
            # limit power going into e-machine controller to
            if self.curMaxAvailElecKw[i] == max(self.veh.mcKwInArray):
                self.mcElecInLimKw[i] = min(
                    self.veh.mcKwOutArray[len(self.veh.mcKwOutArray) - 1], self.veh.maxMotorKw)
            else:
                self.mcElecInLimKw[i] = min(self.veh.mcKwOutArray[np.argmax(self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) -
                                                                                                0.01, self.curMaxAvailElecKw[i])) - 1], self.veh.maxMotorKw)
        else:
            self.mcElecInLimKw[i] = 0.0

        # Motor transient power limit
        self.mcTransiLimKw[i] = abs(
            self.mcMechKwOutAch[i-1]) + ((self.veh.maxMotorKw / self.veh.motorSecsToPeakPwr) * (self.cyc.secs[i]))

        self.curMaxMcKwOut[i] = max(
            min(self.mcElecInLimKw[i], self.mcTransiLimKw[i], 
            np.float64(0 if self.veh.stopStart else 1) * self.veh.maxMotorKw),
            -self.veh.maxMotorKw)

        if self.curMaxMcKwOut[i] == 0:
            self.curMaxMcElecKwIn[i] = 0
        else:
            if self.curMaxMcKwOut[i] == self.veh.maxMotorKw:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / \
                    self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1]
            else:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / self.veh.mcFullEffArray[max(1, np.argmax(self.veh.mcKwOutArray
                                            > min(self.veh.maxMotorKw - 0.01, self.curMaxMcKwOut[i])) - 1)]

        if self.veh.maxMotorKw == 0:
            self.essLimMcRegenPercKw[i] = 0.0

        else:
            self.essLimMcRegenPercKw[i] = min(
                (self.curMaxEssChgKw[i] + self.auxInKw[i]) / self.veh.maxMotorKw, 1)
        if self.curMaxEssChgKw[i] == 0:
            self.essLimMcRegenKw[i] = 0.0

        else:
            if self.veh.maxMotorKw == self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i]:
                self.essLimMcRegenKw[i] = min(
                    self.veh.maxMotorKw, self.curMaxEssChgKw[i] / self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1])
            else:
                self.essLimMcRegenKw[i] = min(self.veh.maxMotorKw, self.curMaxEssChgKw[i] / self.veh.mcFullEffArray
                                                [max(1, np.argmax(self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i])) - 1)])

        self.curMaxMechMcKwIn[i] = min(
            self.essLimMcRegenKw[i], self.veh.maxMotorKw)
        self.curMaxTracKw[i] = (((self.veh.wheelCoefOfFric * self.veh.driveAxleWeightFrac * self.veh.vehKg * gravityMPerSec2)
                                    / (1 + ((self.veh.vehCgM * self.veh.wheelCoefOfFric) / self.veh.wheelBaseM))) / 1000.0) * (self.maxTracMps[i])

        if self.veh.fcEffType == 4:

            if self.veh.noElecSys == True or self.veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = min(
                    (self.curMaxMcKwOut[i] - self.auxInKw[i]) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 1

            else:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] - min(
                    self.curMaxElecKw[i], 0)) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 2

        else:

            if self.veh.noElecSys == True or self.veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                self.auxInKw[i]) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 3

            else:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                min(self.curMaxElecKw[i], 0)) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 4
        
    def set_power_calcs(self, i):
        """Calculate and set power variables at time step 'i'.
        Arguments
        ------------
        i: index of time step"""

        self.cycDragKw[i] = 0.5 * airDensityKgPerM3 * self.veh.dragCoef * \
            self.veh.frontalAreaM2 * \
            (((self.mpsAch[i-1] + self.cyc.cycMps[i]) / 2.0)**3) / 1000.0
        self.cycAccelKw[i] = (self.veh.vehKg / (2.0 * (self.cyc.secs[i]))) * \
            ((self.cyc.cycMps[i]**2) - (self.mpsAch[i-1]**2)) / 1000.0
        self.cycAscentKw[i] = gravityMPerSec2 * np.sin(np.arctan(
            self.cyc.cycGrade[i])) * self.veh.vehKg * ((self.mpsAch[i-1] + self.cyc.cycMps[i]) / 2.0) / 1000.0
        self.cycTracKwReq[i] = self.cycDragKw[i] + \
            self.cycAccelKw[i] + self.cycAscentKw[i]
        self.spareTracKw[i] = self.curMaxTracKw[i] - self.cycTracKwReq[i]
        self.cycRrKw[i] = gravityMPerSec2 * self.veh.wheelRrCoef * \
            self.veh.vehKg * ((self.mpsAch[i-1] + self.cyc.cycMps[i]) / 2.0) / 1000.0
        self.cycWheelRadPerSec[i] = self.cyc.cycMps[i] / self.veh.wheelRadiusM
        self.cycTireInertiaKw[i] = (((0.5) * self.veh.wheelInertiaKgM2 * (self.veh.numWheels * (self.cycWheelRadPerSec[i]**2.0)) / self.cyc.secs[i]) -
                                    ((0.5) * self.veh.wheelInertiaKgM2 * (self.veh.numWheels * ((self.mpsAch[i-1] / self.veh.wheelRadiusM)**2.0)) / self.cyc.secs[i])) / 1000.0

        self.cycWheelKwReq[i] = self.cycTracKwReq[i] + \
            self.cycRrKw[i] + self.cycTireInertiaKw[i]
        self.regenContrLimKwPerc[i] = self.veh.maxRegen / (1 + self.veh.regenA * np.exp(-self.veh.regenB * (
            (self.cyc.cycMph[i] + self.mpsAch[i-1] * mphPerMps) / 2.0 + 1 - 0)))
        self.cycRegenBrakeKw[i] = max(min(
            self.curMaxMechMcKwIn[i] * self.veh.transEff, self.regenContrLimKwPerc[i] * -self.cycWheelKwReq[i]), 0)
        self.cycFricBrakeKw[i] = - \
            min(self.cycRegenBrakeKw[i] + self.cycWheelKwReq[i], 0)
        self.cycTransKwOutReq[i] = self.cycWheelKwReq[i] + \
            self.cycFricBrakeKw[i]

        if self.cycTransKwOutReq[i] <= self.curMaxTransKwOut[i]:
            self.cycMet[i] = 1
            self.transKwOutAch[i] = self.cycTransKwOutReq[i]

        else:
            self.cycMet[i] = -1
            self.transKwOutAch[i] = self.curMaxTransKwOut[i]
        
    def set_speed_dist_calcs(self, i):
        """Calculate and set variables dependent on speed
        Arguments
        ------------
        i: index of time step"""

        # Cycle is met
        if self.cycMet[i] == 1:
            self.mpsAch[i] = self.cyc.cycMps[i]

        #Cycle is not met
        else:
            Drag3 = (1.0 / 16.0) * airDensityKgPerM3 * \
                self.veh.dragCoef * self.veh.frontalAreaM2
            Accel2 = self.veh.vehKg / (2.0 * (self.cyc.secs[i]))
            Drag2 = (3.0 / 16.0) * airDensityKgPerM3 * \
                self.veh.dragCoef * self.veh.frontalAreaM2 * self.mpsAch[i-1]
            Wheel2 = 0.5 * self.veh.wheelInertiaKgM2 * \
                self.veh.numWheels / (self.cyc.secs[i] * (self.veh.wheelRadiusM**2))
            Drag1 = (3.0 / 16.0) * airDensityKgPerM3 * self.veh.dragCoef * \
                self.veh.frontalAreaM2 * ((self.mpsAch[i-1])**2)
            Roll1 = (gravityMPerSec2 * self.veh.wheelRrCoef * self.veh.vehKg / 2.0)
            Ascent1 = (gravityMPerSec2 *
                        np.sin(np.arctan(self.cyc.cycGrade[i])) * self.veh.vehKg / 2.0)
            Accel0 = - \
                (self.veh.vehKg * ((self.mpsAch[i-1])**2)) / (2.0 * (self.cyc.secs[i]))
            Drag0 = (1.0 / 16.0) * airDensityKgPerM3 * self.veh.dragCoef * \
                self.veh.frontalAreaM2 * ((self.mpsAch[i-1])**3)
            Roll0 = (gravityMPerSec2 * self.veh.wheelRrCoef *
                        self.veh.vehKg * self.mpsAch[i-1] / 2.0)
            Ascent0 = (
                gravityMPerSec2 * np.sin(np.arctan(self.cyc.cycGrade[i])) * self.veh.vehKg * self.mpsAch[i-1] / 2.0)
            Wheel0 = -((0.5 * self.veh.wheelInertiaKgM2 * self.veh.numWheels *
                        (self.mpsAch[i-1]**2)) / (self.cyc.secs[i] * (self.veh.wheelRadiusM**2)))

            Total3 = Drag3 / 1e3
            Total2 = (Accel2 + Drag2 + Wheel2) / 1e3
            Total1 = (Drag1 + Roll1 + Ascent1) / 1e3
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / \
                1e3 - self.curMaxTransKwOut[i]

            Total = np.array([Total3, Total2, Total1, Total0])
            Total_roots = np.roots(Total).astype(np.float64)
            ind = np.int32(np.argmin(np.abs(np.array([self.cyc.cycMps[i] - tot_root for tot_root in Total_roots]))))
            self.mpsAch[i] = Total_roots[ind]

        self.mphAch[i] = self.mpsAch[i] * mphPerMps
        self.distMeters[i] = self.mpsAch[i] * self.cyc.secs[i]
        self.distMiles[i] = self.distMeters[i] * (1.0 / metersPerMile)
        
    def set_hybrid_cont_calcs(self, i):
        """Hybrid control calculations.  
        Arguments
        ------------
        i: index of time step"""

        if self.transKwOutAch[i] > 0:
            self.transKwInAch[i] = self.transKwOutAch[i] / self.veh.transEff
        else:
            self.transKwInAch[i] = self.transKwOutAch[i] * self.veh.transEff

        if self.cycMet[i] == 1:

            if self.veh.fcEffType == 4:
                self.minMcKw2HelpFc[i] = max(
                    self.transKwInAch[i], -self.curMaxMechMcKwIn[i])

            else:
                self.minMcKw2HelpFc[i] = max(
                    self.transKwInAch[i] - self.curMaxFcKwOut[i], -self.curMaxMechMcKwIn[i])
        else:
            self.minMcKw2HelpFc[i] = max(
                self.curMaxMcKwOut[i], -self.curMaxMechMcKwIn[i])

        if self.veh.noElecSys == True:
            self.regenBufferSoc[i] = 0

        elif self.veh.chargingOn:
            self.regenBufferSoc[i] = max(
                self.veh.maxSoc - (self.veh.maxRegenKwh / self.veh.maxEssKwh), (self.veh.maxSoc + self.veh.minSoc) / 2)

        else:
            self.regenBufferSoc[i] = max(((self.veh.maxEssKwh * self.veh.maxSoc) - (0.5 * self.veh.vehKg * (self.cyc.cycMps[i]**2) * (1.0 / 1000)
                                                                            * (1.0 / 3600) * self.veh.motorPeakEff * self.veh.maxRegen)) / self.veh.maxEssKwh, self.veh.minSoc)

            self.essRegenBufferDischgKw[i] = min(self.curMaxEssKwOut[i], max(
                0, (self.soc[i-1] - self.regenBufferSoc[i]) * self.veh.maxEssKwh * 3600 / self.cyc.secs[i]))

            self.maxEssRegenBufferChgKw[i] = min(max(
                0, (self.regenBufferSoc[i] - self.soc[i-1]) * self.veh.maxEssKwh * 3600.0 / self.cyc.secs[i]), self.curMaxEssChgKw[i])

        if self.veh.noElecSys == True:
            self.accelBufferSoc[i] = 0

        else:
            self.accelBufferSoc[i] = min(max((((((((self.veh.maxAccelBufferMph * (1 / mphPerMps))**2)) - ((self.cyc.cycMps[i]**2))) /
                                                (((self.veh.maxAccelBufferMph * (1 / mphPerMps))**2))) * (min(self.veh.maxAccelBufferPercOfUseableSoc * \
                                                                            (self.veh.maxSoc - self.veh.minSoc), self.veh.maxRegenKwh / self.veh.maxEssKwh) * self.veh.maxEssKwh)) / self.veh.maxEssKwh) + \
                self.veh.minSoc, self.veh.minSoc), self.veh.maxSoc)

            self.essAccelBufferChgKw[i] = max(
                0, (self.accelBufferSoc[i] - self.soc[i-1]) * self.veh.maxEssKwh * 3600.0 / self.cyc.secs[i])
            self.maxEssAccelBufferDischgKw[i] = min(max(
                0, (self.soc[i-1] - self.accelBufferSoc[i]) * self.veh.maxEssKwh * 3600 / self.cyc.secs[i]), self.curMaxEssKwOut[i])

        if self.regenBufferSoc[i] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(min(((self.soc[i-1] - (self.regenBufferSoc[i] + self.accelBufferSoc[i]) / 2) * self.veh.maxEssKwh * 3600.0) /
                                                    self.cyc.secs[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        elif self.soc[i-1] > self.regenBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(min(
                self.essRegenBufferDischgKw[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        elif self.soc[i-1] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(
                min(-1.0 * self.essAccelBufferChgKw[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        else:
            self.essAccelRegenDischgKw[i] = max(
                min(0, self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        self.fcKwGapFrEff[i] = abs(self.transKwOutAch[i] - self.veh.maxFcEffKw)

        if self.veh.noElecSys == True:
            self.mcElectInKwForMaxFcEff[i] = 0

        elif self.transKwOutAch[i] < self.veh.maxFcEffKw:

            if self.fcKwGapFrEff[i] == self.veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / \
                    self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1] * -1
            else:
                self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / self.veh.mcFullEffArray[max(
                    1, np.argmax(self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1)] * -1

        else:

            if self.fcKwGapFrEff[i] == self.veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = self.veh.mcKwInArray[len(
                    self.veh.mcKwInArray) - 1]
            else:
                self.mcElectInKwForMaxFcEff[i] = self.veh.mcKwInArray[np.argmax(
                    self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1]

        if self.veh.noElecSys == True:
            self.electKwReq4AE[i] = 0

        elif self.transKwInAch[i] > 0:
            if self.transKwInAch[i] == self.veh.maxMotorKw:

                self.electKwReq4AE[i] = self.transKwInAch[i] / \
                    self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1] + self.auxInKw[i]
            else:
                self.electKwReq4AE[i] = self.transKwInAch[i] / self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.transKwInAch[i])) - 1)] + self.auxInKw[i]

        else:
            self.electKwReq4AE[i] = 0

        self.prevfcTimeOn[i] = self.fcTimeOn[i-1]

        # some conditions in the following if statement have a buffer of 1e-6 to prevent false positives/negatives because these have been encountered in practice.   
        if self.veh.maxFuelConvKw == 0:
            self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and  \
                (self.transKwInAch[i] - 1e-6) <= self.curMaxMcKwOut[i] and \
                (self.electKwReq4AE[i] < self.curMaxElecKw[i] or self.veh.maxFuelConvKw == 0)

        else:
            self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and \
                (self.transKwInAch[i] - 1e-6) <= self.curMaxMcKwOut[i] and \
                (self.electKwReq4AE[i] < self.curMaxElecKw[i] or self.veh.maxFuelConvKw == 0) \
                and ((self.cyc.cycMph[i] - 1e-6) <= self.veh.mphFcOn or self.veh.chargingOn) and \
                self.electKwReq4AE[i] <= self.veh.kwDemandFcOn

        if self.canPowerAllElectrically[i]:

            if self.transKwInAch[i] < self.auxInKw[i]:
                self.desiredEssKwOutForAE[i] = self.auxInKw[i] + \
                    self.transKwInAch[i]

            elif self.regenBufferSoc[i] < self.accelBufferSoc[i]:
                self.desiredEssKwOutForAE[i] = self.essAccelRegenDischgKw[i]

            elif self.soc[i-1] > self.regenBufferSoc[i]:
                self.desiredEssKwOutForAE[i] = self.essRegenBufferDischgKw[i]

            elif self.soc[i-1] < self.accelBufferSoc[i]:
                self.desiredEssKwOutForAE[i] = -self.essAccelBufferChgKw[i]

            else:
                self.desiredEssKwOutForAE[i] = self.transKwInAch[i] + \
                    self.auxInKw[i] - self.curMaxRoadwayChgKw[i]

        else:   
            self.desiredEssKwOutForAE[i] = 0

        if self.canPowerAllElectrically[i]:
            self.essAEKwOut[i] = max(-self.curMaxEssChgKw[i], -self.maxEssRegenBufferChgKw[i], min(0, self.curMaxRoadwayChgKw[i] - (
                self.transKwInAch[i] + self.auxInKw[i])), min(self.curMaxEssKwOut[i], self.desiredEssKwOutForAE[i]))

        else:
            self.essAEKwOut[i] = 0

        self.erAEKwOut[i] = min(max(0, self.transKwInAch[i] + self.auxInKw[i] - self.essAEKwOut[i]), self.curMaxRoadwayChgKw[i])
    
    def set_fc_forced_state(self, i):
        """Calculate control variables related to engine on/off state
        Arguments       
        ------------
        i: index of time step"""

        # force fuel converter on if it was on in the previous time step, but only if fc
        # has not been on longer than minFcTimeOn
        if self.prevfcTimeOn[i] > 0 and self.prevfcTimeOn[i] < self.veh.minFcTimeOn - self.cyc.secs[i]:
            self.fcForcedOn[i] = True
        else:
            self.fcForcedOn[i] = False

        # Engine only mode
        if self.fcForcedOn[i] == False or self.canPowerAllElectrically[i] == False:
            self.fcForcedState[i] = 1
            self.mcMechKw4ForcedFc[i] = 0

        # Engine maximum efficiency mode
        elif self.transKwInAch[i] < 0:
            self.fcForcedState[i] = 2
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i]

        # All electric mode
        elif self.veh.maxFcEffKw == self.transKwInAch[i]:
            self.fcForcedState[i] = 3
            self.mcMechKw4ForcedFc[i] = 0

        # Engine forced on mode
        elif self.veh.idleFcKw > self.transKwInAch[i] and self.cycAccelKw[i] >= 0:
            self.fcForcedState[i] = 4
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - self.veh.idleFcKw

        # Engine + motor mode
        elif self.veh.maxFcEffKw > self.transKwInAch[i]:
            self.fcForcedState[i] = 5
            self.mcMechKw4ForcedFc[i] = 0

        # Fuel cell mode
        else:
            self.fcForcedState[i] = 6
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - \
                self.veh.maxFcEffKw

    def set_hybrid_cont_decisions(self, i):
        """Hybrid control decisions.
        Arguments
        ------------
        i: index of time step"""

        if (-self.mcElectInKwForMaxFcEff[i] - self.curMaxRoadwayChgKw[i]) > 0:
            self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] -
                                            self.curMaxRoadwayChgKw[i]) * self.veh.essDischgToFcMaxEffPerc

        else:
            self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] - \
                                            self.curMaxRoadwayChgKw[i]) * self.veh.essChgToFcMaxEffPerc

        if self.accelBufferSoc[i] > self.regenBufferSoc[i]:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], self.essAccelRegenDischgKw[i]))

        elif self.essRegenBufferDischgKw[i] > 0:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], min(self.essAccelRegenDischgKw[i], self.mcElecInLimKw[i] + self.auxInKw[i], max(self.essRegenBufferDischgKw[i], self.essDesiredKw4FcEff[i]))))

        elif self.essAccelBufferChgKw[i] > 0:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], max(-1 * self.maxEssRegenBufferChgKw[i], min(-self.essAccelBufferChgKw[i], self.essDesiredKw4FcEff[i]))))

        elif self.essDesiredKw4FcEff[i] > 0:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], min(self.essDesiredKw4FcEff[i], self.maxEssAccelBufferDischgKw[i])))

        else:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], max(self.essDesiredKw4FcEff[i], -self.maxEssRegenBufferChgKw[i])))

        self.erKwIfFcIsReq[i] = max(0, min(self.curMaxRoadwayChgKw[i], self.curMaxMechMcKwIn[i],
                                    self.essKwIfFcIsReq[i] - self.mcElecInLimKw[i] + self.auxInKw[i]))

        self.mcElecKwInIfFcIsReq[i] = self.essKwIfFcIsReq[i] + self.erKwIfFcIsReq[i] - self.auxInKw[i]

        if self.veh.noElecSys == True:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] == 0:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] > 0:

            if self.mcElecKwInIfFcIsReq[i] == max(self.veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * \
                    self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i])) - 1)]

        else:
            if self.mcElecKwInIfFcIsReq[i] * -1 == max(self.veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / \
                    self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / (self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i] * -1)) - 1)])

        if self.veh.maxMotorKw == 0:
            self.mcMechKwOutAch[i] = 0

        elif self.fcForcedOn[i] == True and self.canPowerAllElectrically[i] == True and (self.veh.vehPtType == 2.0 or self.veh.vehPtType == 3.0) and self.veh.fcEffType !=4:
            self.mcMechKwOutAch[i] = self.mcMechKw4ForcedFc[i]

        elif self.transKwInAch[i] <= 0:

            if self.veh.fcEffType !=4 and self.veh.maxFuelConvKw > 0:
                if self.canPowerAllElectrically[i] == 1:
                    self.mcMechKwOutAch[i] = - \
                        min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i])
                else:
                    self.mcMechKwOutAch[i] = min(-min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]),
                                                    max(-self.curMaxFcKwOut[i], self.mcKwIfFcIsReq[i]))
            else:
                self.mcMechKwOutAch[i] = min(
                    -min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]), -self.transKwInAch[i])

        elif self.canPowerAllElectrically[i] == 1:
            self.mcMechKwOutAch[i] = self.transKwInAch[i]

        else:
            self.mcMechKwOutAch[i] = max(
                self.minMcKw2HelpFc[i], self.mcKwIfFcIsReq[i])

        if self.mcMechKwOutAch[i] == 0:
            self.mcElecKwInAch[i] = 0.0
            self.motor_index_debug[i] = 0

        elif self.mcMechKwOutAch[i] < 0:

            if self.mcMechKwOutAch[i] * -1 == max(self.veh.mcKwInArray):
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * \
                    self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcMechKwOutAch[i] * -1)) - 1)]

        else:
            if self.veh.maxMotorKw == self.mcMechKwOutAch[i]:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / \
                    self.veh.mcFullEffArray[len(self.veh.mcFullEffArray) - 1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.mcMechKwOutAch[i])) - 1)]

        if self.curMaxRoadwayChgKw[i] == 0:
            self.roadwayChgKwOutAch[i] = 0

        elif self.veh.fcEffType == 4:
            self.roadwayChgKwOutAch[i] = max(
                0, self.mcElecKwInAch[i], self.maxEssRegenBufferChgKw[i], self.essRegenBufferDischgKw[i], self.curMaxRoadwayChgKw[i])

        elif self.canPowerAllElectrically[i] == 1:
            self.roadwayChgKwOutAch[i] = self.erAEKwOut[i]

        else:
            self.roadwayChgKwOutAch[i] = self.erKwIfFcIsReq[i]

        self.minEssKw2HelpFc[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - \
            self.curMaxFcKwOut[i] - self.roadwayChgKwOutAch[i]

        if self.veh.maxEssKw == 0 or self.veh.maxEssKwh == 0:
            self.essKwOutAch[i] = 0

        elif self.veh.fcEffType == 4:

            if self.transKwOutAch[i] >=0:
                self.essKwOutAch[i] = min(max(self.minEssKw2HelpFc[i], self.essDesiredKw4FcEff[i], self.essAccelRegenDischgKw[i]),
                                            self.curMaxEssKwOut[i], self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i])

            else:
                self.essKwOutAch[i] = self.mcElecKwInAch[i] + \
                    self.auxInKw[i] - self.roadwayChgKwOutAch[i]

        elif self.highAccFcOnTag[i] == 1 or self.veh.noElecAux == True:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] - \
                self.roadwayChgKwOutAch[i]

        else:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] + \
                self.auxInKw[i] - self.roadwayChgKwOutAch[i]

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

        if self.fcKwOutAch[i] == 0:
            self.fcKwInAch[i] = 0.0
            self.fcKwOutAch_pct[i] = 0

        if self.veh.maxFuelConvKw == 0:
            self.fcKwOutAch_pct[i] = 0
        else:
            self.fcKwOutAch_pct[i] = self.fcKwOutAch[i] / self.veh.maxFuelConvKw

        if self.fcKwOutAch[i] == 0:
            self.fcKwInAch[i] = 0
        else:
            if self.fcKwOutAch[i] == self.veh.fcMaxOutkW:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / \
                    self.veh.fcEffArray[len(self.veh.fcEffArray) - 1]
            else:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / (self.veh.fcEffArray[max(1, np.argmax(
                    self.veh.fcKwOutArray > min(self.fcKwOutAch[i], self.veh.fcMaxOutkW - 0.001)) - 1)])

        self.fsKwOutAch[i] = self.fcKwInAch[i]

        self.fsKwhOutAch[i] = self.fsKwOutAch[i] * \
            self.cyc.secs[i] * (1 / 3600.0)

        if self.veh.noElecSys == True:
            self.essCurKwh[i] = 0

        elif self.essKwOutAch[i] < 0:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * \
                (self.cyc.secs[i] / 3600.0) * np.sqrt(self.veh.essRoundTripEff)

        else:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * \
                (self.cyc.secs[i] / 3600.0) * (1 / np.sqrt(self.veh.essRoundTripEff))

        if self.veh.maxEssKwh == 0:
            self.soc[i] = 0.0

        else:
            self.soc[i] = self.essCurKwh[i] / self.veh.maxEssKwh

        if self.canPowerAllElectrically[i] == True and self.fcForcedOn[i] == False and self.fcKwOutAch[i] == 0:
            self.fcTimeOn[i] = 0
        else:
            self.fcTimeOn[i] = self.fcTimeOn[i-1] + self.cyc.secs[i]
    
    def set_post_scalars(self):
        """Sets scalar variables that can be calculated after a cycle is run."""

        if self.fsKwhOutAch.sum() == 0:
            self.mpgge = 0

        else:
            self.mpgge = self.distMiles.sum() / \
                (self.fsKwhOutAch.sum() * (1 / kWhPerGGE))

        self.roadwayChgKj = (self.roadwayChgKwOutAch * self.cyc.secs).sum()
        self.essDischgKj = - \
            (self.soc[-1] - self.soc[0]) * self.veh.maxEssKwh * 3600.0
        self.battery_kWh_per_mi  = (
            self.essDischgKj / 3600.0) / self.distMiles.sum()
        self.electric_kWh_per_mi  = (
            (self.roadwayChgKj + self.essDischgKj) / 3600.0) / self.distMiles.sum()
        self.fuelKj = (self.fsKwOutAch * self.cyc.secs).sum()

        if (self.fuelKj + self.roadwayChgKj) == 0:
            self.ess2fuelKwh  = 1.0

        else:
            self.ess2fuelKwh  = self.essDischgKj / (self.fuelKj + self.roadwayChgKj)

        if self.mpgge == 0:
            # hardcoded conversion
            self.Gallons_gas_equivalent_per_mile = self.electric_kWh_per_mi / kWhPerGGE

        else:
            self.Gallons_gas_equivalent_per_mile = 1 / \
                self.mpgge + self.electric_kWh_per_mi  / kWhPerGGE

        self.mpgge_elec = 1 / self.Gallons_gas_equivalent_per_mile

class SimDriveClassic(SimDriveCore):
    """Class containing methods for running FASTSim vehicle 
    fuel economy simulations. This class is not compiled and will 
    run slower for large batch runs."""

    def __init__(self, cyc, veh):
        """Initializes numpy arrays for specific cycle
        Arguments:
        -----------
        cyc: instance of TypedCycle or Cycle class
        veh: instance of TypedVehicle or Vehicle class
        """
        self.veh = veh
        self.cyc = cyc

        len_cyc = len(self.cyc.cycSecs)
        # Component Limits -- calculated dynamically"
        self.curMaxFsKwOut = np.zeros(len_cyc)
        self.fcTransLimKw = np.zeros(len_cyc)
        self.fcFsLimKw = np.zeros(len_cyc)
        self.fcMaxKwIn = np.zeros(len_cyc)
        self.curMaxFcKwOut = np.zeros(len_cyc)
        self.essCapLimDischgKw = np.zeros(len_cyc)
        self.curMaxEssKwOut = np.zeros(len_cyc)
        self.curMaxAvailElecKw = np.zeros(len_cyc)
        self.essCapLimChgKw = np.zeros(len_cyc)
        self.curMaxEssChgKw = np.zeros(len_cyc)
        self.curMaxElecKw = np.zeros(len_cyc)
        self.mcElecInLimKw = np.zeros(len_cyc)
        self.mcTransiLimKw = np.zeros(len_cyc)
        self.curMaxMcKwOut = np.zeros(len_cyc)
        self.essLimMcRegenPercKw = np.zeros(len_cyc)
        self.essLimMcRegenKw = np.zeros(len_cyc)
        self.curMaxMechMcKwIn = np.zeros(len_cyc)
        self.curMaxTransKwOut = np.zeros(len_cyc)

        ### Drive Train
        self.cycDragKw = np.zeros(len_cyc)
        self.cycAccelKw = np.zeros(len_cyc)
        self.cycAscentKw = np.zeros(len_cyc)
        self.cycTracKwReq = np.zeros(len_cyc)
        self.curMaxTracKw = np.zeros(len_cyc)
        self.spareTracKw = np.zeros(len_cyc)
        self.cycRrKw = np.zeros(len_cyc)
        self.cycWheelRadPerSec = np.zeros(len_cyc)
        self.cycTireInertiaKw = np.zeros(len_cyc)
        self.cycWheelKwReq = np.zeros(len_cyc)
        self.regenContrLimKwPerc = np.zeros(len_cyc)
        self.cycRegenBrakeKw = np.zeros(len_cyc)
        self.cycFricBrakeKw = np.zeros(len_cyc)
        self.cycTransKwOutReq = np.zeros(len_cyc)
        self.cycMet = np.zeros(len_cyc)
        self.transKwOutAch = np.zeros(len_cyc)
        self.transKwInAch = np.zeros(len_cyc)
        self.curSocTarget = np.zeros(len_cyc)
        self.minMcKw2HelpFc = np.zeros(len_cyc)
        self.mcMechKwOutAch = np.zeros(len_cyc)
        self.mcElecKwInAch = np.zeros(len_cyc)
        self.auxInKw = np.zeros(len_cyc)
        self.roadwayChgKwOutAch = np.zeros(len_cyc)
        self.minEssKw2HelpFc = np.zeros(len_cyc)
        self.essKwOutAch = np.zeros(len_cyc)
        self.fcKwOutAch = np.zeros(len_cyc)
        self.fcKwOutAch_pct = np.zeros(len_cyc)
        self.fcKwInAch = np.zeros(len_cyc)
        self.fsKwOutAch = np.zeros(len_cyc)
        self.fsKwhOutAch = np.zeros(len_cyc)
        self.essCurKwh = np.zeros(len_cyc)
        self.soc = np.zeros(len_cyc)

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = np.zeros(len_cyc)
        self.essRegenBufferDischgKw = np.zeros(len_cyc)
        self.maxEssRegenBufferChgKw = np.zeros(len_cyc)
        self.essAccelBufferChgKw = np.zeros(len_cyc)
        self.accelBufferSoc = np.zeros(len_cyc)
        self.maxEssAccelBufferDischgKw = np.zeros(len_cyc)
        self.essAccelRegenDischgKw = np.zeros(len_cyc)
        self.mcElectInKwForMaxFcEff = np.zeros(len_cyc)
        self.electKwReq4AE = np.zeros(len_cyc)
        self.canPowerAllElectrically = np.array([False] * len_cyc)
        self.desiredEssKwOutForAE = np.zeros(len_cyc)
        self.essAEKwOut = np.zeros(len_cyc)
        self.erAEKwOut = np.zeros(len_cyc)
        self.essDesiredKw4FcEff = np.zeros(len_cyc)
        self.essKwIfFcIsReq = np.zeros(len_cyc)
        self.curMaxMcElecKwIn = np.zeros(len_cyc)
        self.fcKwGapFrEff = np.zeros(len_cyc)
        self.erKwIfFcIsReq = np.zeros(len_cyc)
        self.mcElecKwInIfFcIsReq = np.zeros(len_cyc)
        self.mcKwIfFcIsReq = np.zeros(len_cyc)
        self.fcForcedOn = np.array([False] * len_cyc)
        self.fcForcedState = np.zeros(len_cyc)
        self.mcMechKw4ForcedFc = np.zeros(len_cyc)
        self.fcTimeOn = np.zeros(len_cyc)
        self.prevfcTimeOn = np.zeros(len_cyc)

        ### Additional Variables
        self.mpsAch = np.zeros(len_cyc)
        self.mphAch = np.zeros(len_cyc)
        self.distMeters = np.zeros(len_cyc)
        self.distMiles = np.zeros(len_cyc)
        self.highAccFcOnTag = np.zeros(len_cyc)
        self.reachedBuff = np.zeros(len_cyc)
        self.maxTracMps = np.zeros(len_cyc)
        self.addKwh = np.zeros(len_cyc)
        self.dodCycs = np.zeros(len_cyc)
        self.essPercDeadArray = np.zeros(len_cyc)
        self.dragKw = np.zeros(len_cyc)
        self.essLossKw = np.zeros(len_cyc)
        self.accelKw = np.zeros(len_cyc)
        self.ascentKw = np.zeros(len_cyc)
        self.rrKw = np.zeros(len_cyc)
        self.motor_index_debug = np.zeros(len_cyc)
        self.debug_flag = np.zeros(len_cyc)
        self.curMaxRoadwayChgKw = np.zeros(len_cyc)

    def sim_drive(self, initSoc=None):
        """Initialize and run sim_drive_sub as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        initSoc: (optional) initial SOC for electrified vehicles.  
            Must be between 0 and 1."""

        if initSoc != None:
            if initSoc > 1.0 or initSoc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = None

        if self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0

            self.sim_drive_sub(initSoc)

        elif self.veh.vehPtType == 2 and initSoc == None:  # HEV

            #####################################
            ### Charge Balancing Vehicle SOC ###
            #####################################

            # Charge balancing SOC for PHEV vehicle types. Iterating initsoc and comparing to final SOC.
            # Iterating until tolerance met or 30 attempts made.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            ess2fuelKwh = 1.0
            sim_count = 0
            while ess2fuelKwh > self.veh.essToFuelOkError and sim_count < 30:
                sim_count += 1
                self.sim_drive_sub(initSoc)
                fuelKj = np.sum(self.fsKwOutAch * self.cyc.secs)
                roadwayChgKj = np.sum(self.roadwayChgKwOutAch * self.cyc.secs)
                ess2fuelKwh = np.abs((self.soc[0] - self.soc[-1]) *
                                     self.veh.maxEssKwh * 3600 / (fuelKj + roadwayChgKj))
                initSoc = min(1.0, max(0.0, self.soc[-1]))

            self.sim_drive_sub(initSoc)

        elif (self.veh.vehPtType == 3 and initSoc == None) or (self.veh.vehPtType == 4 and initSoc == None):  # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = self.veh.maxSoc

            self.sim_drive_sub(initSoc)

        else:

            self.sim_drive_sub(initSoc)

# list of array attributes in SimDrive class for generating list of type specification tuples
attr_list = ['curMaxFsKwOut', 'fcTransLimKw', 'fcFsLimKw', 'fcMaxKwIn', 'curMaxFcKwOut', 'essCapLimDischgKw', 'curMaxEssKwOut', 
            'curMaxAvailElecKw', 'essCapLimChgKw', 'curMaxEssChgKw', 'curMaxElecKw', 'mcElecInLimKw', 'mcTransiLimKw', 'curMaxMcKwOut', 
            'essLimMcRegenPercKw', 'essLimMcRegenKw', 'curMaxMechMcKwIn', 'curMaxTransKwOut', 'cycDragKw', 'cycAccelKw', 'cycAscentKw', 
            'cycTracKwReq', 'curMaxTracKw', 'spareTracKw', 'cycRrKw', 'cycWheelRadPerSec', 'cycTireInertiaKw', 'cycWheelKwReq', 
            'regenContrLimKwPerc', 'cycRegenBrakeKw', 'cycFricBrakeKw', 'cycTransKwOutReq', 'cycMet', 'transKwOutAch', 'transKwInAch', 
            'curSocTarget', 'minMcKw2HelpFc', 'mcMechKwOutAch', 'mcElecKwInAch', 'auxInKw', 'roadwayChgKwOutAch', 'minEssKw2HelpFc', 
            'essKwOutAch', 'fcKwOutAch', 'fcKwOutAch_pct', 'fcKwInAch', 'fsKwOutAch', 'fsKwhOutAch', 'essCurKwh', 'soc', 
            'regenBufferSoc', 'essRegenBufferDischgKw', 'maxEssRegenBufferChgKw', 'essAccelBufferChgKw', 'accelBufferSoc', 
            'maxEssAccelBufferDischgKw', 'essAccelRegenDischgKw', 'mcElectInKwForMaxFcEff', 'electKwReq4AE', 'desiredEssKwOutForAE', 
            'essAEKwOut', 'erAEKwOut', 'essDesiredKw4FcEff', 'essKwIfFcIsReq', 'curMaxMcElecKwIn', 'fcKwGapFrEff', 'erKwIfFcIsReq', 
            'mcElecKwInIfFcIsReq', 'mcKwIfFcIsReq', 'mcMechKw4ForcedFc', 'fcTimeOn', 'prevfcTimeOn', 'mpsAch', 'mphAch', 'distMeters',
            'distMiles', 'highAccFcOnTag', 'reachedBuff', 'maxTracMps', 'addKwh', 'dodCycs', 'essPercDeadArray', 'dragKw', 'essLossKw',
            'accelKw', 'ascentKw', 'rrKw', 'motor_index_debug', 'debug_flag', 'curMaxRoadwayChgKw']

# create types for instances of TypedVehicle and TypedCycle
veh_type = TypedVehicle.class_type.instance_type
cyc_type = TypedCycle.class_type.instance_type

spec = [(attr, float64[:]) for attr in attr_list]
spec.extend([('fcForcedOn', bool_[:]),
             ('fcForcedState', int32[:]),
             ('canPowerAllElectrically', bool_[:]),
             ('mpgge', float64),
             ('roadwayChgKj', float64),
             ('essDischgKj', float64),
             ('battery_kWh_per_mi', float64),
             ('electric_kWh_per_mi', float64),
             ('fuelKj', float64),
             ('ess2fuelKwh', float64),
             ('Gallons_gas_equivalent_per_mile', float64),
             ('mpgge_elec', float64),
             ('veh', veh_type),
             ('cyc', cyc_type)
])

@jitclass(spec)
class SimDriveJit(SimDriveCore):
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle fuel economy simulations. This class will be 
    faster for large batch runs."""

    def __init__(self, cyc_jit, veh_jit):
        """Initializes typed numpy arrays for specific cycle
        Arguments:
        -----------
        cyc: instance of TypedCycle class generated from the 
            Vehicle.get_numba_cyc method
        veh: instance of TypedVehicle class generated from the 
            Vehicle.get_numba_veh method
        """
        self.veh = veh_jit
        self.cyc = cyc_jit

        len_cyc = len(self.cyc.cycSecs)
        # Component Limits -- calculated dynamically"
        self.curMaxFsKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.fcTransLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcFsLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcMaxKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxFcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxAvailElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecInLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcTransiLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenPercKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMechMcKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTransKwOut = np.zeros(len_cyc, dtype=np.float64)

        ### Drive Train
        self.cycDragKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAccelKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAscentKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTracKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.spareTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycRrKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelRadPerSec = np.zeros(len_cyc, dtype=np.float64)
        self.cycTireInertiaKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.regenContrLimKwPerc = np.zeros(len_cyc, dtype=np.float64)
        self.cycRegenBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycFricBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTransKwOutReq = np.zeros(len_cyc, dtype=np.float64)
        self.cycMet = np.zeros(len_cyc, dtype=np.float64)
        self.transKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.transKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.curSocTarget = np.zeros(len_cyc, dtype=np.float64)
        self.minMcKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.mcMechKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.auxInKw = np.zeros(len_cyc, dtype=np.float64)
        self.roadwayChgKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.minEssKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.essKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch_pct = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwhOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.essCurKwh = np.zeros(len_cyc, dtype=np.float64)
        self.soc = np.zeros(len_cyc, dtype=np.float64)

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.essRegenBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssRegenBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssAccelBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelRegenDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElectInKwForMaxFcEff = np.zeros(len_cyc, dtype=np.float64)
        self.electKwReq4AE = np.zeros(len_cyc, dtype=np.float64)
        self.canPowerAllElectrically = np.array(
            [False] * len_cyc, dtype=np.bool_)
        self.desiredEssKwOutForAE = np.zeros(len_cyc, dtype=np.float64)
        self.essAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.erAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essDesiredKw4FcEff = np.zeros(len_cyc, dtype=np.float64)
        self.essKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcElecKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwGapFrEff = np.zeros(len_cyc, dtype=np.float64)
        self.erKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.fcForcedOn = np.array([False] * len_cyc, dtype=np.bool_)
        self.fcForcedState = np.zeros(len_cyc, dtype=np.int32)
        self.mcMechKw4ForcedFc = np.zeros(len_cyc, dtype=np.float64)
        self.fcTimeOn = np.zeros(len_cyc, dtype=np.float64)
        self.prevfcTimeOn = np.zeros(len_cyc, dtype=np.float64)
        
        ### Additional Variables
        self.mpsAch = np.zeros(len_cyc, dtype=np.float64)
        self.mphAch = np.zeros(len_cyc, dtype=np.float64)
        self.distMeters = np.zeros(len_cyc, dtype=np.float64)
        self.distMiles = np.zeros(len_cyc, dtype=np.float64)
        self.highAccFcOnTag = np.zeros(len_cyc, dtype=np.float64)
        self.reachedBuff = np.zeros(len_cyc, dtype=np.float64)
        self.maxTracMps = np.zeros(len_cyc, dtype=np.float64)
        self.addKwh = np.zeros(len_cyc, dtype=np.float64)
        self.dodCycs = np.zeros(len_cyc, dtype=np.float64)
        self.essPercDeadArray = np.zeros(len_cyc, dtype=np.float64)
        self.dragKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLossKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelKw = np.zeros(len_cyc, dtype=np.float64)
        self.ascentKw = np.zeros(len_cyc, dtype=np.float64)
        self.rrKw = np.zeros(len_cyc, dtype=np.float64)
        self.motor_index_debug = np.zeros(len_cyc, dtype=np.float64)
        self.debug_flag = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxRoadwayChgKw = np.zeros(len_cyc, dtype=np.float64)

    def sim_drive(self, initSoc):
        """Initialize and run sim_drive_sub as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        initSoc: initial SOC for electrified vehicles.  
            Use -1 for default value.  Otherwise, must be between 0 and 1."""

        if initSoc != -1:
            if initSoc > 1.0 or initSoc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = -1
    
        if self.veh.vehPtType == 1: # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            
            self.sim_drive_sub(initSoc)

        elif self.veh.vehPtType == 2 and initSoc == -1:  # HEV 

            #####################################
            ### Charge Balancing Vehicle SOC ###
            #####################################

            # Charge balancing SOC for PHEV vehicle types. Iterating initsoc and comparing to final SOC.
            # Iterating until tolerance met or 30 attempts made.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            ess2fuelKwh = 1.0
            sim_count = 0
            while ess2fuelKwh > self.veh.essToFuelOkError and sim_count < 30:
                sim_count += 1
                self.sim_drive_sub(initSoc)
                fuelKj = np.sum(self.fsKwOutAch * self.cyc.secs)
                roadwayChgKj = np.sum(self.roadwayChgKwOutAch * self.cyc.secs)
                ess2fuelKwh = np.abs((self.soc[0] - self.soc[-1]) * 
                    self.veh.maxEssKwh * 3600 / (fuelKj + roadwayChgKj))
                initSoc = min(1.0, max(0.0, self.soc[-1]))
                        
            self.sim_drive_sub(initSoc)

        elif (self.veh.vehPtType == 3 and initSoc == -1) or (self.veh.vehPtType == 4 and initSoc == -1): # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = self.veh.maxSoc
            
            self.sim_drive_sub(initSoc)

        else:
            
            self.sim_drive_sub(initSoc)
        
@jitclass(spec)
class SimAccelTestJit(SimDriveCore):
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle acceleration simulation. This class will be 
    faster for large batch runs."""

    def __init__(self, cyc_jit, veh_jit):
        """Initializes typed numpy arrays for specific cycle
        Arguments:
        -----------
        cyc_jit: instance of TypedCycle class generated from the 
            Vehicle.get_numba_cyc method for the 'accel' cycle
        veh_jit: instance of TypedVehicle class generated from the 
            Vehicle.get_numba_veh method
        """
        self.veh = veh_jit
        self.cyc = cyc_jit

        len_cyc = len(self.cyc.cycSecs)
        # Component Limits -- calculated dynamically"
        self.curMaxFsKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.fcTransLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcFsLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcMaxKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxFcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxAvailElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecInLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcTransiLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenPercKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMechMcKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTransKwOut = np.zeros(len_cyc, dtype=np.float64)

        ### Drive Train
        self.cycDragKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAccelKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAscentKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTracKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.spareTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycRrKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelRadPerSec = np.zeros(len_cyc, dtype=np.float64)
        self.cycTireInertiaKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.regenContrLimKwPerc = np.zeros(len_cyc, dtype=np.float64)
        self.cycRegenBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycFricBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTransKwOutReq = np.zeros(len_cyc, dtype=np.float64)
        self.cycMet = np.zeros(len_cyc, dtype=np.float64)
        self.transKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.transKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.curSocTarget = np.zeros(len_cyc, dtype=np.float64)
        self.minMcKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.mcMechKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.auxInKw = np.zeros(len_cyc, dtype=np.float64)
        self.roadwayChgKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.minEssKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.essKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch_pct = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwhOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.essCurKwh = np.zeros(len_cyc, dtype=np.float64)
        self.soc = np.zeros(len_cyc, dtype=np.float64)

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.essRegenBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssRegenBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssAccelBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelRegenDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElectInKwForMaxFcEff = np.zeros(len_cyc, dtype=np.float64)
        self.electKwReq4AE = np.zeros(len_cyc, dtype=np.float64)
        self.canPowerAllElectrically = np.array(
            [False] * len_cyc, dtype=np.bool_)
        self.desiredEssKwOutForAE = np.zeros(len_cyc, dtype=np.float64)
        self.essAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.erAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essDesiredKw4FcEff = np.zeros(len_cyc, dtype=np.float64)
        self.essKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcElecKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwGapFrEff = np.zeros(len_cyc, dtype=np.float64)
        self.erKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.fcForcedOn = np.array([False] * len_cyc, dtype=np.bool_)
        self.fcForcedState = np.zeros(len_cyc, dtype=np.int32)
        self.mcMechKw4ForcedFc = np.zeros(len_cyc, dtype=np.float64)
        self.fcTimeOn = np.zeros(len_cyc, dtype=np.float64)
        self.prevfcTimeOn = np.zeros(len_cyc, dtype=np.float64)

        ### Additional Variables
        self.mpsAch = np.zeros(len_cyc, dtype=np.float64)
        self.mphAch = np.zeros(len_cyc, dtype=np.float64)
        self.distMeters = np.zeros(len_cyc, dtype=np.float64)
        self.distMiles = np.zeros(len_cyc, dtype=np.float64)
        self.highAccFcOnTag = np.zeros(len_cyc, dtype=np.float64)
        self.reachedBuff = np.zeros(len_cyc, dtype=np.float64)
        self.maxTracMps = np.zeros(len_cyc, dtype=np.float64)
        self.addKwh = np.zeros(len_cyc, dtype=np.float64)
        self.dodCycs = np.zeros(len_cyc, dtype=np.float64)
        self.essPercDeadArray = np.zeros(len_cyc, dtype=np.float64)
        self.dragKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLossKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelKw = np.zeros(len_cyc, dtype=np.float64)
        self.ascentKw = np.zeros(len_cyc, dtype=np.float64)
        self.rrKw = np.zeros(len_cyc, dtype=np.float64)
        self.motor_index_debug = np.zeros(len_cyc, dtype=np.float64)
        self.debug_flag = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxRoadwayChgKw = np.zeros(len_cyc, dtype=np.float64)

    def sim_drive(self):
        """Initialize and run sim_drive_sub as appropriate for vehicle attribute vehPtType."""

        if self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_sub(initSoc)

        elif self.veh.vehPtType == 2:  # HEV

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_sub(initSoc)

        else:

            # If EV, initializing initial SOC to maximum SOC.
            initSoc = self.veh.maxSoc
            self.sim_drive_sub(initSoc)

class SimAccelTest(SimDriveCore):
    """Class for running FASTSim vehicle acceleration simulation."""

    def __init__(self, cyc, veh):
        """Initializes typed numpy arrays for specific cycle
        Arguments:
        -----------
        cyc: instance of Cycle class for the 'accel' cycle
        veh: instance of Vehicle class 
        """
        self.veh = veh
        self.cyc = cyc

        len_cyc = len(self.cyc.cycSecs)
        # Component Limits -- calculated dynamically"
        self.curMaxFsKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.fcTransLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcFsLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcMaxKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxFcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxAvailElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecInLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcTransiLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenPercKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMechMcKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTransKwOut = np.zeros(len_cyc, dtype=np.float64)

        ### Drive Train
        self.cycDragKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAccelKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAscentKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTracKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.spareTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycRrKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelRadPerSec = np.zeros(len_cyc, dtype=np.float64)
        self.cycTireInertiaKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.regenContrLimKwPerc = np.zeros(len_cyc, dtype=np.float64)
        self.cycRegenBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycFricBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTransKwOutReq = np.zeros(len_cyc, dtype=np.float64)
        self.cycMet = np.zeros(len_cyc, dtype=np.float64)
        self.transKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.transKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.curSocTarget = np.zeros(len_cyc, dtype=np.float64)
        self.minMcKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.mcMechKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.auxInKw = np.zeros(len_cyc, dtype=np.float64)
        self.roadwayChgKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.minEssKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.essKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch_pct = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwhOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.essCurKwh = np.zeros(len_cyc, dtype=np.float64)
        self.soc = np.zeros(len_cyc, dtype=np.float64)

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.essRegenBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssRegenBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssAccelBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelRegenDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElectInKwForMaxFcEff = np.zeros(len_cyc, dtype=np.float64)
        self.electKwReq4AE = np.zeros(len_cyc, dtype=np.float64)
        self.canPowerAllElectrically = np.array(
            [False] * len_cyc, dtype=np.bool_)
        self.desiredEssKwOutForAE = np.zeros(len_cyc, dtype=np.float64)
        self.essAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.erAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essDesiredKw4FcEff = np.zeros(len_cyc, dtype=np.float64)
        self.essKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcElecKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwGapFrEff = np.zeros(len_cyc, dtype=np.float64)
        self.erKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.fcForcedOn = np.array([False] * len_cyc, dtype=np.bool_)
        self.fcForcedState = np.zeros(len_cyc, dtype=np.int32)
        self.mcMechKw4ForcedFc = np.zeros(len_cyc, dtype=np.float64)
        self.fcTimeOn = np.zeros(len_cyc, dtype=np.float64)
        self.prevfcTimeOn = np.zeros(len_cyc, dtype=np.float64)

        ### Additional Variables
        self.mpsAch = np.zeros(len_cyc, dtype=np.float64)
        self.mphAch = np.zeros(len_cyc, dtype=np.float64)
        self.distMeters = np.zeros(len_cyc, dtype=np.float64)
        self.distMiles = np.zeros(len_cyc, dtype=np.float64)
        self.highAccFcOnTag = np.zeros(len_cyc, dtype=np.float64)
        self.reachedBuff = np.zeros(len_cyc, dtype=np.float64)
        self.maxTracMps = np.zeros(len_cyc, dtype=np.float64)
        self.addKwh = np.zeros(len_cyc, dtype=np.float64)
        self.dodCycs = np.zeros(len_cyc, dtype=np.float64)
        self.essPercDeadArray = np.zeros(len_cyc, dtype=np.float64)
        self.dragKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLossKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelKw = np.zeros(len_cyc, dtype=np.float64)
        self.ascentKw = np.zeros(len_cyc, dtype=np.float64)
        self.rrKw = np.zeros(len_cyc, dtype=np.float64)
        self.motor_index_debug = np.zeros(len_cyc, dtype=np.float64)
        self.debug_flag = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxRoadwayChgKw = np.zeros(len_cyc, dtype=np.float64)

    def sim_drive(self):
        """Initialize and run sim_drive_sub as appropriate for vehicle attribute vehPtType."""

        if self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_sub(initSoc)

        elif self.veh.vehPtType == 2:  # HEV

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_sub(initSoc)

        else:

            # If EV, initializing initial SOC to maximum SOC.
            initSoc = self.veh.maxSoc
            self.sim_drive_sub(initSoc)

class SimDrivePost(object):
    """Class for post-processing of SimDrive instance.  Requires already-run 
    SimDriveJit or SimDriveClassic instance."""
    def __init__(self, sim_drive):
        """Arguments:
        ---------------
        sim_drive: solved sim_drive object"""

        super().__init__()

        sim_drive.set_post_scalars()

        for item in spec:
            self.__setattr__(item[0], sim_drive.__getattribute__(item[0]))

    def get_output(self):
        """Calculate finalized results
        Arguments
        ------------
        initSoc: initial SOC for electrified vehicles
        
        Returns
        ------------
        output: dict of summary output variables"""

        output = {}

        output['mpgge'] = self.mpgge
        output['battery_kWh_per_mi'] = self.battery_kWh_per_mi
        output['electric_kWh_per_mi'] = self.electric_kWh_per_mi
        output['maxTraceMissMph'] = mphPerMps * \
            max(abs(self.cyc.cycMps - self.mpsAch))
        self.maxTraceMissMph = output['maxTraceMissMph']

        output['ess2fuelKwh'] = self.ess2fuelKwh

        output['initial_soc'] = self.soc[0]
        output['final_soc'] = self.soc[-1]

        output['mpgge_elec'] = self.mpgge_elec
        output['soc'] = self.soc
        output['distance_mi'] = sum(self.distMiles)
        duration_sec = self.cyc.cycSecs[-1] - self.cyc.cycSecs[0]
        output['avg_speed_mph'] = sum(
            self.distMiles) / (duration_sec / 3600.0)
        self.avg_speed_mph = output['avg_speed_mph']
        self.accel = np.diff(self.mphAch) / np.diff(self.cyc.cycSecs)
        output['avg_accel_mphps'] = np.mean(self.accel[self.accel > 0])
        self.avg_accel_mphps = output['avg_accel_mphps']

        if max(self.mphAch) > 60:
            output['ZeroToSixtyTime_secs'] = np.interp(60, self.mphAch, self.cyc.cycSecs)

        else:
            output['ZeroToSixtyTime_secs'] = 0.0

        #######################################################################
        ####  Time series information for additional analysis / debugging. ####
        ####             Add parameters of interest as needed.             ####
        #######################################################################

        output['fcKwOutAch'] = np.asarray(self.fcKwOutAch)
        output['fsKwhOutAch'] = np.asarray(self.fsKwhOutAch)
        output['fcKwInAch'] = np.asarray(self.fcKwInAch)
        output['time'] = np.asarray(self.cyc.cycSecs)

        return output

    # optional post-processing methods
    def get_diagnostics(self):
        """This method is to be run after runing sim_drive, if diagnostic variables 
        are needed.  Diagnostic variables are returned in a dict.  Diagnostic variables include:
        - final integrated value of all positive powers
        - final integrated value of all negative powers
        - total distance traveled
        - miles per gallon gasoline equivalent (mpgge)"""
        
        base_var_list = list(self.__dict__.keys())
        pw_var_list = [var for var in base_var_list if re.search(
            '\w*Kw(?!h)\w*', var)] 
            # find all vars containing 'Kw' but not 'Kwh'
        
        prog = re.compile('(\w*)Kw(?!h)(\w*)') 
        # find all vars containing 'Kw' but not Kwh and capture parts before and after 'Kw'
        # using compile speeds up iteration

        # create positive and negative versions of all time series with units of kW
        # then integrate to find cycle end pos and negative energies
        tempvars = {} # dict for contaning intermediate variables
        output = {}
        for var in pw_var_list:
            tempvars[var + 'Pos'] = [x if x >= 0 
                                        else 0 
                                        for x in self.__getattribute__(var)]
            tempvars[var + 'Neg'] = [x if x < 0 
                                        else 0 
                                        for x in self.__getattribute__(var)]    
                        
            # Assign values to output dict for positive and negative energy variable names
            search = prog.search(var)
            output[search[1] + 'Kj' + search[2] + 'Pos'] = np.trapz(tempvars[var + 'Pos'], self.cyc.cycSecs)
            output[search[1] + 'Kj' + search[2] + 'Neg'] = np.trapz(tempvars[var + 'Neg'], self.cyc.cycSecs)
        
        output['distMilesFinal'] = sum(self.distMiles)
        output['mpgge'] = sum(self.distMiles) / sum(self.fsKwhOutAch) * kWhPerGGE
    
        return output

    def set_battery_wear(self):
        """Battery wear calcs"""

        self.addKwh[1:] = np.array([
            (self.essCurKwh[i] - self.essCurKwh[i-1]) + self.addKwh[i-1]
            if self.essCurKwh[i] > self.essCurKwh[i-1]
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.dodCycs[1:] = np.array([
            self.addKwh[i-1] / self.veh.maxEssKwh if self.addKwh[i] == 0
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.essPercDeadArray = np.array([
            np.power(self.veh.essLifeCoefA, 1.0 / self.veh.essLifeCoefB) / np.power(self.dodCycs[i], 
            1.0 / self.veh.essLifeCoefB)
            if self.dodCycs[i] != 0
            else 0
            for i in range(0, len(self.essCurKwh))])

    def set_energy_audit(self):
        """Energy Audit Calculations
        Adapted from Excel:
        '=(SUM(roadwayChgKj,essDischgKj,fuelKj,keKj)-netKj)/SUM(fuelKj,essDischgKj,roadwayChgKj,keKj)'
        """

        self.dragKw = self.cycDragKw
        self.dragKj = (self.dragKw * self.cyc.secs).sum()
        self.ascentKw = self.cycAscentKw
        self.ascentKj = (self.ascentKw * self.cyc.secs).sum()
        self.rrKw = self.cycRrKw
        self.rrKj = (self.rrKw * self.cyc.secs).sum()
        
        self.brakeKj = (self.cycFricBrakeKw * self.cyc.secs).sum()
        self.transKj = ((self.transKwInAch - self.transKwOutAch) * self.cyc.secs).sum()
        self.mcKj = ((self.mcElecKwInAch - self.mcMechKwOutAch) * self.cyc.secs).sum()
        self.essEffKj = (self.essLossKw * self.cyc.secs).sum()
        self.auxKj = (self.auxInKw * self.cyc.secs).sum()
        self.fcKj = ((self.fcKwInAch - self.fcKwOutAch) * self.cyc.secs).sum()
        
        self.netKj = self.dragKj + self.ascentKj + self.rrKj + self.brakeKj + self.transKj \
            + self.mcKj + self.essEffKj + self.auxKj + self.fcKj

        self.keKj = 0.5 * self.veh.vehKg * \
            (self.cyc.cycMps[0]**2 - self.cyc.cycMps[-1]**2) / 1000
        
        self.energyAuditError = ((self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj) - self.netKj) /\
            (self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj)

        self.essLossKw[1:] = np.array(
            [0 if (self.veh.maxEssKw == 0 or self.veh.maxEssKwh == 0) 
            else -self.essKwOutAch[i] - (-self.essKwOutAch[i] * np.sqrt(self.veh.essRoundTripEff)) 
                if self.essKwOutAch[i] < 0 
            else self.essKwOutAch[i] * (1.0 / np.sqrt(self.veh.essRoundTripEff)) - self.essKwOutAch[i] 
            for i in range(1, len(self.cyc.cycSecs))])

        self.accelKw[1:] = (self.veh.vehKg / (2.0 * (self.cyc.secs[1:]))) * \
            ((self.mpsAch[1:]**2) - (self.mpsAch[:-1]**2)) / 1000.0

