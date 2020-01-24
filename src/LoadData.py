"""Module containing function for loading drive cycle data (e.g. speed trace)
and class (Vehicle) for loading and storing vehicle attribute data.  For example usage, 
see ../README.md"""

import pandas as pd
from Globals import *
import numpy as np
import re
from numba import jitclass                 # import the decorator
from numba import float32, int32, bool_    # import the types
from numba import types, typed

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
        numba_cyc = TypedCycle(len(self.cycSecs))
        for key in self.__dict__.keys():
            numba_cyc.__setattr__(key, self.__getattribute__(key).astype(np.float32))
        return numba_cyc

    def set_standard_cycle(self, std_cyc_name):
        """Load time trace of speed, grade, and road type in a pandas dataframe.
        Argument:
        ---------
        std_cyc_name: cycle name string (e.g. 'udds', 'us06', 'hwfet')"""
        csv_path = '..//cycles//' + std_cyc_name + '.csv'
        cyc = pd.read_csv(csv_path)
        for column in cyc.columns:
            self.__setattr__(column, cyc[column].copy().to_numpy())
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
        self.cycMph = np.copy(self.cycMps * mphPerMps)
        self.secs = np.insert(np.diff(self.cycSecs), 0, 0)

# kv_ty = (types.unicode_type, types.float32)
cyc_spec = [('cycSecs', float32[:]),
            ('cycMps', float32[:]),
            ('cycGrade', float32[:]),
            ('cycRoadType', float32[:]),
            ('cycMph', float32[:]),
            ('secs', float32[:])
]
            # ('cycdict', types.DictType(*kv_ty))]
            # ('std_cyc_name', types.unicode_type)]

@jitclass(cyc_spec)
class TypedCycle(object):
    """Object for containing time, speed, road grade, and road charging vectors 
    for drive cycle."""
    def __init__(self, len_cyc):
        self.cycSecs = np.zeros(len_cyc, dtype=np.float32)
        self.cycMps = np.zeros(len_cyc, dtype=np.float32)
        self.cycGrade = np.zeros(len_cyc, dtype=np.float32)
        self.cycRoadType = np.zeros(len_cyc, dtype=np.float32)
        self.cycMph = np.zeros(len_cyc, dtype=np.float32)
        self.secs = np.zeros(len_cyc, dtype=np.float32)
        # self.cycdict = typed.Dict.empty(*kv_ty)

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
        numba_veh = TypedVehicle()
        for item in veh_spec:
            if (type(self.__getattribute__(item[0])) == np.ndarray) | (type(self.__getattribute__(item[0])) == np.float64):
                numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.float32))
            elif type(self.__getattribute__(item[0])) == np.int64:
                numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.int32))
            else:
                numba_veh.__setattr__(
                    item[0], self.__getattribute__(item[0]))
            
        return numba_veh
    
    def load_vnum(self, vnum):
        """Load vehicle parameters based on vnum and assign to self.
        Argument:
        ---------
        vnum: row number of vehicle to simulate in 'FASTSim_py_veh_db.csv'"""

        vehdf = pd.read_csv('..//docs//FASTSim_py_veh_db.csv')
        vehdf.set_index('Selection', inplace=True, drop=False)
        # vehdf = vehdf.loc[[vnum], :]

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

        if self.maxFuelConvKw > 0:

            # Power and efficiency arrays are defined in Globals.py
            
            if self.fcEffType == 1:  # SI engine
                eff = np.copy(eff_si) + self.fcAbsEffImpr

            elif self.fcEffType == 2:  # Atkinson cycle SI engine -- greater expansion
                eff = np.copy(eff_atk) + self.fcAbsEffImpr

            elif self.fcEffType == 3:  # Diesel (compression ignition) engine
                eff = np.copy(eff_diesel) + self.fcAbsEffImpr

            elif self.fcEffType == 4:  # H2 fuel cell
                eff = np.copy(eff_fuel_cell) + self.fcAbsEffImpr

            elif self.fcEffType == 5:  # heavy duty Diesel engine
                eff = np.copy(eff_hd_diesel) + self.fcAbsEffImpr

            # discrete array of possible engine power outputs
            inputKwOutArray = fcPwrOutPerc * self.maxFuelConvKw
            # Relatively continuous array of possible engine power outputs
            fcKwOutArray = self.maxFuelConvKw * fcPercOutArray
            # Initializes relatively continuous array for fcEFF
            fcEffArray = np.array([0.0] * len(fcPercOutArray))

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
            self.fcEffArray = np.copy(fcEffArray)
            self.fcKwOutArray = np.copy(fcKwOutArray)
            self.maxFcEffKw = self.fcKwOutArray[np.argmax(fcEffArray)]
            self.fcMaxOutkW = np.max(inputKwOutArray)
            
        else:
            # these things are all zero for BEV powertrains
            # not sure why `self.fcEffArray` is not being assigned.
            # Maybe it's not used anywhere in this condition.  *** delete this comment before public release
            self.fcKwOutArray = np.array([0] * 101)
            self.maxFcEffKw = 0
            self.fcMaxOutkW = 0
            
        ### Defining MC efficiency curve as lookup table for %power_in vs power_out
        ### see "Motor" tab in FASTSim for Excel
        if self.maxMotorKw > 0:

            maxMotorKw = self.maxMotorKw
            
            # Power and efficiency arrays are defined in Globals.py

            modern_diff = modern_max - max(large_baseline_eff)

            large_baseline_eff_adj = large_baseline_eff + modern_diff

            mcKwAdjPerc = max(0.0, min((maxMotorKw - 7.5)/(75.0 - 7.5), 1.0))
            mcEffArray = np.array([0.0] * len(mcPwrOutPerc))

            for k in range(0, len(mcPwrOutPerc)):
                mcEffArray[k] = mcKwAdjPerc * large_baseline_eff_adj[k] + \
                    (1 - mcKwAdjPerc)*(small_baseline_eff[k])

            mcInputKwOutArray = mcPwrOutPerc * maxMotorKw
            mcFullEffArray = np.array([0.0] * len(mcPercOutArray))
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

            self.mcKwInArray = np.copy(mcKwInArray)
            self.mcKwOutArray = np.copy(mcKwOutArray)
            self.mcMaxElecInKw = np.copy(max(mcKwInArray))
            self.mcFullEffArray = np.copy(mcFullEffArray)
            self.mcEffArray = np.copy(mcEffArray)

            if 'motorAccelAssist' in self.__dir__() and np.isnan(self.__getattribute__('motorAccelAssist')):
                self.motorAccelAssist = True

        else:
            self.mcKwInArray = np.array([0.0] * len(mcPercOutArray))
            self.mcKwOutArray = np.array([0.0] * len(mcPercOutArray))
            self.mcMaxElecInKw = 0

        self.mcMaxElecInKw = max(self.mcKwInArray)

        ### Specify shape of mc regen efficiency curve
        ### see "Regen" tab in FASTSim for Excel
        self.regenA = 500.0  # hardcoded
        self.regenB = 0.99  # hardcoded

    def set_veh_mass(self):
        """Calculate total vehicle mass.  Sum up component masses if 
        positive real number is not specified for vehOverrideKg"""
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
            self.vehKg = np.copy(self.vehOverrideKg)

        self.maxTracMps2 = ((((self.wheelCoefOfFric * self.driveAxleWeightFrac * self.vehKg * gravityMPerSec2) /
                              (1+((self.vehCgM * self.wheelCoefOfFric) / self.wheelBaseM))))/(self.vehKg * gravityMPerSec2)) * gravityMPerSec2
        self.maxRegenKwh = 0.5 * self.vehKg * (27**2) / (3600 * 1000)

veh_spec = [('Selection', int32),
    ('Scenario_name', types.unicode_type),
    ('vehPtType', int32),
    ('dragCoef', float32),
    ('frontalAreaM2', float32),
    ('gliderKg', float32),
    ('vehCgM', float32),
    ('driveAxleWeightFrac', float32),
    ('wheelBaseM', float32),
    ('cargoKg', float32),
    ('vehOverrideKg', float32),
    ('maxFuelStorKw', float32),
    ('fuelStorSecsToPeakPwr', float32),
    ('fuelStorKwh', float32),
    ('fuelStorKwhPerKg', float32),
    ('maxFuelConvKw', float32),
    ('fcEffType', int32),
    ('fcAbsEffImpr', float32),
    ('fuelConvSecsToPeakPwr', float32),
    ('fuelConvBaseKg', float32),
    ('fuelConvKwPerKg', float32),
    ('maxMotorKw', float32),
    ('motorPeakEff', float32),
    ('motorSecsToPeakPwr', float32),
    ('mcPeKgPerKw', float32),
    ('mcPeBaseKg', float32),
    ('maxEssKw', float32),
    ('maxEssKwh', float32),
    ('essKgPerKwh', float32),
    ('essBaseKg', float32),
    ('essRoundTripEff', float32),
    ('essLifeCoefA', float32),
    ('essLifeCoefB', float32),
    ('wheelInertiaKgM2', float32),
    ('numWheels', float32),
    ('wheelRrCoef', float32),
    ('wheelRadiusM', float32),
    ('wheelCoefOfFric', float32),
    ('minSoc', float32),
    ('maxSoc', float32),
    ('essDischgToFcMaxEffPerc', float32),
    ('essChgToFcMaxEffPerc', float32),
    ('maxAccelBufferMph', float32),
    ('maxAccelBufferPercOfUseableSoc', float32),
    ('percHighAccBuf', float32),
    ('mphFcOn', float32),
    ('kwDemandFcOn', float32),
    ('altEff', float32),
    ('chgEff', float32),
    ('auxKw', float32),
    ('forceAuxOnFC', bool_),
    ('transKg', float32),
    ('transEff', float32),
    ('compMassMultiplier', float32),
    ('essToFuelOkError', float32),
    ('maxRegen', float32),
    ('valUddsMpgge', float32),
    ('valHwyMpgge', float32),
    ('valCombMpgge', float32),
    ('valUddsKwhPerMile', float32),
    ('valHwyKwhPerMile', float32),
    ('valCombKwhPerMile', float32),
    ('valCdRangeMi', float32),
    ('valConst65MphKwhPerMile', float32),
    ('valConst60MphKwhPerMile', float32),
    ('valConst55MphKwhPerMile', float32),
    ('valConst45MphKwhPerMile', float32),
    ('valUnadjUddsKwhPerMile', float32),
    ('valUnadjHwyKwhPerMile', float32),
    ('val0To60Mph', float32),
    ('valEssLifeMiles', float32),
    ('valRangeMiles', float32),
    ('valVehBaseCost', float32),
    ('valMsrp', float32),
    ('minFcTimeOn', float32),
    ('idleFcKw', float32),
    ('MaxRoadwayChgKw', float32[:]),
    ('chargingOn', bool_),
    ('noElecSys', bool_),
    ('noElecAux', bool_),
    ('vehTypeSelection', int32),
    ('fcEffArray', float32[:]),
    ('fcKwOutArray', float32[:]),
    ('maxFcEffKw', float32),
    ('fcMaxOutkW', float32),
    ('mcKwInArray', float32[:]),
    ('mcKwOutArray', float32[:]),
    ('mcMaxElecInKw', float32),
    ('mcFullEffArray', float32[:]),
    ('mcEffArray', float32[:]),
    ('regenA', float32),
    ('regenB', float32),
    ('vehKg', float32),
    ('maxTracMps2', float32),
    ('maxRegenKwh', float32)
]

@jitclass(veh_spec)
class TypedVehicle(object):
    """fancy numba vehicle"""
    
    def __init__(self):
       self.Selection = 0
       self.Scenario_name = 'n/a'
       self.vehPtType = 0
       self.dragCoef = 0
       self.frontalAreaM2 = 0
       self.gliderKg = 0
       self.vehCgM = 0
       self.driveAxleWeightFrac = 0
       self.wheelBaseM = 0
       self.cargoKg = 0
       self.vehOverrideKg = 0
       self.maxFuelStorKw = 0
       self.fuelStorSecsToPeakPwr = 0
       self.fuelStorKwh = 0
       self.fuelStorKwhPerKg = 0
       self.maxFuelConvKw = 0
       self.fcEffType = 0
       self.fcAbsEffImpr = 0
       self.fuelConvSecsToPeakPwr = 0
       self.fuelConvBaseKg = 0
       self.fuelConvKwPerKg = 0
       self.maxMotorKw = 0
       self.motorPeakEff = 0
       self.motorSecsToPeakPwr = 0
       self.mcPeKgPerKw = 0
       self.mcPeBaseKg = 0
       self.maxEssKw = 0
       self.maxEssKwh = 0
       self.essKgPerKwh = 0
       self.essBaseKg = 0
       self.essRoundTripEff = 0
       self.essLifeCoefA = 0
       self.essLifeCoefB = 0
       self.wheelInertiaKgM2 = 0
       self.numWheels = 0
       self.wheelRrCoef = 0
       self.wheelRadiusM = 0
       self.wheelCoefOfFric = 0
       self.minSoc = 0
       self.maxSoc = 0
       self.essDischgToFcMaxEffPerc = 0
       self.essChgToFcMaxEffPerc = 0
       self.maxAccelBufferMph = 0
       self.maxAccelBufferPercOfUseableSoc = 0
       self.percHighAccBuf = 0
       self.mphFcOn = 0
       self.kwDemandFcOn = 0
       self.altEff = 0
       self.chgEff = 0
       self.auxKw = 0
       self.forceAuxOnFC = False
       self.transKg = 0
       self.transEff = 0
       self.compMassMultiplier = 0
       self.essToFuelOkError = 0
       self.maxRegen = 0
       self.valUddsMpgge = 0
       self.valHwyMpgge = 0
       self.valCombMpgge = 0
       self.valUddsKwhPerMile = 0
       self.valHwyKwhPerMile = 0
       self.valCombKwhPerMile = 0
       self.valCdRangeMi = 0
       self.valConst65MphKwhPerMile = 0
       self.valConst60MphKwhPerMile = 0
       self.valConst55MphKwhPerMile = 0
       self.valConst45MphKwhPerMile = 0
       self.valUnadjUddsKwhPerMile = 0
       self.valUnadjHwyKwhPerMile = 0
       self.val0To60Mph = 0
       self.valEssLifeMiles = 0
       self.valRangeMiles = 0
       self.valVehBaseCost = 0
       self.valMsrp = 0
       self.minFcTimeOn = 0
       self.idleFcKw = 0
       self.MaxRoadwayChgKw = np.zeros(6, dtype=np.float32)
       self.chargingOn = False
       self.noElecSys = False
       self.noElecAux = False
       self.vehTypeSelection = 0
       self.fcEffArray = np.zeros(100, dtype=np.float32)
       self.fcKwOutArray = np.zeros(100, dtype=np.float32)
       self.maxFcEffKw = 0
       self.fcMaxOutkW = 0
       self.mcKwInArray = np.zeros(101, dtype=np.float32)
       self.mcKwOutArray = np.zeros(101, dtype=np.float32)
       self.mcMaxElecInKw = 0
       self.mcFullEffArray = np.zeros(101, dtype=np.float32)
       self.mcEffArray = np.zeros(11, dtype=np.float32)
       self.regenA = 0
       self.regenB = 0
       self.vehKg = 0
       self.maxTracMps2 = 0
       self.maxRegenKwh = 0
