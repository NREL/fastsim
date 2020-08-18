"""Module containing classes and methods for for loading vehicle and cycle data.
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
from pathlib import Path
import ast

# local modules
from . import parameters as params

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VEH_DB = os.path.abspath(
        os.path.join(
            THIS_DIR, '..', 'docs', 'FASTSim_py_veh_db.csv'))

props = params.PhysicalProperties()


class Vehicle(object):
    """Class for loading and contaning vehicle attributes
    Optional Arguments:
    ---------
    vnum: row number of vehicle to simulate in 'FASTSim_py_veh_db.csv'
    veh_file: string or filelike obj, alternative to default FASTSim_py_veh_db
    
    If a sinparamse vehicle veh_file is provided, vnum cannot be passed, and 
    veh_file must be passed a keyword argument."""

    def __init__(self, vnum=None, veh_file=None):
        super().__init__()
        if veh_file and vnum:
            self.load_veh(vnum, veh_file=veh_file)
        elif vnum and not veh_file:
            self.load_veh(vnum)
        else:
            # vnum = 0 tells load_veh that the file contains only 1 vehicle
            self.load_veh(0, veh_file=veh_file)

    def get_numba_veh(self):
        """Load numba JIT-compiled vehicle."""
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
    
    def load_veh(self, vnum, veh_file=None):
        """Load vehicle parameters from string or filelike obj, 
        alternative to default FASTSim_py_veh_db
        Arguments:
        ---------
        vnum: row number of vehicle to simulate in 'FASTSim_py_veh_db.csv'
        veh_file (optional override): string or filelike obj, alternative 
        to default FASTSim_py_veh_db"""

        if vnum != 0:
            if veh_file:
                vehdf = pd.read_csv(Path(veh_file))
            else:
                vehdf = pd.read_csv(DEFAULT_VEH_DB)
        else:
            vehdf = pd.read_csv(Path(veh_file))
            vehdf = vehdf.transpose()
            vehdf.columns = vehdf.iloc[1]
            vehdf.drop(vehdf.index[0], inplace=True)
            vehdf['Selection'] = np.nan * np.ones(vehdf.shape[0])
            vehdf.loc['Param Value', 'Selection'] = 0
            vnum = 0
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
            # remove percent signs if any are found,
            # but sometimes they exist in names / scenario names, so allow for that
            if '%' in data:
                _data = data.replace('%', '')
                try:
                    data = float(_data)
                    data = data / 100.0
                    return data
                except:
                    # not supposed to be numeric, quit
                    pass

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

        # empty strings for cells that had no values easier to deal with
        vehdf.loc[vnum] = vehdf.loc[vnum].apply(clean_data)

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

        # Power and efficiency arrays are defined in parameters.py
        # Can also be input in CSV as array under column fcEffMap of form
        # [0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30]
        # no quotes necessary
        try:
            self.fcEffMap = np.array(ast.literal_eval(self.fcEffMap))
        except ValueError:
            if self.fcEffType == 1:  # SI engine
                self.fcEffMap = params.eff_si + self.fcAbsEffImpr

            elif self.fcEffType == 2:  # Atkinson cycle SI engine -- greater expansion
                self.fcEffMap = params.eff_atk + self.fcAbsEffImpr

            elif self.fcEffType == 3:  # Diesel (compression ignition) engine
                self.fcEffMap = params.eff_diesel + self.fcAbsEffImpr

            elif self.fcEffType == 4:  # H2 fuel cell
                self.fcEffMap = params.eff_fuel_cell + self.fcAbsEffImpr

            elif self.fcEffType == 5:  # heavy duty Diesel engine
                self.fcEffMap = params.eff_hd_diesel + self.fcAbsEffImpr
        if len(self.fcEffMap) != 12:
            raise ValueError('fcEffMap has length of {}, but should have length of 12'.
                format(len(self.fcEffMap)))

        # discrete array of possible engine power outputs
        inputKwOutArray = params.fcPwrOutPerc * self.maxFuelConvKw
        # Relatively continuous array of possible engine power outputs
        fcKwOutArray = self.maxFuelConvKw * params.fcPercOutArray
        # Initializes relatively continuous array for fcEFF
        fcEffArray = np.zeros(len(params.fcPercOutArray))

        # the following for loop populates fcEffArray
        for j in range(0, len(params.fcPercOutArray) - 1):
            low_index = np.argmax(inputKwOutArray >= fcKwOutArray[j])
            fcinterp_x_1 = inputKwOutArray[low_index-1]
            fcinterp_x_2 = inputKwOutArray[low_index]
            fcinterp_y_1 = self.fcEffMap[low_index-1]
            fcinterp_y_2 = self.fcEffMap[low_index]
            fcEffArray[j] = (fcKwOutArray[j] - fcinterp_x_1)/(fcinterp_x_2 -
                                fcinterp_x_1) * (fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        # populate final value
        fcEffArray[-1] = self.fcEffMap[-1]

        # assign corresponding values in veh dict
        self.fcEffArray = fcEffArray
        self.fcKwOutArray = fcKwOutArray
        self.maxFcEffKw = self.fcKwOutArray[np.argmax(fcEffArray)]
        self.fcMaxOutkW = np.max(inputKwOutArray)
            
        ### Defining MC efficiency curve as lookup table for %power_in vs power_out
        ### see "Motor" tab in FASTSim for Excel

        # Power and efficiency arrays are defined in parameters.py
        # can also be overridden by motor power and efficiency columns in the input file
        # ensure that the column existed and the value in the cell wasn't empty (becomes NaN)
        try:
            self.mcPwrOutPerc = np.array(ast.literal_eval(self.mcPwrOutPerc))
        except ValueError:
            self.mcPwrOutPerc = params.mcPwrOutPerc
        if len(self.mcPwrOutPerc) != 11:
            raise ValueError('mcPwrOutPerc has length of {}, but should have length of 11'.
                             format(len(self.mcPwrOutPerc)))

        try:
            self.largeBaselineEff = np.array(ast.literal_eval(self.largeBaselineEff))
        except ValueError:
            self.largeBaselineEff = params.large_baseline_eff
        if len(self.largeBaselineEff) != 11:
            raise ValueError('largeBaselineEff has length of {}, but should have length of 11'.
                             format(len(self.largeBaselineEff)))
        
        try:
            self.smallBaselineEff = np.array(ast.literal_eval(self.smallBaselineEff))
        except ValueError:
            self.smallBaselineEff = params.small_baseline_eff
        if len(self.smallBaselineEff) != 11:
            raise ValueError('smallBaselineEff has length of {}, but should have length of 11'.
                             format(len(self.smallBaselineEff)))

        if np.isnan(self.modernMax):
            self.modernMax = params.modern_max            
        
        modern_diff = self.modernMax - max(self.largeBaselineEff)

        large_baseline_eff_adj = self.largeBaselineEff + modern_diff

        mcKwAdjPerc = max(0.0, min((self.maxMotorKw - 7.5)/(75.0 - 7.5), 1.0))
        mcEffArray = np.zeros(len(self.mcPwrOutPerc))

        for k in range(0, len(self.mcPwrOutPerc)):
            mcEffArray[k] = mcKwAdjPerc * large_baseline_eff_adj[k] + \
                (1 - mcKwAdjPerc)*(self.smallBaselineEff[k])

        mcInputKwOutArray = self.mcPwrOutPerc * self.maxMotorKw
        mcFullEffArray = np.zeros(len(params.mcPercOutArray))
        mcKwOutArray = np.linspace(0, 1, len(params.mcPercOutArray)) * self.maxMotorKw

        for m in range(1, len(params.mcPercOutArray) - 1):
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
        ess_mass_kg = 0
        mc_mass_kg = 0
        fc_mass_kg = 0
        fs_mass_kg = 0

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

        self.maxTracMps2 = ((((self.wheelCoefOfFric * self.driveAxleWeightFrac * self.vehKg * props.gravityMPerSec2) /
                              (1+((self.vehCgM * self.wheelCoefOfFric) / self.wheelBaseM))))/(self.vehKg * props.gravityMPerSec2)) * props.gravityMPerSec2
        self.maxRegenKwh = 0.5 * self.vehKg * (27**2) / (3600 * 1000)

        # for stats and interest
        self.essMassKg = ess_mass_kg
        self.mcMassKg =  mc_mass_kg
        self.fcMassKg =  fc_mass_kg
        self.fsMassKg =  fs_mass_kg


# type specifications for attributions of Vehicle class
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
