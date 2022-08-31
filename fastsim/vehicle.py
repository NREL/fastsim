"""
Module containing classes and methods for for loading vehicle data. For example usage, see ../README.md
"""

### Import necessary python modules
import numpy as np
import pandas as pd
import types as pytypes
import re
from pathlib import Path
from copy import deepcopy
import ast
import warnings
import numbers

# local modules
from fastsim import parameters as params

THIS_DIR = Path(__file__).parent
DEFAULT_VEH_DB = THIS_DIR / 'resources' / 'FASTSim_py_veh_db.csv'
DEFAULT_VEHDF = pd.read_csv(DEFAULT_VEH_DB)

__doc__ += f"""To create a new vehicle model, copy \n`{(THIS_DIR / 'resources/vehdb/template.csv').resolve()}`
to a working directory not inside \n`{THIS_DIR.resolve()}`
and edit as appropriate.
"""

def get_template_df():
    vehdf = pd.read_csv(THIS_DIR / 'resources' / 'vehdb' / 'template.csv')
    vehdf = vehdf.transpose()
    vehdf.columns = vehdf.iloc[1]
    vehdf.drop(vehdf.index[0], inplace=True)
    vehdf['Selection'] = np.nan * np.ones(vehdf.shape[0])
    vehdf.loc['Param Value', 'Selection'] = 0
    return vehdf

TEMPLATE_VEHDF = get_template_df()

# list of optional parameters
OPT_INIT_PARAMS = ['fcPeakEffOverride', 'mcPeakEffOverride']

VEH_PT_TYPES = ("Conv", "HEV", "PHEV", "BEV")
CONV = VEH_PT_TYPES[0]
HEV = VEH_PT_TYPES[1]
PHEV = VEH_PT_TYPES[2]
BEV = VEH_PT_TYPES[3]

FC_EFF_TYPES = ("SI", "Atkinson", "Diesel", "H2FC", "HD_Diesel")
SI = FC_EFF_TYPES[0]
ATKINSON = FC_EFF_TYPES[1]
DIESEL = FC_EFF_TYPES[2]
H2FC = FC_EFF_TYPES[3]
HD_DIESEL = FC_EFF_TYPES[4]

# Model parameter data validation
# tuple of (type, (min, max) or [acceptable options], is_optional_bool)
# is_optional_bool=True will ensure the parameter is not nan 
POSITIVE_FLOAT_REQ = (float, (0, np.inf), False)
FRACTION_REQ = (float, (0, 1), False)
DATA_VALIDATION_DICT = {
    "vehPtType": (str, VEH_PT_TYPES, False),
    "dragCoef": POSITIVE_FLOAT_REQ,
    "frontalAreaM2": POSITIVE_FLOAT_REQ,
    "gliderKg": POSITIVE_FLOAT_REQ,
    "vehCgM": (float, (-np.inf, np.inf), False),
    "driveAxleWeightFrac": POSITIVE_FLOAT_REQ,
    "wheelBaseM": POSITIVE_FLOAT_REQ,
    "cargoKg": POSITIVE_FLOAT_REQ,
    "vehOverrideKg": (float, (0, np.inf), True),
    "compMassMultiplier": POSITIVE_FLOAT_REQ,
    "maxFuelStorKw": POSITIVE_FLOAT_REQ,
    "fuelStorSecsToPeakPwr": POSITIVE_FLOAT_REQ,
    "fuelStorKwh": POSITIVE_FLOAT_REQ,
    "fuelStorKwhPerKg": POSITIVE_FLOAT_REQ,
    "maxFuelConvKw": POSITIVE_FLOAT_REQ,
    "fcPwrOutPerc": (np.ndarray, (0, 1), True),
    "fcEffMap": (np.ndarray, (0, 1), True),
    "fcEffType": (str, FC_EFF_TYPES, False),
        # This could be optional=True, but would allow bad input for H2FC
    "fuelConvSecsToPeakPwr": POSITIVE_FLOAT_REQ,
    "fuelConvBaseKg": POSITIVE_FLOAT_REQ,
    "fuelConvKwPerKg": POSITIVE_FLOAT_REQ,
    "minFcTimeOn": POSITIVE_FLOAT_REQ,
    "idleFcKw": POSITIVE_FLOAT_REQ,
    "maxMotorKw": POSITIVE_FLOAT_REQ,
    "mcPwrOutPerc": (np.ndarray, (0, 1), True),
    "mcEffMap": (np.ndarray, (0, 1), True),
    "motorSecsToPeakPwr": POSITIVE_FLOAT_REQ,
    "mcPeKgPerKw": POSITIVE_FLOAT_REQ,
    "mcPeBaseKg": POSITIVE_FLOAT_REQ,
    "maxEssKw": POSITIVE_FLOAT_REQ,
    "maxEssKwh": POSITIVE_FLOAT_REQ,
    "essKgPerKwh": POSITIVE_FLOAT_REQ,
    "essBaseKg": POSITIVE_FLOAT_REQ,
    "essRoundTripEff": FRACTION_REQ,
    "essLifeCoefA": (float, (-np.inf, np.inf), False),
    "essLifeCoefB": (float, (-np.inf, np.inf), False),
    "minSoc": FRACTION_REQ,
    "maxSoc": FRACTION_REQ,
    "essDischgToFcMaxEffPerc": FRACTION_REQ,
    "essChgToFcMaxEffPerc": FRACTION_REQ,
    "wheelInertiaKgM2": POSITIVE_FLOAT_REQ,
    "numWheels": (int, (0, np.inf), False),
    "wheelRrCoef": POSITIVE_FLOAT_REQ,
    "wheelRadiusM": POSITIVE_FLOAT_REQ,
    "wheelCoefOfFric": POSITIVE_FLOAT_REQ,
    "maxAccelBufferMph": POSITIVE_FLOAT_REQ,
    "maxAccelBufferPercOfUseableSoc": FRACTION_REQ,
    "percHighAccBuf": FRACTION_REQ,
    "mphFcOn": POSITIVE_FLOAT_REQ,
    "kwDemandFcOn": POSITIVE_FLOAT_REQ,
    "maxRegen": FRACTION_REQ,
    "altEff": FRACTION_REQ,
    "chgEff": FRACTION_REQ,
    "auxKw": POSITIVE_FLOAT_REQ,
    "transKg": POSITIVE_FLOAT_REQ,
    "transEff": FRACTION_REQ,
    "essToFuelOkError": POSITIVE_FLOAT_REQ,
}

class Vehicle(object):
    """Class for loading and contaning vehicle attributes"""

    def __init__(self, vnum_or_file=None, veh_file=None, veh_dict=None, verbose=True, **kwargs):
        """Arguments:
        vnum_or_file: if provided as dict, gets treated as `veh_dict`.  Otherwise,
            default `load_veh` behavior.  
        veh_dict: If provided, vehicle is instantiated from dictionary, which must 
            contain a fully instantiated parameter set.
        verbose: print information during vehicle loading
        
        See below for `load_veh` method documentaion.\n""" + self.load_veh.__doc__
        
        self.props = params.PhysicalProperties() 
        self.fcPercOutArray = params.fcPercOutArray
        self.mcPercOutArray = params.mcPercOutArray

        if veh_dict or (type(vnum_or_file) == dict):
            for key, val in veh_dict.items():
                try:
                    self.__setattr__(key, val)
                except AttributeError as err_msg: 
                    # exceptions for properties that can't be set.  
                    if str(err_msg) != "can't set attribute":
                        raise AttributeError
        else:
            self.load_veh(vnum_or_file=vnum_or_file, veh_file=veh_file, verbose=verbose, **kwargs)

    def load_veh(self, vnum_or_file=None, veh_file=None, return_vehdf=False, verbose=True, **kwargs):
        """Load vehicle parameters from file.

        Arguments:
        ---------
        vnum_or_file: row number (int) of vehicle to simulate in 'FASTSim_py_veh_db.csv' 
            or supplied `veh_file` path (str or pathlib.Path) containing rows of vehicle 
            specs.  If only filename is passed, vehicle is assumed to be in resources/vehdb
        veh_file: path (str or pathlib.Path) to vehicle file if vnum_or_file is also 
            supplied as an int
        return_vehdf: (Boolean) if True, returns vehdf.  Useful for debugging purpsoses.   

        If default values are modified after loading, you may need to 
        rerun set_init_calcs() and set_veh_mass() to propagate changes."""

        def get_single_vehicle_df(veh_file):
            vehdf = pd.read_csv(veh_file)
            vehdf = vehdf.transpose()
            vehdf.columns = vehdf.iloc[1]
            vehdf.drop(vehdf.index[0], inplace=True)
            vehdf['Selection'] = np.nan * np.ones(vehdf.shape[0])
            vehdf.loc['Param Value', 'Selection'] = 0
            return vehdf

        if (type(vnum_or_file) in [type(Path()), str]):
            # if a file path is passed
            if not str(vnum_or_file).endswith('.csv'):
                vnum_or_file = Path(str(vnum_or_file) + '.csv')
            if (Path(vnum_or_file).name == str(Path(vnum_or_file))) and not (Path().resolve() / vnum_or_file).exists():
                vnum_or_file = THIS_DIR / 'resources/vehdb' / vnum_or_file
                if verbose: print(f'Loading vehicle from\n{Path(vnum_or_file).resolve()}')
                # if only filename is provided and not in local dir, assume in resources/vehdb
            veh_file = vnum_or_file
            vehdf = get_single_vehicle_df(veh_file)
            vnum = 0
        elif vnum_or_file and str(int(vnum_or_file)).isnumeric() and veh_file:
            # if a numeric is passed along with veh_file
            vnum = vnum_or_file
            vehdf = pd.read_csv(veh_file)
        elif vnum_or_file and str(int(vnum_or_file)).isnumeric():
            # if a numeric is passed alone
            vnum = vnum_or_file
            veh_file = DEFAULT_VEH_DB
            vehdf = DEFAULT_VEHDF
        elif veh_file:
            vehdf = get_single_vehicle_df(veh_file)
            vnum = 0
        else:
            raise Exception('load_veh requires `vnum_or_file` and/or `veh_file`.')
        vehdf.set_index('Selection', inplace=True, drop=False)

        # verify that only allowed columns have been provided
        for col in vehdf.columns:
            assert col in list(TEMPLATE_VEHDF.columns) + OPT_INIT_PARAMS, f"`{col}` is deprecated and must be removed from {veh_file}."

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
            elif re.search('(?i)true', data) is not None:
                data = True
            # replace string for FALSE with Boolean False
            elif re.search('(?i)false', data) is not None:
                data = False
            else:
                try:
                    data = float(data)
                except:
                    pass

            return data

        vehdf.loc[vnum] = vehdf.loc[vnum].apply(clean_data)

        # set columns and values as instance attributes and values
        for col in vehdf.columns:
            col1 = col.replace(' ', '_')
            # assign dataframe columns to vehicle
            self.__setattr__(col1, vehdf.loc[vnum, col])
        if not hasattr(self, "fcPeakEffOverride"):
            self.fcPeakEffOverride = -1
        if not hasattr(self, "mcPeakEffOverride"):
            self.mcPeakEffOverride = -1

        # make sure all the attributes needed by CycleJit are set
        # this could potentially cause unexpected behaviors
        missing_cols = set(TEMPLATE_VEHDF.columns) - set(vehdf.columns)
        if len(missing_cols) > 0:
            if verbose:
                print(f"np.nan filled in for {list(missing_cols)} missing from '{str(veh_file)}'.")
                print(f"Turn this warning off by passing `verbose=False` when instantiating {type(self)}.")
            for col in missing_cols:
                self.__setattr__(col, np.nan)

        veh_dict = dict(vehdf.loc[vnum, :])
        # Power and efficiency arrays are defined in parameters.py
        # Can also be input in CSV as array under column fcEffMap of form
        # [0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30]
        # no quotes necessary
        self.fcEffType = str(self.fcEffType)

        try:
            # check if optional parameter fcEffMap is provided in vehicle csv file
            self.fcEffMap = np.array(ast.literal_eval(veh_dict['fcEffMap']))
            if verbose:
                print(f"fcEffMap is overriding fcEffType in {veh_file}")
        
        except:
            warn_str = f"""fcEffType {self.fcEffType} is not in {FC_EFF_TYPES},
            and `fcEffMap` is not provided."""
            assert self.fcEffType in FC_EFF_TYPES, warn_str

            if self.fcEffType == SI:  # SI engine
                self.fcEffMap = params.fcEffMap_si

            elif self.fcEffType == ATKINSON:  # Atkinson cycle SI engine -- greater expansion
                self.fcEffMap = params.fcEffMap_atk

            elif self.fcEffType == DIESEL:  # Diesel (compression ignition) engine
                self.fcEffMap = params.fcEffMap_diesel

            elif self.fcEffType == H2FC:  # H2 fuel cell
                self.fcEffMap = params.fcEffMap_fuel_cell

            elif self.fcEffType == HD_DIESEL:  # heavy duty Diesel engine
                self.fcEffMap = params.fcEffMap_hd_diesel

        try:
            # check if optional parameter fcPwrOutPerc is provided in vehicle csv file
            self.fcPwrOutPerc = np.array(ast.literal_eval(veh_dict['fcPwrOutPerc']))
        except:
            self.fcPwrOutPerc = params.fcPwrOutPerc

        fc_eff_len_err = f'len(fcPwrOutPerc) ({len(self.fcPwrOutPerc)}) is not' +\
            f'equal to len(fcEffMap) ({len(self.fcEffMap)})'
        assert len(self.fcPwrOutPerc) == len(self.fcEffMap), fc_eff_len_err

        ### Defining MC efficiency curve as lookup table for %power_in vs power_out
        ### see "Motor" tab in FASTSim for Excel

        # Power and efficiency arrays are defined in parameters.py
        # can also be overridden by motor power and efficiency columns in the input file
        # ensure that the column existed and the value in the cell wasn't empty (becomes NaN)
        try:
            # check if mcPwrOutPerc is provided in vehicle csv file
            self.mcPwrOutPerc = np.array(ast.literal_eval(veh_dict['mcPwrOutPerc']))
        except:
            self.mcPwrOutPerc = params.mcPwrOutPerc

        try:
            # check if mcEffMap is provided in vehicle csv file
            self.mcEffMap = np.array(ast.literal_eval(veh_dict['mcEffMap']))
        except:
            self.mcEffMap = None

        self.largeBaselineEff = kwargs.pop('largeBaselineEff', params.large_baseline_eff)
        self.smallBaselineEff = params.small_baseline_eff

        mc_large_eff_len_err = f'len(mcPwrOutPerc) ({len(self.mcPwrOutPerc)}) is not' +\
            f'equal to len(largeBaselineEff) ({len(self.largeBaselineEff)})'
        assert len(self.mcPwrOutPerc) == len(self.largeBaselineEff), mc_large_eff_len_err
        mc_small_eff_len_err = f'len(mcPwrOutPerc) ({len(self.mcPwrOutPerc)}) is not' +\
            f'equal to len(smallBaselineEff) ({len(self.smallBaselineEff)})'
        assert len(self.mcPwrOutPerc) == len(self.smallBaselineEff), mc_small_eff_len_err

        # set stopStart if not provided
        if 'stopStart' in self.__dir__() and np.isnan(self.__getattribute__('stopStart')):
            self.stopStart = False

        self.smallMotorPowerKw = 7.5 # default (float)
        self.largeMotorPowerKw = 75.0 # default (float)

        # check if vehYear provided in file, and, if not, provide value from Scenario_name or default of 0
        if ('vehYear' not in veh_dict) or np.isnan(self.vehYear):
            # regex is for vehicle model year if Scenario_name starts with any 4 digit string
            if re.match('\d{4}', str(self.Scenario_name)):
                self.vehYear = np.int32(
                    re.match('\d{4}', str(self.Scenario_name)).group()
                )
            else:
                self.vehYear = np.int32(0) # set 0 as default to get correct type
        
        # in case vehYear gets loaded from file as float
        self.vehYear = np.int32(self.vehYear)
        
        self.set_init_calcs()
        self.set_veh_mass()

        # Parameter data validation
        for key, val_info in DATA_VALIDATION_DICT.items():
            value = self.__getattribute__(key)
            target_type = val_info[0]
        
            # If value is wrong type
            if not isinstance(value, target_type):
                # If value is float and would be converted to int
                if isinstance(value, (np.floating, float)) and (target_type is int):
                    assert value.is_integer(), \
                        f"Non-integer value of {key} '{value}' would be converted to {int(value)}"
                try:
                    # Try to convert type
                    value = target_type(value)
                    self.__setattr__(key, value)
                except ValueError:
                    # Continue and fail assert
                    pass
            assert isinstance(value, target_type), \
                f"Value of {key} '{value}' {type(value)} not of acceptable type {target_type}"

            # Check value
            if isinstance(value, str):
                # Check that string is in given options
                value_options = val_info[1]
                assert value in value_options, \
                    f"Value of {key} '{value}' not in acceptable options {value_options}"
            elif isinstance(value, np.ndarray):
                # Check that each array element is within given range
                value_min = val_info[1][0]
                value_max = val_info[1][1]
                optional = val_info[2]
                if np.isnan(value).any():
                    assert optional, f"Elements of {key} {value} must not be nan"
                assert np.all(value_min <= value) and np.all(value <= value_max), \
                    f"Element(s) of {key} {value} outside acceptable range [{value_min}, {value_max}]"
            elif isinstance(value, numbers.Number) and not isinstance(value, bool):
                # Check that numeric value is within given range
                value_min = val_info[1][0]
                value_max = val_info[1][1]
                optional = val_info[2]
                if np.isnan(value):
                    assert optional, f"Value of {key} must not be nan"
                else:
                    assert (value_min <= value) and (value <= value_max), \
                        f"Value of {key} '{value}' outside acceptable range [{value_min}, {value_max}]"

        if return_vehdf:
            return vehdf

    def get_numba_veh(self):
        """Load numba JIT-compiled vehicle."""
        from .vehiclejit import VehicleJit, veh_spec
        if 'numba_veh' not in self.__dict__:
            numba_veh = VehicleJit()
        for item in veh_spec:
            if (type(self.__getattribute__(item[0])) in [np.ndarray, np.float64]):
                numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.float64))
            elif type(self.__getattribute__(item[0])) == np.int64:
                numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.int32))
            elif item[0] == 'props':
                numba_veh.__setattr__(item[0], params.PhysicalPropertiesJit())
            else:
                numba_veh.__setattr__(
                    item[0], self.__getattribute__(item[0]))
            
        return numba_veh
    
    def set_init_calcs(self):
        """
        Legacy method for calling set_dependents.
        """

        self.set_dependents()

    def set_dependents(self):
        """
        Sets derived parameters.
        """
        
        if self.Scenario_name != 'Template Vehicle for setting up data types':
            if self.vehPtType == BEV:
                assert self.maxFuelStorKw == 0, 'maxFuelStorKw must be zero for provided BEV powertrain type'
                assert self.fuelStorKwh  == 0, 'fuelStorKwh must be zero for provided BEV powertrain type'
                assert self.maxFuelConvKw == 0, 'maxFuelConvKw must be zero for provided BEV powertrain type'
            elif (self.vehPtType == CONV) and not(self.stopStart):
                assert self.maxMotorKw == 0, 'maxMotorKw must be zero for provided Conv powertrain type'
                assert self.maxEssKw == 0, 'maxEssKw must be zero for provided Conv powertrain type'
                assert self.maxEssKwh == 0, 'maxEssKwh must be zero for provided Conv powertrain type'

        ### Build roadway power lookup table
        self.MaxRoadwayChgKw = np.zeros(6)
        self.chargingOn = False

        # Checking if a vehicle has any hybrid components
        if (self.maxEssKwh == 0) or (self.maxEssKw == 0) or (self.maxMotorKw == 0):
            self.noElecSys = True
        else:
            self.noElecSys = False

        # Checking if aux loads go through an alternator
        if (self.noElecSys == True) or (self.maxMotorKw <= self.auxKw) or (self.forceAuxOnFC == True):
            self.noElecAux = True
        else:
            self.noElecAux = False

        # discrete array of possible engine power outputs
        self.inputKwOutArray = self.fcPwrOutPerc * self.maxFuelConvKw
        # Relatively continuous array of possible engine power outputs
        self.fcKwOutArray = self.maxFuelConvKw * self.fcPercOutArray
        # Creates relatively continuous array for fcEff
        self.fcEffArray = np.interp(x=self.fcPercOutArray, xp=self.fcPwrOutPerc, fp=self.fcEffMap)

        self.modernMax = params.modern_max            
        
        modern_diff = self.modernMax - max(self.largeBaselineEff)

        large_baseline_eff_adj = self.largeBaselineEff + modern_diff

        mcKwAdjPerc = max(
            0.0, 
            min(
                (self.maxMotorKw - self.smallMotorPowerKw)/(self.largeMotorPowerKw - self.smallMotorPowerKw), 
                1.0)
            )

        if self.mcEffMap is None:
            self.mcEffArray = mcKwAdjPerc * large_baseline_eff_adj + \
                    (1 - mcKwAdjPerc) * self.smallBaselineEff
            self.mcEffMap = self.mcEffArray
        else:
            self.mcEffArray = self.mcEffMap

        mcKwOutArray = np.linspace(0, 1, len(self.mcPercOutArray)) * self.maxMotorKw

        mcFullEffArray = np.interp(
            x=self.mcPercOutArray, xp=self.mcPwrOutPerc, fp=self.mcEffArray)
        mcFullEffArray[0] = 0
        mcFullEffArray[-1] = self.mcEffArray[-1]

        mcKwInArray = np.concatenate(
            (np.zeros(1, dtype=np.float64), mcKwOutArray[1:] / mcFullEffArray[1:]))
        mcKwInArray[0] = 0

        self.mcKwInArray = mcKwInArray
        self.mcKwOutArray = mcKwOutArray
        self.mcMaxElecInKw = max(mcKwInArray)
        self.mcFullEffArray = mcFullEffArray

        self.mcMaxElecInKw = max(self.mcKwInArray)

        ### Specify shape of mc regen efficiency curve
        ### see "Regen" tab in FASTSim for Excel
        self.regenA = 500.0  # hardcoded
        self.regenB = 0.99  # hardcoded

        if self.fcPeakEffOverride != -1:
            self.fcPeakEff = self.fcPeakEffOverride
            print("fcPeakEffOverride is modifying efficiency curve.")
        if self.mcPeakEffOverride != -1:
            self.mcPeakEff = self.mcPeakEffOverride
            print("mcPeakEffOverride is modifying efficiency curve.")

        # check that efficiencies are not violating the first law of thermo
        assert self.fcEffArray.min() >= 0, f"min MC eff < 0 is not allowed"
        assert self.fcPeakEff < 1, f"fcPeakEff >= 1 is not allowed."
        assert self.mcFullEffArray.min() >= 0, f"min MC eff < 0 is not allowed"
        assert self.mcPeakEff < 1, f"mcPeakEff >= 1 is not allowed."

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
                fc_mass_kg = (1 / self.fuelConvKwPerKg * self.maxFuelConvKw +
                    self.fuelConvBaseKg) * self.compMassMultiplier
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

        self.maxTracMps2 = (
            self.wheelCoefOfFric * self.driveAxleWeightFrac * self.vehKg * self.props.gravityMPerSec2 /
            (1 + self.vehCgM * self.wheelCoefOfFric / self.wheelBaseM)
            ) / (self.vehKg * self.props.gravityMPerSec2)  * self.props.gravityMPerSec2

        # copying to instance attributes
        self.essMassKg = np.float64(ess_mass_kg)
        self.mcMassKg =  np.float64(mc_mass_kg)
        self.fcMassKg =  np.float64(fc_mass_kg)
        self.fsMassKg =  np.float64(fs_mass_kg)

    # properties -- these were created to make sure modifications to curves propagate

    @property
    def maxFcEffKw(self): return self.fcKwOutArray[np.argmax(self.fcEffArray)]
    @property
    def fcMaxOutkW(self): return np.max(self.inputKwOutArray)
    @property
    def maxRegenKwh(self): return 0.5 * self.vehKg * (27**2) / (3600 * 1000)    

    @property
    def vehTypeSelection(self): 
        """
        Copying vehPtType to additional key
        to be consistent with Excel version but not used in Python version
        """
        return self.vehPtType

    def get_mcPeakEff(self):
        "Return `np.max(self.mcEffArray)`"
        assert np.max(self.mcFullEffArray) == np.max(self.mcEffArray)
        return np.max(self.mcFullEffArray)

    def set_mcPeakEff(self, new_peak):
        """
        Set motor peak efficiency EVERWHERE.  
        
        Arguments:
        ----------
        new_peak: float, new peak motor efficiency in decimal form 
        """
        self.mcEffArray *= new_peak / self.mcEffArray.max()
        self.mcFullEffArray *= new_peak / self.mcFullEffArray.max()

    mcPeakEff = property(get_mcPeakEff, set_mcPeakEff)

    def get_fcPeakEff(self):
        "Return `np.max(self.fcEffArray)`"
        return np.max(self.fcEffArray)

    def set_fcPeakEff(self, new_peak):
        """
        Set fc peak efficiency EVERWHERE.  
        
        Arguments:
        ----------
        new_peak: float, new peak fc efficiency in decimal form 
        """
        self.fcEffArray *= new_peak / self.fcEffArray.max()
        self.fcEffMap *= new_peak / self.fcEffArray.max()

    fcPeakEff = property(get_fcPeakEff, set_fcPeakEff)


def copy_vehicle(veh:Vehicle, return_dict=False, use_jit=None) -> Vehicle:
    """
    Returns copy of Vehicle or VehicleJit.
    Arguments:
    veh: instantiated Vehicle or VehicleJit
    return_dict: (Boolean) if True, returns vehicle as dict. 
        Otherwise, returns exact copy.
    use_jit: (Boolean)
        default -- infer from arg
        True -- use numba
        False -- don't use numba
    """

    veh_dict = {}

    from . import vehiclejit
    for keytup in vehiclejit.veh_spec:
        key = keytup[0]
        if key != 'props':         
            veh_dict[key] = deepcopy(veh.__getattribute__(key))

    if return_dict:
        return veh_dict
        
    if use_jit is None:
        use_jit = "Jit" in str(type(veh))

    veh = Vehicle(veh_dict=veh_dict)
    if use_jit:
        veh = veh.get_numba_veh()

    return veh  

def veh_equal(veh1, veh2, full_out=False):
    """Given veh1 and veh2, which can be Vehicle and/or VehicleJit
    instances, return True if equal.
    
    Arguments:
    ----------
    """

    veh_dict1 = copy_vehicle(veh1, True)
    veh_dict2 = copy_vehicle(veh2, True)
    err_list = []
    keys = list(veh_dict1.keys())
    for key in keys:
        if pd.api.types.is_list_like(veh_dict1[key]):
            if (veh_dict1[key] != veh_dict2[key]).any():
                if not full_out: return False
                err_list.append(
                    {'key': key, 'val1': veh_dict1[key], 'val2': veh_dict2[key]})
        elif veh_dict1[key] != veh_dict2[key]:
            try:
                if np.isnan(veh_dict1[key]) and np.isnan(veh_dict2[key]):
                    continue # treat as equal if both nan
            except:
                pass
            if not full_out: return False
            err_list.append(
                {'key': key, 'val1': veh_dict1[key], 'val2': veh_dict2[key]})
    if full_out: return err_list

    return True
