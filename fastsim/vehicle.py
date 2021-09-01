"""Module containing classes and methods for for loading vehicle and
cycle data. For example usage, see ../README.md"""

### Import necessary python modules
import numpy as np
import pandas as pd
import types as pytypes
import re
from numba.experimental import jitclass                 # import the decorator
import warnings
warnings.simplefilter('ignore')
from pathlib import Path
import ast

# local modules
from . import parameters as params
from .buildspec import build_spec


THIS_DIR = Path(__file__).parent
DEFAULT_VEH_DB = THIS_DIR / 'resources' / 'FASTSim_py_veh_db.csv'
DEFAULT_VEHDF = pd.read_csv(DEFAULT_VEH_DB)


class Vehicle(object):
    """Class for loading and contaning vehicle attributes"""

    def __init__(self, vnum_or_file=None, veh_file=None, veh_dict=None, verbose=True):
        """Arguments:
        vnum_or_file: if provided as dict, gets treated as `veh_dict`.  Otherwise,
            default `load_veh` behavior.  
        veh_dict: If provided, vehicle is instantiated from dictionary, which must 
            contain a fully instantiated parameter set.
        
        See below for `load_veh` method documentaion.\n""" + self.load_veh.__doc__
        
        self.props = params.PhysicalProperties()
        self.fcPercOutArray = params.fcPercOutArray
        self.mcPercOutArray = params.mcPercOutArray

        if veh_dict or (type(vnum_or_file) == dict):
            for key, val in veh_dict.items():
                self.__setattr__(key, val)
        else:
            self.load_veh(vnum_or_file=vnum_or_file, veh_file=veh_file)

    def load_veh(self, vnum_or_file=None, veh_file=None, return_vehdf=False, verbose=True):
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

        if (type(vnum_or_file) in [type(Path()), str]):
            # if a file path is passed
            if Path(vnum_or_file).name == str((Path(vnum_or_file))):
                # if only filename is provided assume in resources/vehdb
                vnum_or_file = THIS_DIR / 'resources/vehdb' / vnum_or_file
            vehdf = pd.read_csv(Path(vnum_or_file))
            vehdf = vehdf.transpose()
            vehdf.columns = vehdf.iloc[1]
            vehdf.drop(vehdf.index[0], inplace=True)
            vehdf['Selection'] = np.nan * np.ones(vehdf.shape[0])
            vehdf.loc['Param Value', 'Selection'] = 0
            vnum = 0
        elif str(vnum_or_file).isnumeric() and veh_file:
            # if a numeric is passed along with veh_file
            vnum = vnum_or_file
            vehdf = pd.read_csv(Path(veh_file))
        elif str(vnum_or_file).isnumeric():
            # if a numeric is passed alone
            vnum = vnum_or_file
            vehdf = DEFAULT_VEHDF
        else:
            raise Exception('load_veh requires `vnum_or_file` and/or `veh_file`.')
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
            # assign dataframe columns
            self.__setattr__(col1, vehdf.loc[vnum, col])

        # make sure all the attributes needed by CycleJit are set
        # this could potentially cause unexpected behaviors
        missing_cols = set(DEFAULT_VEHDF.columns) - set(vehdf.columns)
        if len(missing_cols) > 0:
            if verbose:
                print(f"np.nan filled in for {list(missing_cols)} missing from " +
                      "'" + str(veh_file) + "'")
            for col in missing_cols:
                self.__setattr__(col, np.nan)

        # Power and efficiency arrays are defined in parameters.py
        # Can also be input in CSV as array under column fcEffMap of form
        # [0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30]
        # no quotes necessary
        try:
            # check if fcEffMap is provided in vehicle csv file
            self.fcEffMap = np.array(ast.literal_eval(vehdf['fcEffMap']))
        
        except:
            if self.fcEffType == 1:  # SI engine
                self.fcEffMap = params.fcEffMap_si + self.fcAbsEffImpr

            elif self.fcEffType == 2:  # Atkinson cycle SI engine -- greater expansion
                self.fcEffMap = params.fcEffMap_atk + self.fcAbsEffImpr

            elif self.fcEffType == 3:  # Diesel (compression ignition) engine
                self.fcEffMap = params.fcEffMap_diesel + self.fcAbsEffImpr

            elif self.fcEffType == 4:  # H2 fuel cell
                self.fcEffMap = params.fcEffMap_fuel_cell + self.fcAbsEffImpr

            elif self.fcEffType == 5:  # heavy duty Diesel engine
                self.fcEffMap = params.fcEffMap_hd_diesel + self.fcAbsEffImpr

        try:
            # check if fcPwrOutPerc is provided in vehicle csv file
            self.fcPwrOutPerc = np.array(ast.literal_eval(vehdf['fcPwrOutPerc']))
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
            self.mcPwrOutPerc = np.array(ast.literal_eval(vehdf['mcPwrOutPerc']))
        except:
            self.mcPwrOutPerc = params.mcPwrOutPerc

        try:
            self.largeBaselineEff = np.array(
                ast.literal_eval(vehdf['largeBaselineEff']))
        except:
            self.largeBaselineEff = params.large_baseline_eff

        try:
            self.smallBaselineEff = np.array(ast.literal_eval(vehdf['smallBaselineEff']))
        except:
            self.smallBaselineEff = params.small_baseline_eff

        mc_large_eff_len_err = f'len(mcPwrOutPerc) ({len(self.mcPwrOutPerc)}) is not' +\
            f'equal to len(largeBaselineEff) ({len(self.largeBaselineEff)})'
        assert len(self.mcPwrOutPerc) == len(
            self.largeBaselineEff), mc_large_eff_len_err
        mc_small_eff_len_err = f'len(mcPwrOutPerc) ({len(self.mcPwrOutPerc)}) is not' +\
            f'equal to len(smallBaselineEff) ({len(self.smallBaselineEff)})'
        assert len(self.mcPwrOutPerc) == len(
            self.smallBaselineEff), mc_small_eff_len_err

        # set stopStart if not provided
        if 'stopStart' in self.__dir__() and np.isnan(self.__getattribute__('stopStart')):
            self.stopStart = False

        # assign motor efficiency params
        if ('smallMotorPowerKw' not in vehdf.columns) or np.isnan(self.smallMotorPowerKw):
            self.smallMotorPowerKw = 7.5 # default (float)
        if ('largMotorPowerKw' not in vehdf.columns) or np.isnan(self.largMotorPowerKw):
            self.largeMotorPowerKw = 75.0 # default (float)

        # assigning vehYear if not provided
        if ('vehYear' not in vehdf.columns) or np.isnan(self.vehYear):
            # re is for vehicle model year if Scenario_name starts with any 4 digit string
            if re.match('\d{4}', self.Scenario_name):
                self.vehYear = np.int32(
                    re.match('\d{4}', self.Scenario_name).group()
                )
            else:
                self.vehYear = np.int32(0000)
        
        # in case vehYear gets loaded from file as float
        self.vehYear = np.int32(self.vehYear)
            
        self.set_init_calcs()
        self.set_veh_mass()

        if return_vehdf:
            return vehdf

    def get_numba_veh(self):
        """Load numba JIT-compiled vehicle."""
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
        """Legacy method for calling set_dependents."""

        self.set_dependents()

    def set_dependents(self):
        """Sets derived parameters."""
        
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

        # Copying vehPtType to additional key
        self.vehTypeSelection = self.vehPtType
        # to be consistent with Excel version but not used in Python version

        # discrete array of possible engine power outputs
        self.inputKwOutArray = self.fcPwrOutPerc * self.maxFuelConvKw
        # Relatively continuous array of possible engine power outputs
        self.fcKwOutArray = self.maxFuelConvKw * self.fcPercOutArray
        # Creates relatively continuous array for fcEff
        self.fcEffArray = np.interp(x=self.fcPercOutArray, xp=self.fcPwrOutPerc, fp=self.fcEffMap)

        self.maxFcEffKw = self.fcKwOutArray[np.argmax(self.fcEffArray)]
        self.fcMaxOutkW = np.max(self.inputKwOutArray)
            
        if np.isnan(self.modernMax):
            self.modernMax = params.modern_max            
        
        modern_diff = self.modernMax - max(self.largeBaselineEff)

        large_baseline_eff_adj = self.largeBaselineEff + modern_diff

        mcKwAdjPerc = max(
            0.0, 
            min(
                (self.maxMotorKw - self.smallMotorPowerKw)/(self.largeMotorPowerKw - self.smallMotorPowerKw), 
                1.0)
            )

        self.mcEffArray = mcKwAdjPerc * large_baseline_eff_adj + \
                (1 - mcKwAdjPerc) * self.smallBaselineEff

        mcInputKwOutArray = self.mcPwrOutPerc * self.maxMotorKw
        mcKwOutArray = np.linspace(0, 1, len(self.mcPercOutArray)) * self.maxMotorKw

        mcFullEffArray = np.interp(
            x=self.mcPercOutArray, xp=self.mcPwrOutPerc, fp=self.mcEffArray)
        mcFullEffArray[0] = 0.0 #

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

        self.maxTracMps2 = (
            self.wheelCoefOfFric * self.driveAxleWeightFrac * self.vehKg * self.props.gravityMPerSec2 /
            (1 + self.vehCgM * self.wheelCoefOfFric / self.wheelBaseM)
            ) / (self.vehKg * self.props.gravityMPerSec2)  * self.props.gravityMPerSec2
        self.maxRegenKwh = 0.5 * self.vehKg * (27**2) / (3600 * 1000)

        # copying to instance attributes
        self.essMassKg = np.float64(ess_mass_kg)
        self.mcMassKg =  np.float64(mc_mass_kg)
        self.fcMassKg =  np.float64(fc_mass_kg)
        self.fsMassKg =  np.float64(fs_mass_kg)

veh_spec = build_spec(Vehicle('template.csv'))


@jitclass(veh_spec)
class VehicleJit(Vehicle):
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

        self.props = params.PhysicalPropertiesJit()

    def get_numba_veh(self):
        """Overrides parent class (Cycle) with dummy method 
        to avoid numba incompatibilities."""
        print(self.get_numba_veh.__doc__)

    def load_veh(self):
        """Overrides parent class (Cycle) with dummy method 
        to avoid numba incompatibilities."""
        print(self.load_veh.__doc__)

    def set_init_calcs(self):
        """Overrides parent class (Cycle) with dummy method 
        to avoid numba incompatibilities.
        Runs self.set_dependents()"""
        self.set_dependents()


def copy_vehicle(veh, return_dict=False):
    """Returns copy of Vehicle or VehicleJit.
    Arguments:
    veh: instantiated Vehicle or VehicleJit
    return_dict: (Boolean) if True, returns vehicle as dict. 
        Otherwise, returns exact copy.
    """

    veh_dict = {}

    for key in veh.__dir__():
        if (not key.startswith('_')) and \
                (type(veh.__getattribute__(key)) != pytypes.MethodType):
            veh_dict[key] = veh.__getattribute__(key)

    if return_dict:
        return veh_dict
    else:
        if type(veh) == Vehicle:
            veh = Vehicle(veh_dict=veh_dict)
        else:
            # expects `veh` to be instantiated VehicleJit
            veh = Vehicle(veh_dict=veh_dict).get_numba_veh()
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
    keys.remove('props') # no need to compare this one
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
