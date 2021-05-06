"""Module containing classes and methods for for loading vehicle and
cycle data. For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
import pandas as pd
import re
import sys
from numba.experimental import jitclass                 # import the decorator
from numba import float64, int32, bool_, types    # import the types
import warnings
warnings.simplefilter('ignore')
from pathlib import Path
import ast

# local modules
from . import parameters as params
from .buildspec import build_spec


THIS_DIR = Path(__file__).parent
DEFAULT_VEH_DB = Path(THIS_DIR) / 'resources' / 'FASTSim_py_veh_db.csv'
DEFAULT_VEHDF = pd.read_csv(DEFAULT_VEH_DB)


class Vehicle(object):
    """Class for loading and contaning vehicle attributes"""

    def __init__(self, vnum=None, veh_file=None, verbose=True):
        """See doc string for load_veh for additional details on arguments.
    
        If a single vehicle `veh_file` is provided, vnum cannot be passed, and
        veh_file must be passed as a keyword argument. Files contained in
        fastsim/resources/vehdb can be loaded with the filename if provided as
        the `vnum` argument.  Specifying `veh_file` will explicitly load whatever
        file path is provided, using `vnum` if appropriate."""
        
        self.props = params.PhysicalProperties()
        self.fcPwrOutPerc = params.fcPwrOutPerc
        self.fcPercOutArray = params.fcPercOutArray
        self.mcPercOutArray = params.mcPercOutArray

        if veh_file and vnum:
            self.load_veh(vnum, veh_file=veh_file, verbose=verbose)
        elif vnum and not veh_file:
            try:
                # load numbered vehicle
                int(vnum)
                self.load_veh(int(vnum), verbose=verbose)
            except:
                # load FASTSim's standalone vehicles 
                # (vnum is a filename (str or pathlib.Path) in this case)
                self.load_veh(veh_file=Path(THIS_DIR) / 'resources/vehdb' / vnum, verbose=verbose)

        else:
            # not passing `vnum` tells load_veh that the file contains only 1 vehicle
            self.load_veh(veh_file=veh_file, verbose=verbose)

    def get_numba_veh(self):
        """Load numba JIT-compiled vehicle."""
        if 'numba_veh' not in self.__dict__:
            self.numba_veh = VehicleJit()
        for item in veh_spec:
            if (type(self.__getattribute__(item[0])) == np.ndarray) | (type(self.__getattribute__(item[0])) == np.float64):
                self.numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.float64))
            elif type(self.__getattribute__(item[0])) == np.int64:
                self.numba_veh.__setattr__(item[0], self.__getattribute__(item[0]).astype(np.int32))
            elif item[0] == 'props':
                self.numba_veh.__setattr__(item[0], params.PhysicalPropertiesJit())
            else:
                self.numba_veh.__setattr__(
                    item[0], self.__getattribute__(item[0]))
            
        return self.numba_veh
    
    def load_veh(self, vnum=None, veh_file=None, return_vehdf=False, verbose=True):
        """Load vehicle parameters from file.

        Arguments:
        ---------
        vnum: row number (int) of vehicle to simulate in 'FASTSim_py_veh_db.csv'
        veh_file: path (str or pathlib.Path) to vehicle file 
        return_vehdf: (Boolean) if True, returns vehdf.  Mostly useful for 
            debugging purpsose.   

        If default values are modified after loading, you may need to 
        rerun set_init_calcs() and set_veh_mass() to propagate changes."""

        if vnum and veh_file:
            vehdf = pd.read_csv(Path(veh_file))
        elif vnum:
            vehdf = DEFAULT_VEHDF
        elif veh_file:
            vehdf = pd.read_csv(Path(veh_file))
            vehdf = vehdf.transpose()
            vehdf.columns = vehdf.iloc[1]
            vehdf.drop(vehdf.index[0], inplace=True)
            vehdf['Selection'] = np.nan * np.ones(vehdf.shape[0])
            vehdf.loc['Param Value', 'Selection'] = 0
            vnum = 0
        else:
            raise Exception('load_veh requires `vnum` and/or `veh_file`.')
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
                print("np.nan filled in for values missing from " + "'" + str(veh_file) + "'")
            for col in missing_cols:
                self.__setattr__(col, np.nan)

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
        ### Defining MC efficiency curve as lookup table for %power_in vs power_out
        ### see "Motor" tab in FASTSim for Excel

        # Power and efficiency arrays are defined in parameters.py
        # can also be overridden by motor power and efficiency columns in the input file
        # ensure that the column existed and the value in the cell wasn't empty (becomes NaN)
        try:
            self.mcPwrOutPerc = np.array(ast.literal_eval(self.mcPwrOutPerc))
        except ValueError:
            self.mcPwrOutPerc = params.mcPwrOutPerc


        self.set_init_calcs()
        self.set_veh_mass()            

        if return_vehdf:
            return vehdf

    def set_init_calcs(self):
        """Set parameters that can be calculated after loading vehicle data"""
        ### Defining Fuel Converter efficiency curve as lookup table for %power_in vs power_out
        ### see "FC Model" tab in FASTSim for Excel

        if len(self.fcEffMap) != 12:
            raise ValueError('fcEffMap has length of {}, but should have length of 12'.
                             format(len(self.fcEffMap)))

        if len(self.mcPwrOutPerc) != 11:
            raise ValueError('mcPwrOutPerc has length of {}, but should have length of 11'.
                             format(len(self.mcPwrOutPerc)))

        try:
            self.largeBaselineEff = np.array(
                ast.literal_eval(self.largeBaselineEff))
        except ValueError:
            self.largeBaselineEff = params.large_baseline_eff
        if len(self.largeBaselineEff) != 11:
            raise ValueError('largeBaselineEff has length of {}, but should have length of 11'.
                             format(len(self.largeBaselineEff)))

        try:
            self.smallBaselineEff = np.array(
                ast.literal_eval(self.smallBaselineEff))
        except ValueError:
            self.smallBaselineEff = params.small_baseline_eff

        if len(self.smallBaselineEff) != 11:
            raise ValueError('smallBaselineEff has length of {}, but should have length of 11'.
                format(len(self.smallBaselineEff)))

        if 'stopStart' in self.__dir__() and np.isnan(self.__getattribute__('stopStart')):
            self.stopStart = False

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

        mcKwAdjPerc = max(0.0, min((self.maxMotorKw - 7.5)/(75.0 - 7.5), 1.0))
        self.mcEffArray = np.zeros(len(self.mcPwrOutPerc))

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

        # for stats and interest
        self.essMassKg = ess_mass_kg
        self.mcMassKg =  mc_mass_kg
        self.fcMassKg =  fc_mass_kg
        self.fsMassKg =  fs_mass_kg

veh_spec = build_spec(Vehicle(1))


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
