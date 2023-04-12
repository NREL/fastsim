"""
Module containing classes and methods for for loading vehicle data.
"""

# Typing
from __future__ import annotations
from typing import Dict, Optional
from typing_extensions import Self

# Import necessary python modules
import numpy as np
from dataclasses import dataclass, field, InitVar
import pandas as pd
import types as pytypes
import re
from pathlib import Path
from copy import deepcopy
import ast
import copy

# local modules
from fastsim import parameters as params
from fastsim import utils
from fastsim.vehicle_base import keys_and_types, NEW_TO_OLD
from .rustext import RUST_AVAILABLE

if RUST_AVAILABLE:
    import fastsimrust as fsr
    from fastsimrust import RustVehicle

# Logging
import logging
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
DEFAULT_VEH_DB = THIS_DIR / 'resources' / 'FASTSim_py_veh_db.csv'
DEFAULT_VEHDF = pd.read_csv(DEFAULT_VEH_DB)
VEHICLE_DIR = THIS_DIR / 'resources' / 'vehdb'

KEYS_TO_REMOVE = [
    'large_baseline_eff',
    'small_baseline_eff',
    'large_motor_power_kw',
    'small_motor_power_kw',
    'charging_on',
    'no_elec_sys',
    'no_elec_aux',
    'max_roadway_chg_kw',
    'input_kw_out_array',
    'fc_kw_out_array',
    'fc_eff_array',
    'modern_max',
    'mc_eff_array',
    'mc_kw_in_array',
    'mc_kw_out_array',
    'mc_max_elec_in_kw',
    'mc_full_eff_array',
    'veh_kg',
    'max_trac_mps2',
    'ess_mass_kg',
    'mc_mass_kg',
    'fc_mass_kg',
    'fs_mass_kg',
    'fc_perc_out_array',
    'mc_perc_out_array',
    'converted_to_rust',
]

__doc__ += f"""To create a new vehicle model, copy \n`{(THIS_DIR / 'resources/vehdb/template.csv').resolve()}`
to a working directory not inside \n`{THIS_DIR.resolve()}`
and edit as appropriate.
"""


def get_template_df():
    vehdf = pd.read_csv(VEHICLE_DIR / 'template.csv')
    vehdf = vehdf.transpose()
    vehdf.columns = vehdf.iloc[1]
    vehdf.drop(vehdf.index[0], inplace=True)
    vehdf['selection'] = np.nan * np.ones(vehdf.shape[0])
    vehdf.loc['Param Value', 'selection'] = 0
    return vehdf


TEMPLATE_VEHDF = get_template_df()

# list of optional parameters that do not get assigned as vehicle attributes
OPT_INIT_PARAMS = ['fc_peak_eff_override', 'mc_peak_eff_override']

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

OLD_TO_NEW = {val: key for key, val in NEW_TO_OLD.items()}


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


@dataclass
class Vehicle(object):
    """
    Class for loading and contaning vehicle attributes
    See `from_vehdb`, `from_file`, and `from_dict` methods for usage instructions.
    """

    scenario_name: str
    selection: int
    veh_year: int
    veh_pt_type: str
    drag_coef: float
    frontal_area_m2: float
    glider_kg: float
    veh_cg_m: float
    drive_axle_weight_frac: float
    wheel_base_m: float
    cargo_kg: float
    veh_override_kg: float
    comp_mass_multiplier: float
    fs_max_kw: float
    fs_secs_to_peak_pwr: float
    fs_kwh: float
    fs_kwh_per_kg: float
    fc_max_kw: float
    fc_pwr_out_perc: np.ndarray
    fc_eff_map: np.ndarray
    fc_eff_type: str
    fc_sec_to_peak_pwr: float
    fc_base_kg: float
    fc_kw_per_kg: float
    min_fc_time_on: float
    idle_fc_kw: float
    mc_max_kw: float
    mc_pwr_out_perc: np.ndarray
    mc_eff_map: np.ndarray
    mc_sec_to_peak_pwr: float
    mc_pe_kg_per_kw: float
    mc_pe_base_kg: float
    ess_max_kw: float
    ess_max_kwh: float
    ess_kg_per_kwh: float
    ess_base_kg: float
    ess_round_trip_eff: float
    ess_life_coef_a: float
    ess_life_coef_b: float
    min_soc: float
    max_soc: float
    ess_dischg_to_fc_max_eff_perc: float
    ess_chg_to_fc_max_eff_perc: float
    wheel_inertia_kg_m2: float
    num_wheels: int
    wheel_rr_coef: float
    wheel_radius_m: float
    wheel_coef_of_fric: float
    max_accel_buffer_mph: float
    max_accel_buffer_perc_of_useable_soc: float
    perc_high_acc_buf: float
    mph_fc_on: float
    kw_demand_fc_on: float
    max_regen: bool
    stop_start: bool
    force_aux_on_fc: bool
    alt_eff: float
    chg_eff: float
    aux_kw: float
    trans_kg: float
    trans_eff: float
    ess_to_fuel_ok_error: float
    val_udds_mpgge: float
    val_hwy_mpgge: float
    val_comb_mpgge: float
    val_udds_kwh_per_mile: float
    val_hwy_kwh_per_mile: float
    val_comb_kwh_per_mile: float
    val_cd_range_mi: float
    val_const65_mph_kwh_per_mile: float
    val_const60_mph_kwh_per_mile: float
    val_const55_mph_kwh_per_mile: float
    val_const45_mph_kwh_per_mile: float
    val_unadj_udds_kwh_per_mile: float
    val_unadj_hwy_kwh_per_mile: float
    val0_to60_mph: float
    val_ess_life_miles: float
    val_range_miles: float
    val_veh_base_cost: float
    val_msrp: float
    fc_peak_eff_override: Optional[float] = field(default=None)
    mc_peak_eff_override: Optional[float] = field(default=None)
    # don't mess with this
    props: params.PhysicalProperties = params.PhysicalProperties()
    # gets set during __post_init__
    large_baseline_eff: np.ndarray = field(init=False)
    small_baseline_eff: np.ndarray = field(init=False)
    large_motor_power_kw: float = field(init=False)
    small_motor_power_kw: float = field(init=False)
    fc_perc_out_array: np.ndarray = field(init=False)
    mc_perc_out_array: np.ndarray = field(init=False)

    # Specify shape of mc regen efficiency curve
    # see "Regen" tab in FASTSim for Excel
    # Not sure if these params need further API modification;
    # will keep it as it-is
    regen_a: float = params.regen_a
    regen_b: float = params.regen_b

    # gets set during __post_init__
    max_roadway_chg_kw: np.ndarray = field(init=False)
    input_kw_out_array: np.ndarray = field(init=False)
    fc_kw_out_array: np.ndarray = field(init=False)
    fc_eff_array: np.ndarray = field(init=False)
    charging_on: bool = field(init=False)
    no_elec_sys: bool = field(init=False)
    no_elec_aux: bool = field(init=False)
    modern_max: float = field(init=False)
    mc_eff_array: np.ndarray = field(init=False)
    mc_kw_in_array: np.ndarray = field(init=False)
    mc_kw_out_array: np.ndarray = field(init=False)
    mc_max_elec_in_kw: float = field(init=False)
    mc_full_eff_array: np.ndarray = field(init=False)
    veh_kg: float = field(init=False)
    max_trac_mps2: float = field(init=False)
    ess_mass_kg: float = field(init=False)
    mc_mass_kg: float = field(init=False)
    fc_mass_kg: float = field(init=False)
    fs_mass_kg: float = field(init=False)

    # if True, some derived vehicle attributes are not calulated in python
    # but instead are calculated in Rust
    converted_to_rust: InitVar[bool] = field(default=True)

    @classmethod
    def from_vehdb(cls, vnum: int, veh_file: str = None, to_rust: bool = False) -> Self:
        """
        Load vehicle `vnum` from default vehdb or `veh_file`.
        Arguments:
            vnum: vehicle number
            veh_file: path to vehicle database file
            to_rust: if True, convert to rust-compatible vehicle
        """
        vehdf = DEFAULT_VEHDF
        if veh_file is not None:
            vehdf = pd.read_csv(veh_file)
        vehdf.set_index('selection', inplace=True, drop=False)

        return cls.from_df(vehdf, vnum, veh_file, to_rust)

    @classmethod
    def from_file(cls, filename: str, vnum: int = None, to_rust: bool = False) -> Self:
        """
        Loads vehicle from file `filename` (str).  Looks in working dir and then 
        fastsim/resources/vehdb, which also contains numerous examples of vehicle csv files.
        `vnum` is needed for multi-vehicle files.  
        Arguments:
            filename: path to vehicle database file
            vnum: vehicle number
            to_rust: if True, convert to rust-compatible vehicle
        """
        filename = str(filename)
        if not(str(filename).endswith('.csv')):
            filename = str(filename) + '.csv'
        if Path(filename).exists():
            filename = Path(filename)
        elif (VEHICLE_DIR / filename).exists():
            filename = VEHICLE_DIR / filename
        else:
            raise ValueError("Invalid vehicle filename.")

        vehdf = pd.read_csv(filename)

        if vnum is None:
            vehdf = vehdf.transpose()
            vehdf.columns = vehdf.iloc[1]
            vehdf.drop(vehdf.index[0], inplace=True)
            vehdf['selection'] = np.nan * np.ones(vehdf.shape[0])
            vehdf.loc['Param Value', 'selection'] = 0
            vnum = 0

        vehdf.set_index('selection', inplace=True, drop=False)

        veh_file = filename

        return cls.from_df(vehdf, vnum, veh_file, to_rust)

    @classmethod
    def from_df(cls, vehdf: pd.DataFrame, vnum: int, veh_file: Path, to_rust: bool = False) -> Self:
        """
        Given vehdf, generates dict to feed to `from_dict`.
        Arguments:
            vehdf: pandas dataframe of vehicle attributes
            vnum: vehicle number
            veh_file: path to vehicle database file
            to_rust: if True, convert to rust-compatible vehicle
        """
        # verify that only allowed columns have been provided
        for col in vehdf.columns:
            col = col.replace(" ", "_")
            assert col in list(OLD_TO_NEW.keys()) + list(NEW_TO_OLD.keys()) + \
                OPT_INIT_PARAMS, f"`{col}` is deprecated and must be removed from {veh_file}."

        vehdf.loc[vnum] = vehdf.loc[vnum].apply(clean_data)

        veh_dict = {}
        # set columns and values as instance attributes and values
        for col in vehdf.columns:
            col1 = str(col).replace(' ', '_')
            col1 = OLD_TO_NEW.get(col1, col1)
            # assign dataframe columns to vehicle
            veh_dict[col1] = vehdf.loc[vnum, col]

        # TODO: should this block be uncommented and converted to use logging?
        # missing_cols = set(TEMPLATE_VEHDF.columns) - set(veh_dict.keys())
        # if len(missing_cols) > 0:
        #     if verbose:
        #         print(f"0 filled in for {list(missing_cols)} missing from '{str(veh_file)}'.")
        #         print(f"Turn this warning off by passing `verbose=False` when instantiating vehicle.")
        #     for col in missing_cols:
        #         veh_dict[col] = 0.0

        return cls.from_dict(veh_dict, to_rust)

    @classmethod
    def from_dict(cls, veh_dict: dict, to_rust: bool = False) -> Self:
        """
        Load vehicle from dict with snake_case key names.  
        Arguments:
            veh_dict: dict of vehicle attributes
            to_rust: if True, convert to rust-compatible vehicle
        """
        # Power and efficiency arrays are defined in parameters.py
        # Can also be input in CSV as array under column fc_eff_map of form
        # [0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30]
        # no quotes necessary
        veh_dict['fc_eff_type'] = str(veh_dict['fc_eff_type'])

        try:
            # check if optional parameter fc_eff_map is provided in vehicle csv file
            veh_dict['fc_eff_map'] = np.array(ast.literal_eval(
                veh_dict['fc_eff_map']))
            logger.info("fc_eff_map is overriding fc_eff_type")

        except:
            warn_str = f"""fc_eff_type {
                veh_dict['fc_eff_map']} is not in {FC_EFF_TYPES},
                and `fc_eff_map` is not provided."""
            assert veh_dict['fc_eff_type'] in FC_EFF_TYPES, warn_str

            if veh_dict['fc_eff_type'] == SI:  # SI engine
                veh_dict['fc_eff_map'] = params.fc_eff_map_si

            # Atkinson cycle SI engine -- greater expansion
            elif veh_dict['fc_eff_type'] == ATKINSON:
                veh_dict['fc_eff_map'] = params.fc_eff_map_atk

            # Diesel (compression ignition) engine
            elif veh_dict['fc_eff_type'] == DIESEL:
                veh_dict['fc_eff_map'] = params.fc_eff_map_diesel

            elif veh_dict['fc_eff_type'] == H2FC:  # H2 fuel cell
                veh_dict['fc_eff_map'] = params.fc_eff_map_fuel_cell

            elif veh_dict['fc_eff_type'] == HD_DIESEL:  # heavy duty Diesel engine
                veh_dict['fc_eff_map'] = params.fc_eff_map_hd_diesel

        try:
            # check if optional parameter fc_pwr_out_perc is provided in vehicle csv file
            veh_dict['fc_pwr_out_perc'] = np.array(ast.literal_eval(
                veh_dict['fc_pwr_out_perc']))
        except:
            veh_dict['fc_pwr_out_perc'] = params.fc_pwr_out_perc

        fc_eff_map_len = len(veh_dict['fc_eff_map'])
        fc_pwr_len = len(veh_dict['fc_pwr_out_perc'])
        fc_eff_len_err = f'len(fcPwrOutPerc) ({fc_pwr_len}) is not' +\
            f'equal to len(fcEffMap) ({fc_eff_map_len})'
        assert len(veh_dict['fc_pwr_out_perc']) == len(
            veh_dict['fc_eff_map']), fc_eff_len_err

        # Defining MC efficiency curve as lookup table for %power_in vs power_out
        # see "Motor" tab in FASTSim for Excel

        # Power and efficiency arrays are defined in parameters.py
        # can also be overridden by motor power and efficiency columns in the input file
        # ensure that the column existed and the value in the cell wasn't empty (becomes NaN)
        try:
            # check if mc_pwr_out_perc is provided in vehicle csv file
            veh_dict['mc_pwr_out_perc'] = np.array(
                ast.literal_eval(veh_dict['mc_pwr_out_perc']))
        except:
            logger.info("no proper mc_pwr_out_perc provided; using default values")
            veh_dict['mc_pwr_out_perc'] = params.mc_pwr_out_perc

        # check if self-provided mc_pwr_out_perc has the required length
        large_baseline_eff_len = len(params.large_baseline_eff)
        mc_pwr_out_len = len(veh_dict['mc_pwr_out_perc'])
        mc_large_eff_len_err = f'len(mcPwrOutPerc) ({mc_pwr_out_len}) is not' +\
            f'equal to given len(largeBaselineEff) ({large_baseline_eff_len})'
        assert len(veh_dict['mc_pwr_out_perc']) == len(
            params.large_baseline_eff), mc_large_eff_len_err

        try:
            # check if mc_eff_map is provided in vehicle csv file
            veh_dict['mc_eff_map'] = np.array(
                ast.literal_eval(veh_dict['mc_eff_map']))
            mc_eff_map_len = len(veh_dict['mc_eff_map'])
            mc_eff_map_len_err = f'len(mcEffMap) ({mc_eff_map_len}) is not' +\
                f'equal to given len(largeBaselineEff) ({large_baseline_eff_len})'
            assert len(veh_dict['mc_pwr_out_perc']) == len(
                params.large_baseline_eff), mc_eff_map_len_err
        except:
            logger.info("no proper mc_eff_map provided; using default method to calculate")
            veh_dict['mc_eff_map'] = None

        # set stop_start if not provided
        if 'stop_start' in veh_dict and np.isnan(veh_dict['stop_start']):
            veh_dict['stop_start'] = False

        if ('veh_override_kg' not in veh_dict) or (veh_dict['veh_override_kg'] is None) or np.isnan(veh_dict['veh_override_kg']):
            veh_dict['veh_override_kg'] = None

        # check if veh_year provided in file, and, if not, provide value from scenario_name or default of 0
        if ('veh_year' not in veh_dict) or np.isnan(veh_dict['veh_year']):
            # regex is for vehicle model year if scenario_name starts with any 4 digit string
            if re.match('\d{4}', str(veh_dict['scenario_name'])):
                veh_dict['veh_year'] = np.int32(
                    re.match('\d{4}', str(veh_dict['scenario_name'])).group()
                )
            else:
                # set 0 as default to get correct type
                veh_dict['veh_year'] = np.int32(0)

        # in case veh_year gets loaded from file as float
        veh_dict['veh_year'] = np.int32(veh_dict['veh_year'])

        assert veh_dict['veh_pt_type'] in VEH_PT_TYPES, f"veh_pt_type {veh_dict['veh_pt_type']} not in {VEH_PT_TYPES}"

        if "fc_peak_eff_override" in veh_dict:
            try:
                veh_dict['fc_peak_eff_override'] = np.float64(
                    veh_dict['fc_peak_eff_override'])
                assert 0.0 < veh_dict['fc_peak_eff_override'] < 1.0
            except:
                logger.info("no proper fc_peak_eff_override provided; will not override fc_peak_eff")
                veh_dict['fc_peak_eff_override'] = None
        if "mc_peak_eff_override" in veh_dict:
            try:
                veh_dict['mc_peak_eff_override'] = np.float64(
                    veh_dict['mc_peak_eff_override'])
                assert 0.0 < veh_dict['mc_peak_eff_override'] < 1.0
            except:
                logger.info("no proper mc_peak_eff_override provided; will not override mc_peak_eff")
                veh_dict['mc_peak_eff_override'] = None

        # make sure types are right
        for key, val in veh_dict.items():
            if key != 'props':
                if key in ['veh_override_kg', 'mc_eff_map', 'fc_peak_eff_override', 'mc_peak_eff_override'] and val is None:
                    pass
                else:
                    veh_dict[key] = keys_and_types[key](val)

        #veh_dict['converted_to_rust'] = to_rust

        # for key in KEYS_TO_REMOVE:
        #    if key in veh_dict:
        #        del veh_dict[key]

        return cls(**veh_dict, converted_to_rust=to_rust)

    def __post_init__(self, converted_to_rust: bool = False):
        """
        Sets derived parameters.
        Arguments:
        ----------
        fc_peak_eff_override: float (0, 1) or -1, if provided and not -1, overrides engine peak efficiency
            with proportional scaling.  Default of -1 has no effect.
        mc_peak_eff_override: float (0, 1) or -1, if provided and not -1, overrides motor peak efficiency
            with proportional scaling.  Default of -1 has no effect.
        """
        if not converted_to_rust:
            self.set_derived()

    def set_derived(self):
        """
        Sets derived parameters.
        Arguments:
        ----------
        fc_peak_eff_override: float (0, 1) or -1, if provided and not -1, overrides engine peak efficiency
            with proportional scaling.  Default of -1 has no effect.
        mc_peak_eff_override: float (0, 1) or -1, if provided and not -1, overrides motor peak efficiency
            with proportional scaling.  Default of -1 has no effect.
        """
        # Build up constant
        self.large_baseline_eff = params.large_baseline_eff
        self.small_baseline_eff = params.small_baseline_eff
        self.small_motor_power_kw = 7.5   # default (float)
        self.large_motor_power_kw = 75.0  # default (float)
        self.fc_perc_out_array = params.fc_perc_out_array
        self.mc_perc_out_array = params.mc_perc_out_array
        # Build roadway power lookup table
        self.max_roadway_chg_kw = np.zeros(6)
        self.charging_on = False

        if self.scenario_name != 'Template Vehicle for setting up data types':
            if self.veh_pt_type == BEV:
                assert self.fs_max_kw == 0, f'max_fuel_stor_kw must be zero for provided BEV powertrain type in {self.scenario_name}'
                assert self.fs_kwh == 0, f'fuel_stor_kwh must be zero for provided BEV powertrain type in {self.scenario_name}'
                assert self.fc_max_kw == 0, f'max_fuel_conv_kw must be zero for provided BEV powertrain type in {self.scenario_name}'
            elif (self.veh_pt_type == CONV) and not(self.stop_start):
                assert self.mc_max_kw == 0, f'max_mc_kw must be zero for provided Conv powertrain type in {self.scenario_name}'
                assert self.ess_max_kw == 0, f'max_ess_kw must be zero for provided Conv powertrain type in {self.scenario_name}'
                assert self.ess_max_kwh == 0, f'max_ess_kwh must be zero for provided Conv powertrain type in {self.scenario_name}'

        # Checking if a vehicle has any hybrid components
        if (self.ess_max_kwh == 0) or (self.ess_max_kw == 0) or (self.mc_max_kw == 0):
            self.no_elec_sys = True
        else:
            self.no_elec_sys = False

        # Checking if aux loads go through an alternator
        if (self.no_elec_sys == True) or (self.mc_max_kw <= self.aux_kw) or (self.force_aux_on_fc == True):
            self.no_elec_aux = True
        else:
            self.no_elec_aux = False

        # discrete array of possible engine power outputs
        self.input_kw_out_array = self.fc_pwr_out_perc * self.fc_max_kw
        # Relatively continuous array of possible engine power outputs
        self.fc_kw_out_array = self.fc_max_kw * self.fc_perc_out_array
        # Creates relatively continuous array for fc_eff
        self.fc_eff_array = np.interp(
            x=self.fc_perc_out_array, xp=self.fc_pwr_out_perc, fp=self.fc_eff_map)

        self.modern_max = params.modern_max

        modern_diff = self.modern_max - max(self.large_baseline_eff)

        large_baseline_eff_adj = self.large_baseline_eff + modern_diff

        mc_kw_adj_perc = max(
            0.0,
            min(
                (self.mc_max_kw - self.small_motor_power_kw) /
                (self.large_motor_power_kw - self.small_motor_power_kw),
                1.0)
        )

        if self.mc_eff_map is None:
            self.mc_eff_map = mc_kw_adj_perc * large_baseline_eff_adj + \
                (1 - mc_kw_adj_perc) * self.small_baseline_eff
            self.mc_eff_array = self.mc_eff_map
        else:
            self.mc_eff_array = self.mc_eff_map

        mc_kw_out_array = np.linspace(
            0, 1, len(self.mc_perc_out_array)) * self.mc_max_kw

        mc_full_eff_array = np.interp(
            x=self.mc_perc_out_array, xp=self.mc_pwr_out_perc, fp=self.mc_eff_array)
        mc_full_eff_array[0] = 0
        mc_full_eff_array[-1] = self.mc_eff_array[-1]

        mc_kw_in_array = np.concatenate(
            (np.zeros(1, dtype=np.float64), mc_kw_out_array[1:] / mc_full_eff_array[1:]))
        mc_kw_in_array[0] = 0

        self.mc_kw_in_array = mc_kw_in_array
        self.mc_kw_out_array = mc_kw_out_array
        self.mc_max_elec_in_kw = max(mc_kw_in_array)
        self.mc_full_eff_array = mc_full_eff_array

        self.mc_max_elec_in_kw = max(self.mc_kw_in_array)

        if self.fc_peak_eff_override is not None:
            self.fc_peak_eff = self.fc_peak_eff_override
            logger.info("fc_peak_eff_override is modifying fc efficiency curve")
        if self.mc_peak_eff_override is not None:
            self.mc_peak_eff = self.mc_peak_eff_override
            logger.info("mc_peak_eff_override is modifying mc efficiency curve")

        # check that efficiencies are not violating the first law of thermo
        assert self.fc_eff_array.min() >= 0, f"min MC eff < 0 is not allowed"
        assert self.fc_peak_eff < 1, f"fcPeakEff >= 1 is not allowed."
        assert self.mc_full_eff_array.min() >= 0, f"min MC eff < 0 is not allowed"
        assert self.mc_peak_eff < 1, f"mcPeakEff >= 1 is not allowed."

        self.set_veh_mass()

    def set_veh_mass(self):
        """Calculate total vehicle mass.  Sum up component masses if 
        positive real number is not specified for self.veh_override_kg"""
        ess_mass_kg = 0
        mc_mass_kg = 0
        fc_mass_kg = 0
        fs_mass_kg = 0

        if (self.veh_override_kg is None) or (not(self.veh_override_kg > 0)):
            self.veh_override_kg = 0.0  # Rust/Python compatibility fix
            if self.ess_max_kwh == 0 or self.ess_max_kw == 0:
                ess_mass_kg = 0.0
            else:
                ess_mass_kg = ((self.ess_max_kwh * self.ess_kg_per_kwh) +
                               self.ess_base_kg) * self.comp_mass_multiplier
            if self.mc_max_kw == 0:
                mc_mass_kg = 0.0
            else:
                mc_mass_kg = (self.mc_pe_base_kg+(self.mc_pe_kg_per_kw
                                                  * self.mc_max_kw)) * self.comp_mass_multiplier
            if self.fc_max_kw == 0:
                fc_mass_kg = 0.0
            else:
                fc_mass_kg = (1 / self.fc_kw_per_kg * self.fc_max_kw +
                              self.fc_base_kg) * self.comp_mass_multiplier
            if self.fs_max_kw == 0:
                fs_mass_kg = 0.0
            else:
                fs_mass_kg = ((1 / self.fs_kwh_per_kg) *
                              self.fs_kwh) * self.comp_mass_multiplier
            self.veh_kg = self.cargo_kg + self.glider_kg + self.trans_kg * \
                self.comp_mass_multiplier + ess_mass_kg + \
                mc_mass_kg + fc_mass_kg + fs_mass_kg
        # if positive real number is specified for veh_override_kg, use that
        else:
            self.veh_kg = self.veh_override_kg

        self.max_trac_mps2 = (
            self.wheel_coef_of_fric * self.drive_axle_weight_frac * self.veh_kg * self.props.a_grav_mps2 /
            (1 + self.veh_cg_m * self.wheel_coef_of_fric / self.wheel_base_m)
        ) / (self.veh_kg * self.props.a_grav_mps2) * self.props.a_grav_mps2

        # copying to instance attributes
        self.ess_mass_kg = np.float64(ess_mass_kg)
        self.mc_mass_kg = np.float64(mc_mass_kg)
        self.fc_mass_kg = np.float64(fc_mass_kg)
        self.fs_mass_kg = np.float64(fs_mass_kg)

    # properties -- these were created to make sure modifications to curves propagate

    @property
    def max_fc_eff_kw(
        self): return self.fc_kw_out_array[np.argmax(self.fc_eff_array)]

    @property
    def max_regen_kwh(self): return 0.5 * self.veh_kg * (27**2) / (3600 * 1000)

    @property
    def veh_type_selection(self) -> str:
        """
        Copying veh_pt_type to additional key
        to be consistent with Excel version but not used in Python version
        """
        return self.veh_pt_type

    def get_mcPeakEff(self) -> float:
        "Return `np.max(self.mc_eff_array)`"
        assert np.max(self.mc_full_eff_array) == np.max(self.mc_eff_array)
        return np.max(self.mc_full_eff_array)

    def set_mcPeakEff(self, new_peak):
        """
        Set motor peak efficiency EVERYWHERE.  

        Arguments:
        ----------
        new_peak: float, new peak motor efficiency in decimal form 
        """
        self.mc_eff_array *= new_peak / self.mc_eff_array.max()
        self.mc_full_eff_array *= new_peak / self.mc_full_eff_array.max()

    mc_peak_eff = property(get_mcPeakEff, set_mcPeakEff)

    def get_fcPeakEff(self) -> float:
        "Return `np.max(self.fc_eff_array)`"
        return np.max(self.fc_eff_array)

    def set_fcPeakEff(self, new_peak):
        """
        Set fc peak efficiency EVERWHERE.  

        Arguments:
        ----------
        new_peak: float, new peak fc efficiency in decimal form 
        """
        self.fc_eff_array *= new_peak / self.fc_eff_array.max()
        self.fc_eff_map *= new_peak / self.fc_eff_array.max()

    fc_peak_eff = property(get_fcPeakEff, set_fcPeakEff)

    def get_numba_veh(self):
        """Deprecated."""
        raise NotImplementedError(
            "This method has been deprecated")

    def to_rust(self) -> RustVehicle:
        """Return a Rust version of the vehicle"""
        # NOTE: copying calls the constructor again which calls RustVehicle's post_init()
        return copy_vehicle(self, 'rust')

    def reset_orphaned(self):
        """Dummy method for flexibility between Rust/Python version interfaces"""
        pass


ref_veh = Vehicle.from_vehdb(5)


class LegacyVehicle(object):
    """
    Implementation of Vehicle with legacy keys.
    """

    def __init__(self, vehicle: Vehicle):
        """
        Given cycle, returns legacy cycle.
        """
        for key, val in NEW_TO_OLD.items():
            self.__setattr__(val, copy.deepcopy(vehicle.__getattribute__(key)))


RETURN_TYPES = ['dict', 'vehicle', 'legacy', 'rust']
DICT, VEHICLE, LEGACY, RUST = RETURN_TYPES


def copy_vehicle(veh: Vehicle, return_type: str = None, deep: bool = True) -> Dict[str, np.ndarray] | Vehicle | LegacyVehicle | RustVehicle:
    """Returns copy of Vehicle.
    Arguments:
    veh: instantiated Vehicle or RustVehicle
    return_type: 
        'dict': dict
        'vehicle': Vehicle 
        'legacy': LegacyVehicle
        'rust': RustVehicle
    """

    veh_dict = {}

    if return_type is None:
        if RUST_AVAILABLE and isinstance(veh, RustVehicle):
            return_type = RUST
        elif isinstance(veh, Vehicle):
            return_type = VEHICLE
        elif isinstance(veh, LegacyVehicle):
            return_type = LEGACY
        else:
            raise NotImplementedError(
                "Only implemented for rust, dict, vehicle, or legacy.")

    for key in keys_and_types.keys():
        if key in KEYS_TO_REMOVE:
            continue
        if (
            RUST_AVAILABLE
            and type(veh.__getattribute__(key)) == fsr.RustPhysicalProperties
        ):
            pp = veh.__getattribute__(key)
            # TODO: replace the below with a call to copy_physical_properties(...)
            new_pp = fsr.RustPhysicalProperties()
            new_pp.air_density_kg_per_m3 = pp.air_density_kg_per_m3
            new_pp.a_grav_mps2 = pp.a_grav_mps2
            new_pp.kwh_per_gge = pp.kwh_per_gge
            new_pp.fuel_rho_kg__L = pp.fuel_rho_kg__L
            new_pp.fuel_afr_stoich = pp.fuel_afr_stoich
            veh_dict[key] = new_pp
        elif ((RUST_AVAILABLE)
              and (
                (type(veh.__getattribute__(key)) == fsr.Pyo3ArrayU32)
                or (type(veh.__getattribute__(key)) == fsr.Pyo3ArrayF64)
                or (type(veh.__getattribute__(key)) == fsr.Pyo3ArrayBool)
                or (type(veh.__getattribute__(key)) == fsr.Pyo3ArrayF64)
        )
        ):
            veh_dict[key] = list(veh.__getattribute__(key))
        else:
            veh_dict[key] = copy.deepcopy(
                veh.__getattribute__(key)) if deep else veh.__getattribute__(key)

    if return_type == DICT:
        return veh_dict
    elif return_type == VEHICLE:
        return Vehicle.from_dict(veh_dict)
    elif return_type == LEGACY:
        return LegacyVehicle(veh_dict)
    elif RUST_AVAILABLE and return_type == RUST:
        veh_dict['props'] = params.copy_physical_properties(
            veh_dict['props'], return_type, deep)
        veh_dict = {key: veh_dict[key] for key in veh_dict if key not in [
            "large_baseline_eff", "small_baseline_eff"]}
        return RustVehicle(**veh_dict)
    else:
        raise ValueError(f"Invalid return_type: '{return_type}'")


def veh_equal(veh1: Vehicle, veh2: Vehicle, full_out: bool = False) -> bool:
    """Given veh1 and veh2, which can be Vehicle and/or RustVehicle
    instances, return True if equal.

    Arguments:
    ----------
    """
    TOL = 1e-6

    veh_dict1 = copy_vehicle(veh1, return_type='dict', deep=True)
    veh_dict2 = copy_vehicle(veh2, return_type='dict', deep=True)
    err_list = []
    keys = list(veh_dict1.keys())
    for key in keys:
        if key == "props":
            p1 = veh_dict1[key]
            p2 = veh_dict2[key]
            if not params.physical_properties_equal(p1, p2):
                if not full_out:
                    return False
                err_list.append(
                    {'key': key, 'val1': veh_dict1[key], 'val2': veh_dict2[key]})
        elif pd.api.types.is_list_like(veh_dict1[key]):
            if (np.abs(np.array(veh_dict1[key]) - np.array(veh_dict2[key])) > TOL).any():
                if not full_out:
                    return False
                err_list.append(
                    {'key': key, 'val1': veh_dict1[key], 'val2': veh_dict2[key]})
        elif veh_dict1[key] != veh_dict2[key]:
            try:
                if np.isnan(veh_dict1[key]) and np.isnan(veh_dict2[key]):
                    continue  # treat as equal if both nan
            except:
                pass
            if not full_out:
                return False
            err_list.append(
                {'key': key, 'val1': veh_dict1[key], 'val2': veh_dict2[key]})
    if full_out:
        return err_list

    return True
