"""
Calibration script for 2021_Hyundai_Sonata_Hybrid_Blue    
"""
import pprint
from pathlib import Path
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns
import pandas as pd  # noqa: F401
import polars as pl  # noqa: F401
from typing import List, Dict
from pymoo.core.problem import StarmapParallelization
from copy import deepcopy

import fastsim as fsim
from fastsim import pymoo_api

# Unit conversion constants
mps_per_mph = 0.447
celsius_to_kelvin_offset = 273.15
lhv_btu_per_lbm = 18_575 
lhv_joules_per_gram = 43_205.450 

# Initialize seaborn plot configuration
sns.set_style("darkgrid")

veh = fsim.Vehicle.from_file(Path(__file__).parent / "f3-vehicles/2021_Hyundai_Sonata_Hybrid_Blue.yaml")
veh_dict = veh.to_pydict()

sim_params_dict = fsim.SimParams.default().to_pydict()
sim_params_dict["trace_miss_opts"] = "AllowChecked"
sim_params = fsim.SimParams.from_pydict(sim_params_dict, skip_init=False)

# Obtain the data from
# https://www.anl.gov/taps/d3-2021-hyundai-sonata-hybrid
# and then copy it to the local folder below
cyc_folder_path = Path(__file__).parent / "dyno_test_data/2021 Hyundai Sonata Hybrid/Extended Datasets"
assert cyc_folder_path.exists()

# Test data columns
time_column = "Time[s]_RawFacilities"
speed_column = "Dyno_Spd[mph]"
cabin_temp_column = "Cabin_Temp[C]"
eng_clnt_temp_column = "engine_coolant_temp_PCAN__C"
fuel_column = "Eng_FuelFlow_Direct2[gps]"
cell_temp_column = "Cell_Temp[C]"

# See 2021_Hyundai_Sonata_Hybrid_TestSummary_2022-03-01_D3.xlsx for cycle-level data
cyc_files_dict: Dict[str, Dict] = {
    # TODO: pipe in solar load
    # HWYx2, 2 bag in 95°F test cell with solar @850W/m^2, HVAC-ON-AUTO-72°F, ECO drive mode
    "62202004 Test Data.txt": {cell_temp_column: 35, "solar load [W/m^2]": 850, "set temp [*C]": 22}, 

    # US06x2, 4 (split) bag in 95°F test cell with solar @850W/m^2, HVAC-ON-AUTO-72°F, ECO drive mode
    "62202005 Test Data.txt": {cell_temp_column: 35, "solar load [W/m^2]": 850, "set temp [*C]": 22},

    # UDDS, 2 bag, warm start in ECO mode
    # UDDS x1, room temperature ambient
    "62201013 Test Data.txt": {cell_temp_column: 25, "solar load [W/m^2]": 0, "set temp [*C]": None},

    # Hwyx2, 2 bag, warm start in ECO mode
    # HWY x2, room temperature ambient
    "62201014 Test Data.txt": {cell_temp_column: 25, "solar load [W/m^2]": 0, "set temp [*C]": None},

    # TODO: trim this to stop at ~1400 s
    # UDDSx2, 4 bag (FTP), cold start, in COLD (20°F) test cell, HVAC-AUTO-72°F, ECO drive mode
    "62202013 Test Data.txt": {cell_temp_column: -6.7, "solar load [W/m^2]": 0, "set temp [*C]": 22},

    # UDDS, 2 bag, warm start, in COLD (20°F) test cell, HVAC-AUTO-72°F, ECO drive mode
    "62202014 Test Data.txt": {cell_temp_column: -6.7, "solar load [W/m^2]": 0, "set temp [*C]": 22},

    # US06x2, 4 (split) bag, warm start, in COLD (20°F) test cell, HVAC-AUTO-72°F, ECO drive mode
    "62202016 Test Data.txt": {cell_temp_column: -6.7, "solar load [W/m^2]": 0, "set temp [*C]": 22},

    # TODO: check for seat heater usage in cold cycles and account for that in model!
}
cyc_files: List[Path] = [cyc_folder_path / cyc_file for cyc_file in cyc_files_dict.keys()]
print("\ncyc_files:\n", '\n'.join([cf.name for cf in cyc_files]), sep='')

# use random or manual selection to retain ~70% of cycles for calibration,
# and reserve the remaining for validation
cyc_files_for_cal: List[str] = [
    # HWY x2, hot (M155), HVAC active (B155)
    "62202004 Test Data.txt", 
    # "62202005 Test Data.txt",
    # UDDS x1, room temperature ambient
    "62201013 Test Data.txt",
    # HWY x2, room temperature ambient
    "62201014 Test Data.txt",
    # UDDSx2, 4 bag (FTP), cold start, in COLD (20°F) test cell, HVAC-AUTO-72°F, ECO drive mode
    "62202013 Test Data.txt",
    # "62202014 Test Data.txt", 
    # US06x2, 4 (split) bag, warm start, in COLD (20°F) test cell, HVAC-AUTO-72°F, ECO drive mode
    "62202016 Test Data.txt",
]
cyc_files_for_cal: List[Path] = [cyc_file for cyc_file in cyc_files if cyc_file.name in cyc_files_for_cal]
assert len(cyc_files_for_cal) > 0
print("\ncyc_files_for_cal:\n", '\n'.join([cf.name for cf in cyc_files_for_cal]), sep='')

def df_to_cyc(df: pd.DataFrame) -> fsim.Cycle:
    cyc_dict = {
        "time_seconds": df[time_column].to_list(),
        "speed_meters_per_second": (df[speed_column] * mps_per_mph).to_list(),
        "temp_amb_air_kelvin": (df[cell_temp_column] + celsius_to_kelvin_offset).to_list(),
        # TODO: pipe solar load from `Cycle` into cabin thermal model
        # "pwr_solar_load_watts": df[],
    }
    return fsim.Cycle.from_pydict(cyc_dict, skip_init=False)

pt_type_var = "HybridElectricVehicle"

def veh_init(cyc_file_stem: str, dfs: Dict[str, pd.DataFrame]) -> fsim.Vehicle:
    vd = deepcopy(veh_dict)
    # initialize SOC
    vd['pt_type'][pt_type_var]['res']['state']['soc'] = \
        dfs[cyc_file_stem]["HVBatt_SOC_high_precision_PCAN__per"].iloc[1] / 100
    assert 0 < vd['pt_type'][pt_type_var]['res']['state']['soc'] < 1, "\ninit soc: {}\nhead: {}".format(
        vd['pt_type'][pt_type_var]['res']['state']['soc'], dfs[cyc_file_stem]["HVBatt_SOC_high_precision_PCAN__per"].head())
    # initialize cabin temp
    vd['cabin']['LumpedCabin']['state']['temperature_kelvin'] = \
        dfs[cyc_file_stem][cabin_temp_column][0] + celsius_to_kelvin_offset
    # initialize battery temperature to match cabin temperature because battery
    # temperature is not available in test data
    # Also, battery temperature has no effect in the HEV because efficiency data
    # does not go below 23*C and there is no active thermal management
    vd['pt_type'][pt_type_var]['res']['thrml']['RESLumpedThermal']['state']['temperature_kelvin'] = \
        dfs[cyc_file_stem]["Cabin_Temp[C]"][0] + celsius_to_kelvin_offset
    # initialize engine temperature
    vd['pt_type'][pt_type_var]['fc']['thrml']['FuelConverterThermal']['state']['temperature_kelvin'] = \
        dfs[cyc_file_stem][eng_clnt_temp_column][0] + celsius_to_kelvin_offset
    # set HVAC set point temperature
    te_set = next(iter([v["set temp [*C]"] for k, v in cyc_files_dict.items() if k.replace(".txt", "") == cyc_file_stem]))
    vd['hvac']['LumpedCabin']['te_set_kelvin'] = te_set + celsius_to_kelvin_offset if te_set is not None else None

    return fsim.Vehicle.from_pydict(vd, skip_init=False)

def resample_df(df: pd.DataFrame) -> pd.DataFrame:
    # filter out "before" time
    df = df[df[time_column] >= 0.0]
    dt = np.diff(df[time_column], prepend=1)
    df['cumu. fuel [g]'] = (dt * df[fuel_column]).cumsum()
    df['cumu. dist [mph*s]'] = (dt * df[speed_column]).cumsum()
    init_speed = df[speed_column].iloc[0]
    init_fuel = df[fuel_column].iloc[0]
    df = df[::10] # convert to ~1 Hz
    df.reset_index(inplace=True)
    dt_new = np.diff(df[time_column])
    df[speed_column] = np.concatenate(([init_speed], np.diff(df['cumu. dist [mph*s]']) / dt_new))
    df[fuel_column] = np.concatenate(([init_fuel], np.diff(df['cumu. fuel [g]']) / dt_new))

    return df


dfs_for_cal: Dict[str, pd.DataFrame] = {
    # `delimiter="\t"` should work for tab separated variables
    cyc_file.stem: resample_df(pd.read_csv(cyc_file, delimiter="\t")) for cyc_file in cyc_files_for_cal
}
    
cycs_for_cal: Dict[str, fsim.Cycle] = {}
# populate `cycs_for_cal`
for (cyc_file_stem, df) in dfs_for_cal.items():
    cyc_file_stem: str
    df: pd.DataFrame
    cyc_dict_raw = df.to_dict()
    cyc_file_stem: str
    df: pd.DataFrame
    cycs_for_cal[cyc_file_stem] = df_to_cyc(df)

sds_for_cal: Dict[str, fsim.SimDrive] = {}
# populate `sds_for_cal`
for (cyc_file_stem, cyc) in cycs_for_cal.items():
    cyc_file_stem: str
    cyc: fsim.Cycle
    # NOTE: maybe change `save_interval` to 5
    veh = veh_init(cyc_file_stem, dfs_for_cal)
    sds_for_cal[cyc_file_stem] = fsim.SimDrive(veh, cyc, sim_params).to_pydict()

cyc_files_for_val: List[Path] = list(set(cyc_files) - set(cyc_files_for_cal))
assert len(cyc_files_for_val) > 0
print("\ncyc_files_for_val:\n", '\n'.join([cf.name for cf in cyc_files_for_val]), sep='')

dfs_for_val: Dict[str, pd.DataFrame] = {
    # `delimiter="\t"` should work for tab separated variables
    cyc_file.stem: resample_df(pd.read_csv(cyc_file, delimiter="\t")) for cyc_file in cyc_files_for_val
}

cycs_for_val: Dict[str, fsim.Cycle] = {}
# populate `cycs_for_val`
for (cyc_file_stem, df) in dfs_for_val.items():
    cyc_file_stem: str
    df: pd.DataFrame
    cycs_for_val[cyc_file_stem] = df_to_cyc(df)

sds_for_val: Dict[str, fsim.SimDrive] = {}
# populate `sds_for_val`
for (cyc_file_stem, cyc) in cycs_for_val.items():
    cyc_file_stem: str
    cyc: fsim.Cycle
    veh = veh_init(cyc_file_stem, dfs_for_val)
    sds_for_val[cyc_file_stem] = fsim.SimDrive(veh, cyc, sim_params).to_pydict()

# Setup model parameters and objectives
## Parameter Functions `param_fns`
def new_em_eff_max(sd_dict, new_eff_max) -> Dict:
    """
    Set `new_eff_max` in `ElectricMachine`
    """
    em = fsim.ElectricMachine.from_pydict(sd_dict['veh']['pt_type'][pt_type_var]['em'])
    em.__eff_fwd_max = new_eff_max
    sd_dict['veh']['pt_type'][pt_type_var]['em'] = em.to_pydict()
    return sd_dict

def new_em_eff_range(sd_dict, new_eff_range) -> Dict:
    """
    Set `new_eff_range` in `ElectricMachine`
    """
    em = fsim.ElectricMachine.from_pydict(sd_dict['veh']['pt_type'][pt_type_var]['em'])
    em.__eff_fwd_range = new_eff_range
    sd_dict['veh']['pt_type'][pt_type_var]['em'] = em.to_pydict()
    return sd_dict

def new_fc_eff_max(sd_dict, new_eff_max) -> Dict:
    """
    Set `new_eff_max` in `FuelConverter`
    """
    fc = fsim.FuelConverter.from_pydict(sd_dict['veh']['pt_type'][pt_type_var]['fc'])
    fc.__eff_max = new_eff_max
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"] = fc.to_pydict()
    return sd_dict

def new_fc_eff_range(sd_dict, new_eff_range) -> Dict:
    """
    Set `new_eff_range` in `FuelConverter`
    """
    fc = fsim.FuelConverter.from_pydict(sd_dict['veh']['pt_type'][pt_type_var]['fc'])
    fc_eff_max = fc.eff_max
    # TODO: this is a quick and dirty apprach, change to using constraints in PyMOO
    fc.__eff_range = min(new_eff_range, fc_eff_max * 0.95)
    sd_dict['veh']['pt_type'][pt_type_var]['fc'] = fc.to_pydict()
    return sd_dict

def new_cab_shell_htc_w_per_m2_k(sd_dict, new_val) -> Dict:
    sd_dict['veh']['cabin']['LumpedCabin']['cab_shell_htc_to_amb_watts_per_square_meter_kelvin'] = new_val
    return sd_dict

def new_cab_htc_to_amb_stop_w_per_m2_k(sd_dict, new_val) -> Dict:
    sd_dict['veh']['cabin']['LumpedCabin']['cab_htc_to_amb_stop_watts_per_square_meter_kelvin'] = new_val
    return sd_dict

def new_cab_tm_j_per_k(sd_dict, new_val) -> Dict:
    sd_dict['veh']['cabin']['LumpedCabin']['heat_capacitance_joules_per_kelvin'] = new_val
    return sd_dict

def new_cab_length_m(sd_dict, new_val) -> Dict:
    sd_dict['veh']['cabin']['LumpedCabin']['length_meters'] = new_val
    return sd_dict

def new_speed_soc_disch_buffer_m_per_s(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["pt_cntrl"]["RGWDB"]["speed_soc_disch_buffer_meters_per_second"] = new_val
    return sd_dict
    
def new_speed_soc_disch_buffer_coeff(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["pt_cntrl"]["RGWDB"]["speed_soc_disch_buffer_coeff"] = new_val
    return sd_dict
    
def new_speed_soc_fc_on_buffer_m_per_s(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["pt_cntrl"]["RGWDB"]["speed_soc_fc_on_buffer_meters_per_second"] = new_val
    return sd_dict
    
def new_speed_soc_fc_on_buffer_coeff(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["pt_cntrl"]["RGWDB"]["speed_soc_fc_on_buffer_coeff"] = new_val
    return sd_dict
    
def new_fc_min_time_on_s(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["pt_cntrl"]["RGWDB"]["fc_min_time_on_seconds"] = new_val
    return sd_dict
    
def new_frac_pwr_demand_fc_forced_on(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["pt_cntrl"]["RGWDB"]["frac_pwr_demand_fc_forced_on"] = new_val
    return sd_dict
    
def new_frac_of_most_eff_pwr_to_run_fc(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["pt_cntrl"]["RGWDB"]["frac_of_most_eff_pwr_to_run_fc"] = new_val
    return sd_dict

def new_hvac_p_w_per_k(sd_dict, new_val) -> Dict:
    sd_dict['veh']['hvac']['LumpedCabin']['p_watts_per_kelvin'] = new_val
    return sd_dict

def new_hvac_i(sd_dict, new_val) -> Dict:
    """Set `new_val` for HVAC integral control gain"""
    sd_dict['veh']['hvac']['LumpedCabin']['i'] = new_val
    return sd_dict

# def new_hvac_pwr_i_max_watts(sd_dict, new_val) -> Dict:
#     sd_dict['veh']['hvac']['LumpedCabin']['pwr_i_max_watts'] = new_val
#     return sd_dict

# def new_hvac_d(sd_dict, new_val) -> Dict:
#     sd_dict['veh']['hvac']['LumpedCabin']['d'] = new_val
#     return sd_dict

# def new_hvac_pwr_thrml_max_watts(sd_dict, new_val) -> Dict:
#     sd_dict['veh']['hvac']['LumpedCabin']['pwr_thrml_max_watts'] = new_val
#     return sd_dict

def new_hvac_frac_of_ideal_cop(sd_dict, new_val) -> Dict:
    sd_dict['veh']['hvac']['LumpedCabin']['frac_of_ideal_cop'] = new_val
    return sd_dict

# def new_hvac_pwr_aux_for_hvac_max_watt(sd_dict, new_val) -> Dict:
#     sd_dict['veh']['hvac']['LumpedCabin']['pwr_aux_for_hvac_max_watts'] = new_val
#     return sd_dict
    
def new_fc_thrml_heat_capacitance_j_per_k(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["heat_capacitance_joules_per_kelvin"] = new_val
    return sd_dict

def new_fc_thrml_length_for_convection_m(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["length_for_convection_meters"] = new_val
    return sd_dict

def new_fc_thrml_htc_to_amb_stop_w_per_m2_k(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["htc_to_amb_stop_watts_per_square_meter_kelvin"] = new_val
    return sd_dict

def new_fc_thrml_conductance_from_comb_w_per_k(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["conductance_from_comb_watts_per_kelvin"] = new_val
    return sd_dict

def new_fc_thrml_max_frac_from_comb(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["max_frac_from_comb"] = new_val
    return sd_dict

def new_fc_thrml_radiator_effectiveness(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["radiator_effectiveness"] = new_val
    return sd_dict

def new_fc_thrml_fc_eff_model_Exponential_offset(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["fc_eff_model"]["Exponential"]["offset"] = new_val
    return sd_dict

def new_fc_thrml_fc_eff_model_Exponential_lag(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["fc_eff_model"]["Exponential"]["lag"] = new_val
    return sd_dict

def new_fc_thrml_fc_eff_model_Exponential_minimum(sd_dict, new_val) -> Dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["fc"]["thrml"]["FuelConverterThermal"]["fc_eff_model"]["Exponential"]["minimum"] = new_val
    return sd_dict

# veh.pt_type.HybridElectricVehicle.pt_cntrl.RGWDB.speed_soc_regen_buffer_meters_per_second
# veh.pt_type.HybridElectricVehicle.pt_cntrl.RGWDB.speed_soc_regen_buffer_coeff
# veh.pt_type.HybridElectricVehicle.pt_cntrl.RGWDB.speed_fc_forced_on_meters_per_second
# veh.pt_type.HybridElectricVehicle.pt_cntrl.RGWDB.temp_fc_forced_on_kelvin
# veh.pt_type.HybridElectricVehicle.pt_cntrl.RGWDB.temp_fc_allowed_off_kelvin

# Objective Functions -- `obj_fns`
def get_mod_soc(sd_dict):
    return np.array(sd_dict['veh']['pt_type'][pt_type_var]['res']['history']['soc'])

def get_exp_soc(df):
    return df['HVBatt_SOC_high_precision_PCAN__per'] / 100

def get_mod_fc_temp_celsius(sd_dict):
    return np.array(sd_dict['veh']['pt_type'][pt_type_var]['fc']['thrml']['FuelConverterThermal']['history']['temperature_kelvin']) - celsius_to_kelvin_offset

def get_exp_fc_temp_celsius(df):
    return df[eng_clnt_temp_column] 

def get_mod_cab_temp_celsius(sd_dict):
    return np.array(sd_dict['veh']['cabin']['LumpedCabin']['history']['temperature_kelvin']) - celsius_to_kelvin_offset

def get_exp_cab_temp_celsius(df):
    return df[cabin_temp_column]

def get_mod_speed_m_per_s(sd_dict):
    return np.array(sd_dict['veh']['history']['speed_ach_meters_per_second'])

def get_exp_speed_m_per_s(df):
    return df[speed_column] * mps_per_mph

def get_mod_pwr_fuel_kw(sd_dict):
    return np.array(sd_dict['veh']['pt_type'][pt_type_var]['fc']['history']['pwr_fuel_watts']) / 1e3

def get_exp_pwr_fuel_kw(df):
    return df[fuel_column] * lhv_joules_per_gram / 1e3

def get_mod_energy_fuel_megajoules(sd_dict):
    return np.array(sd_dict['veh']['pt_type'][pt_type_var]['fc']['history']['energy_fuel_joules']) / 1e6

def get_exp_energy_fuel_megajoules(df):
    pwr_fuel_watts = df[fuel_column] * lhv_joules_per_gram
    dt = np.diff(df[time_column], prepend=0)
    energy_fuel_joules = np.cumsum(pwr_fuel_watts * dt) / 1e6
    return energy_fuel_joules

def get_mod_pwr_hvac_kw(sd_dict):
    return np.array(sd_dict['veh']['hvac']['LumpedCabin']['history']['pwr_aux_for_hvac_watts']) / 1e3

def get_exp_pwr_hvac_kw(df):
    if df[cell_temp_column].mean() < 15:
        pwr_hvac = [0] * len(df)
    else:
        pwr_hvac = df["HVAC_Power_Hioki_P3[W]"] / 1e3
    return pwr_hvac

## Constraint functions
def get_fc_temp_too_hot(sd_dict):
    te_fc_deg_c = sd_dict['veh']['pt_type'][pt_type_var]['fc']['thrml']['FuelConverterThermal']['state']['temperature_kelvin'] - celsius_to_kelvin_offset
    if np.any(te_fc_deg_c > 115):
        return 1
    else:
        return -1

save_path = Path(__file__).parent / "pymoo_res" / Path(__file__).stem
save_path.mkdir(exist_ok=True, parents=True)

## Model Objectives
cal_mod_obj = pymoo_api.ModelObjectives(
    models = sds_for_cal,
    dfs = dfs_for_cal,
    obj_fns=(
        (
            get_mod_soc,
            get_exp_soc
        ),
        (
            get_mod_energy_fuel_megajoules,
            get_exp_energy_fuel_megajoules
        ),
        (
            get_mod_pwr_fuel_kw,
            get_exp_pwr_fuel_kw
        ),
        (
            get_mod_cab_temp_celsius,
            get_exp_cab_temp_celsius  
        ),
        (
            get_mod_fc_temp_celsius,
            get_exp_fc_temp_celsius  
        ),
        (
            get_mod_speed_m_per_s,
            get_exp_speed_m_per_s  
        ),
        (
            get_mod_pwr_hvac_kw,
            get_exp_pwr_hvac_kw  
        ),
    ),
    param_fns_and_bounds=(
        (new_em_eff_max, (0.80, 0.99)), # new_em_eff_max
        (new_em_eff_range, (0.1, 0.6)), # new_em_eff_range
        (new_fc_eff_max, (0.32, 0.45)), # new_fc_eff_max
        # (new_fc_eff_range, # (0.2, 0.45)), # range is not working # #  # range is not working
        (new_cab_shell_htc_w_per_m2_k, (10, 250)), # new_cab_shell_htc
        (new_cab_htc_to_amb_stop_w_per_m2_k, (10, 250)), # new_cab_htc_to_amb_stop
        (new_cab_tm_j_per_k, (100e3, 350e3)), # new_cab_tm
        (new_cab_length_m, (1.5, 7)), # new_cab_length
        (new_speed_soc_disch_buffer_m_per_s, (5, 100)), # new_speed_soc_disch_buffer_meters_per_second
        (new_speed_soc_disch_buffer_coeff, (0.25, 5.0)), # new_speed_soc_disch_buffer_coeff
        (new_speed_soc_fc_on_buffer_m_per_s, (5, 100)), # new_speed_soc_fc_on_buffer_meters_per_second
        (new_speed_soc_fc_on_buffer_coeff, (0.25, 5.0)), # new_speed_soc_fc_on_buffer_coeff
        (new_fc_min_time_on_s, (3, 30)), # new_fc_min_time_on_seconds
        (new_frac_pwr_demand_fc_forced_on, (0.3, 0.8)), # new_frac_pwr_demand_fc_forced_on
        (new_frac_of_most_eff_pwr_to_run_fc, (0.1, 1.0)), # new_frac_of_most_eff_pwr_to_run_fc
        (new_hvac_p_w_per_k, (5, 1_000)), # new_hvac_p_watts_per_kelvin
        (new_hvac_i, (1, 100)), # new_hvac_i
        (new_hvac_frac_of_ideal_cop, (0.05, 0.35)), # new_hvac_frac_of_ideal_cop
        (new_fc_thrml_heat_capacitance_j_per_k, (50e3, 300e3)), # new_fc_thrml_heat_capacitance_joules_per_kelvin,
        (new_fc_thrml_length_for_convection_m, (0.2, 3)), # new_fc_thrml_length_for_convection_meters,
        (new_fc_thrml_htc_to_amb_stop_w_per_m2_k, (5, 100)), # new_fc_thrml_htc_to_amb_stop_watts_per_square_meter_kelvin,
        (new_fc_thrml_conductance_from_comb_w_per_k, (5, 5_000)), # new_fc_thrml_conductance_from_comb_watts_per_kelvin,
        # (new_fc_thrml_max_frac_from_comb, # ()), # new_fc_thrml_max_frac_from_comb,
        (new_fc_thrml_radiator_effectiveness, (3, 300)), # new_fc_thrml_radiator_effectiveness,
        (new_fc_thrml_fc_eff_model_Exponential_offset, (220, 350)), # new_fc_thrml_fc_eff_model_Exponential_offset,
        (new_fc_thrml_fc_eff_model_Exponential_lag, (10, 60)), # new_fc_thrml_fc_eff_model_Exponential_lag,
        (new_fc_thrml_fc_eff_model_Exponential_minimum, (0.15, 0.35)), # new_fc_thrml_fc_eff_model_Exponential_minimum,
        # TODO: make sure this has functions for modifying
        # - aux power
        # - battery thermal -- not necessary for HEV because battery temperature has no real effect
        #     - thermal mass
        #     - convection to ambient
        #     - convection to cabin
    ),
    # must match order and length of `params_fns`
    constr_fns=(
        get_fc_temp_too_hot,
    ),
    verbose=False,    
)

val_mod_obj = deepcopy(cal_mod_obj)
val_mod_obj.dfs = dfs_for_val
val_mod_obj.models = sds_for_val

def perturb_params(pos_perturb_dec: float = 0.05, neg_perturb_dec: float = 0.1):
    """
    # Arguments:
    # - `pos_perturb_doc`: perturbation percentage added to all params.  Can be overridden invididually
    # - `neg_perturb_doc`: perturbation percentage subtracted from all params.  Can be overridden invididually
    """
    em = fsim.ElectricMachine.from_pydict(veh_dict['pt_type'][pt_type_var]['em'], skip_init=False)
    fc = fsim.FuelConverter.from_pydict(veh_dict['pt_type'][pt_type_var]['fc'], skip_init=False)
    baseline_params_and_bounds = [
        (em.eff_fwd_max, None),
        (em.eff_fwd_range, None),
        (fc.eff_max, None),
        # (fc.eff_range, None),
        (veh_dict['cabin']['LumpedCabin']['cab_shell_htc_to_amb_watts_per_square_meter_kelvin'], None),
        (veh_dict['cabin']['LumpedCabin']['cab_htc_to_amb_stop_watts_per_square_meter_kelvin'], None),
        (veh_dict['cabin']['LumpedCabin']['heat_capacitance_joules_per_kelvin'], None),
        (veh_dict['cabin']['LumpedCabin']['length_meters'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['speed_soc_disch_buffer_meters_per_second'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['speed_soc_disch_buffer_coeff'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['speed_soc_fc_on_buffer_meters_per_second'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['speed_soc_fc_on_buffer_coeff'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['fc_min_time_on_seconds'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['frac_pwr_demand_fc_forced_on'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['frac_of_most_eff_pwr_to_run_fc'], None),
        (veh_dict['hvac']['LumpedCabin']['p_watts_per_kelvin'], None),
        (veh_dict['hvac']['LumpedCabin']['i'], None),
        (veh_dict['hvac']['LumpedCabin']['frac_of_ideal_cop'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['heat_capacitance_joules_per_kelvin'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['length_for_convection_meters'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['htc_to_amb_stop_watts_per_square_meter_kelvin'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['conductance_from_comb_watts_per_kelvin'], None),
        # (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['max_frac_from_comb'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['radiator_effectiveness'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['fc_eff_model']['Exponential']['offset'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['fc_eff_model']['Exponential']['lag'], None),
        (veh_dict['pt_type']['HybridElectricVehicle']['fc']['thrml']['FuelConverterThermal']['fc_eff_model']['Exponential']['minimum'], None)
    ]

    baseline_params = [bpb[0] for bpb in baseline_params_and_bounds]

    print("Verifying that model responds to input parameter changes by individually perturbing parameters")
    baseline_errors = cal_mod_obj.get_errors(
        cal_mod_obj.update_params([param for param in baseline_params])
    )[0]
    
    for i, param_and_bounds in enumerate(baseline_params_and_bounds):
        param = param_and_bounds[0]
        bounds = param_and_bounds[1]
        # +5%
        if bounds is not None:
            param_pos_perturb_dec = bounds[0]
            param_neg_perturb_dec = bounds[1]
        else:
            param_pos_perturb_dec = pos_perturb_dec
            param_neg_perturb_dec = neg_perturb_dec

        assert param_pos_perturb_dec >= 0
        assert param_neg_perturb_dec >= 0

        perturbed_params = baseline_params.copy()
        perturbed_params[i] = param * (1 + param_pos_perturb_dec)
        perturbed_errors = cal_mod_obj.get_errors(cal_mod_obj.update_params(perturbed_params))
        if np.all(perturbed_errors == baseline_errors):
          print("\nperturbed_errros:")
          pprint.pp(perturbed_errors) 
          print("baseline_errors")
          pprint.pp(baseline_errors)
          print("")
          raise Exception(f"+{100 * param_pos_perturb_dec}% perturbation failed for param {cal_mod_obj.param_fns[i].__name__}")

        # -5%
        perturbed_params = baseline_params.copy()
        perturbed_params[i] = param * (1 - param_neg_perturb_dec)
        perturbed_errors = cal_mod_obj.get_errors(cal_mod_obj.update_params(perturbed_params))
        if np.all(perturbed_errors == baseline_errors):
            print("\nperturbed_errros:")
            pprint.pp(perturbed_errors) 
            print("baseline_errors")
            pprint.pp(baseline_errors)
            print("")
            raise Exception(f"-{100 * param_neg_perturb_dec}% perturbation failed for param {cal_mod_obj.param_fns[i].__name__}")

    print("Success!")

if __name__ == "__main__":
    print("Params and bounds:")
    pprint.pp(cal_mod_obj.param_fns_and_bounds)
    print("")
    perturb_params()
    parser = pymoo_api.get_parser()
    args = parser.parse_args()

    n_processes = args.processes 
    n_max_gen = args.n_max_gen 
    # should be at least as big as n_processes
    pop_size = args.pop_size 
    run_minimize = not (args.skip_minimize)

    print("Starting calibration.")
    algorithm = pymoo_api.NSGA2(
        # size of each population
        pop_size=pop_size,
        # LatinHyperCube sampling seems to be more effective than the default
        # random sampling
        sampling=pymoo_api.LHS(),
    )
    termination = pymoo_api.DMOT(
        # max number of generations, default of 10 is very small
        n_max_gen=n_max_gen,
        # evaluate tolerance over this interval of generations every
        period=5,
        # parameter variation tolerance
        xtol=args.xtol,
        # objective variation tolerance
        ftol=args.ftol
    )

    if n_processes == 1:
        print("Running serial evaluation.")
        # series evaluation
        # Setup calibration problem
        cal_prob = pymoo_api.CalibrationProblem(
            mod_obj=cal_mod_obj,
        )
        
        res, res_df = pymoo_api.run_minimize(
            problem=cal_prob,
            algorithm=algorithm,
            termination=termination,
            save_path=save_path,
        )
    else:
        print(f"Running parallel evaluation with n_processes: {n_processes}.")
        assert n_processes > 1
        # parallel evaluation
        import multiprocessing

        with multiprocessing.Pool(n_processes) as pool:
            problem = pymoo_api.CalibrationProblem(
                mod_obj=cal_mod_obj,
                elementwise_runner=StarmapParallelization(pool.starmap),
            )
            res, res_df = pymoo_api.run_minimize(
                problem=problem,
                algorithm=algorithm,
                termination=termination,
                save_path=save_path,
            )

