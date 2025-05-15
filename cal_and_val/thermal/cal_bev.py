"""
Calibration script for 2020 Chevrolet Bolt EV
"""

import pprint
from pathlib import Path
import numpy as np  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns
import pandas as pd  # noqa: F401
import polars as pl  # noqa: F401
from pymoo.core.problem import StarmapParallelization
from copy import deepcopy

import fastsim as fsim
from fastsim import pymoo_api

# Unit conversion constants
mps_per_mph = 0.447
celsius_to_kelvin_offset = 273.15

# Initialize seaborn plot configuration
sns.set_style("darkgrid")

veh = fsim.Vehicle.from_file(
    Path(__file__).parent / "f3-vehicles/2020 Chevrolet Bolt EV.yaml")
veh_dict = veh.to_pydict()

sim_params_dict = fsim.SimParams.default().to_pydict()
sim_params_dict["trace_miss_opts"] = "AllowChecked"
sim_params = fsim.SimParams.from_pydict(sim_params_dict, skip_init=False)


# Obtain the data from
# https://www.anl.gov/taps/d3-2020-chevrolet-bolt
# and then copy it to the local folder below
cyc_folder_path = Path(__file__).parent / \
    "dyno_test_data/2020 Chevrolet Bolt/Extended Datasets"
assert cyc_folder_path.exists(), cyc_folder_path

# Test data columns
time_column = "Time[s]_RawFacilities"
speed_column = "Dyno_Spd[mph]"
cabin_temp_column = "Cabin_Driver_Headrest_Temp__C"
batt_temp_column = "HVBatt_pack_average_temp_HPCM2__C"
eng_clnt_temp_column = "engine_coolant_temp_PCAN__C"
cell_temp_column = "Cell_Temp[C]"
soc_column = "HVBatt_SOC_CAN4__per"

# See 2020_Chevrolet_Bolt_TestSummary_201005.xlsm for cycle-level data
cyc_files_dict: dict[str, dict] = {
    # TODO: check for seat heater usage in cold cycles and account for that in model!
    # 20F (heater maybe on? Col R in test summary), UDDS + HWY + UDDS + US06
    "62009051 Test Data.txt": {
        cell_temp_column: -6.7,
        "solar load [W/m^2]": None,
        "set temp [*C]": 22,
    },
    # 20F (heater maybe on? Col R in test summary), US06 + UDDS + HWY + UDDS
    "62009053 Test Data.txt": {
        cell_temp_column: -6.7,
        "solar load [W/m^2]": None,
        "set temp [*C]": 22,
    },
    # room temperature (no HVAC), UDDS + HWY + UDDS + US06
    "62009019 Test Data.txt": {
        cell_temp_column: 22,
        "solar load [W/m^2]": None,
        "set temp [*C]": None,
    },
    # room temperature (no HVAC), US06 + UDDS + HWY + UDDS
    "62009021 Test Data.txt": {
        cell_temp_column: 22,
        "solar load [W/m^2]": None,
        "set temp [*C]": None,
    },
    # TODO: check for solar load (should be around 1 kW / m^2) and implement or
    # this somewhere (`drive_cycle`???)
    # 95F (HVAC on), UDDS + HWY + UDDS
    "62009040 Test Data.txt": {
        cell_temp_column: 35,
        "solar load [W/m^2]": 850,
        # 28 *C is approximately the steady state temperature
        "set temp [*C]": 28,
    },
    # Commented out due to junk data for headrest temperatures
    # # 95F (HVAC on), US06
    # "62009041 Test Data.txt": {
    #     cell_temp_column: 35,
    #     "solar load [W/m^2]": None,
    #     "set temp [*C]": 22,
    # },
    # Data quality of the following cycles has not been checked
    # 95F (HVAC on), US06 x2
    "62009043 Test Data.txt": {
        cell_temp_column: 38,
        "solar load [W/m^2]": 850,
        "set temp [*C]": 28,
    },
    # # 95F (HVAC on), SC03 x 3, #1
    # "62009044 Test Data.txt": {
    #     cell_temp_column: 38,
    #     "solar load [W/m^2]": 850,
    #     "set temp [*C]": 22,
    # },
}
assert len(cyc_files_dict) > 0
cyc_files: list[Path] = [cyc_folder_path /
                         cyc_file for cyc_file in cyc_files_dict.keys()]
print("\ncyc_files:\n", "\n".join([cf.name for cf in cyc_files]), sep="")

# use random or manual selection to retain ~70% of cycles for calibration,
# and reserve the remaining for validation
cyc_files_for_cal: list[str] = [
    "62009051 Test Data.txt",
    # "62009053 Test Data.txt"
    "62009019 Test Data.txt",
    # "62009021 Test Data.txt",
    "62009040 Test Data.txt",
    # "62009041 Test Data.txt"
    # "62009044 Test Data.txt",
]
cyc_files_for_cal: list[Path] = [
    cyc_file for cyc_file in cyc_files if cyc_file.name in cyc_files_for_cal
]
assert len(cyc_files_for_cal) > 0, cyc_files_for_cal
print("\ncyc_files_for_cal:\n", "\n".join(
    [cf.name for cf in cyc_files_for_cal]), sep="")


def df_to_cyc(df: pd.DataFrame) -> fsim.Cycle:
    cyc_dict = {
        "time_seconds": df[time_column].to_list(),
        "speed_meters_per_second": (df[speed_column] * mps_per_mph).to_list(),
        "temp_amb_air_kelvin": (df[cell_temp_column] + celsius_to_kelvin_offset).to_list(),
        # TODO: pipe solar load from `Cycle` into cabin thermal model
        # "pwr_solar_load_watts": df[],
    }
    return fsim.Cycle.from_pydict(cyc_dict, skip_init=False)


pt_type_var = "BatteryElectricVehicle"
cabin_type_var = "LumpedCabin"
hvac_type_var = "LumpedCabinAndRES"


def veh_init(cyc_file_stem: str, dfs: dict[str, pd.DataFrame]) -> fsim.Vehicle:
    vd = deepcopy(veh_dict)

    # initialize SOC
    vd["pt_type"][pt_type_var]["res"]["state"]["soc"] = dfs[cyc_file_stem][soc_column].iloc[1] / 100
    assert 0 < vd["pt_type"][pt_type_var]["res"]["state"]["soc"] < 1, (
        "\ninit soc: {}\nhead: {}".format(
            vd["pt_type"][pt_type_var]["res"]["state"]["soc"],
            dfs[cyc_file_stem]["HVBatt_SOC_CAN4__per"].head(),
        )
    )
    # initialize cabin temp
    vd["cabin"][cabin_type_var]["state"]["temperature_kelvin"] = (
        dfs[cyc_file_stem][cabin_temp_column][0] + celsius_to_kelvin_offset
    )

    vd["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"]["state"][
        "temperature_kelvin"
    ] = dfs[cyc_file_stem][batt_temp_column][0] + celsius_to_kelvin_offset

    # set HVAC set point temperature
    te_set = next(
        iter(
            [
                v["set temp [*C]"]
                for k, v in cyc_files_dict.items()
                if k.replace(".txt", "") == cyc_file_stem
            ]
        )
    )
    vd["hvac"][hvac_type_var]["te_set_res_kelvin"] = (
        22 + celsius_to_kelvin_offset if te_set is not None else None
    )
    vd["hvac"][hvac_type_var]["te_set_cab_kelvin"] = (
        te_set + celsius_to_kelvin_offset if te_set is not None else None
    )

    return fsim.Vehicle.from_pydict(vd, skip_init=False)


def resample_df(df: pd.DataFrame) -> pd.DataFrame:
    # filter out "before" time
    df = df[df[time_column] >= 0.0]
    dt = np.diff(df[time_column], prepend=1)
    df["cumu. dist [mph*s]"] = (dt * df[speed_column]).cumsum()
    init_speed = df[speed_column].iloc[0]
    df = df[::10]  # convert to ~1 Hz
    df.reset_index(inplace=True)
    dt_new = np.diff(df[time_column])
    df[speed_column] = np.concatenate(
        ([init_speed], np.diff(df["cumu. dist [mph*s]"]) / dt_new))
    df = df[df[time_column] < 2160]

    return df


dfs_for_cal: dict[str, pd.DataFrame] = {
    # `delimiter="\t"` should work for tab separated variables
    cyc_file.stem: resample_df(pd.read_csv(cyc_file, delimiter="\t"))
    for cyc_file in cyc_files_for_cal
}

cycs_for_cal: dict[str, fsim.Cycle] = {}  # populate `cycs_for_cal`
for cyc_file_stem, df in dfs_for_cal.items():
    cyc_file_stem: str
    df: pd.DataFrame
    cyc_dict_raw = df.to_dict()
    cyc_file_stem: str
    df: pd.DataFrame
    cycs_for_cal[cyc_file_stem] = df_to_cyc(df)

sds_for_cal: dict[str, fsim.SimDrive] = {}
# populate `sds_for_cal`
for cyc_file_stem, cyc in cycs_for_cal.items():
    cyc_file_stem: str
    cyc: fsim.Cycle
    # NOTE: maybe change `save_interval` to 5
    veh = veh_init(cyc_file_stem, dfs_for_cal)
    sds_for_cal[cyc_file_stem] = fsim.SimDrive(
        veh, cyc, sim_params).to_pydict()

cyc_files_for_val: list[Path] = list(set(cyc_files) - set(cyc_files_for_cal))
assert len(cyc_files_for_val) > 0
print("\ncyc_files_for_val:\n", "\n".join(
    [cf.name for cf in cyc_files_for_val]), sep="")

dfs_for_val: dict[str, pd.DataFrame] = {
    # `delimiter="\t"` should work for tab separated variables
    cyc_file.stem: resample_df(pd.read_csv(cyc_file, delimiter="\t"))
    for cyc_file in cyc_files_for_val
}

cycs_for_val: dict[str, fsim.Cycle] = {}
# populate `cycs_for_val`
for cyc_file_stem, df in dfs_for_val.items():
    cyc_file_stem: str
    df: pd.DataFrame
    if "9043" in cyc_file_stem:
        # trim out junk data
        df = df[df[time_column] < 500]
        dfs_for_val[cyc_file_stem] = df
    cycs_for_val[cyc_file_stem] = df_to_cyc(df)

sds_for_val: dict[str, fsim.SimDrive] = {}
# populate `sds_for_val`
for cyc_file_stem, cyc in cycs_for_val.items():
    cyc_file_stem: str
    cyc: fsim.Cycle
    veh = veh_init(cyc_file_stem, dfs_for_val)
    sds_for_val[cyc_file_stem] = fsim.SimDrive(
        veh, cyc, sim_params).to_pydict()


# Setup model objectives
# Parameter Functions
# `param_fns`
def new_em_eff_max(sd_dict: dict, new_eff_max: float) -> dict:
    """
    Set `new_eff_max` in `ElectricMachine`
    """
    em = fsim.ElectricMachine.from_pydict(
        sd_dict["veh"]["pt_type"][pt_type_var]["em"])
    em.__eff_fwd_max = new_eff_max
    sd_dict["veh"]["pt_type"][pt_type_var]["em"] = em.to_pydict()
    return sd_dict


def new_em_eff_range(sd_dict, new_eff_range) -> dict:
    """
    Set `new_eff_range` in `ElectricMachine`
    """
    em = fsim.ElectricMachine.from_pydict(
        sd_dict["veh"]["pt_type"][pt_type_var]["em"])
    em.__eff_fwd_range = new_eff_range
    sd_dict["veh"]["pt_type"][pt_type_var]["em"] = em.to_pydict()
    return sd_dict


def new_cab_shell_htc_w_per_m2_k(sd_dict, new_val) -> dict:
    sd_dict["veh"]["cabin"][cabin_type_var][
        "cab_shell_htc_to_amb_watts_per_square_meter_kelvin"
    ] = new_val
    return sd_dict


def new_cab_htc_to_amb_stop_w_per_m2_k(sd_dict, new_val) -> dict:
    sd_dict["veh"]["cabin"][cabin_type_var]["cab_htc_to_amb_stop_watts_per_square_meter_kelvin"] = (
        new_val
    )
    return sd_dict


def new_cab_tm_j_per_k(sd_dict, new_val) -> dict:
    sd_dict["veh"]["cabin"][cabin_type_var]["heat_capacitance_joules_per_kelvin"] = new_val
    return sd_dict


def new_cab_length_m(sd_dict, new_val) -> dict:
    sd_dict["veh"]["cabin"][cabin_type_var]["length_meters"] = new_val
    return sd_dict


def new_res_cndctnc_to_amb(sd_dict, new_val) -> dict:
    """
    Sets conductance from res to amb to `new_val`
    """
    sd_dict["veh"]["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"][
        "conductance_to_amb_watts_per_kelvin"
    ] = new_val
    return sd_dict


def new_res_cndctnc_to_cab(sd_dict, new_val) -> dict:
    """
    Sets conductance from res to cabin to `new_val`
    """
    sd_dict["veh"]["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"][
        "conductance_to_cab_watts_per_kelvin"
    ] = new_val
    return sd_dict


def new_res_tm_j_per_k(sd_dict, new_val) -> dict:
    sd_dict["veh"]["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"][
        "heat_capacitance_joules_per_kelvin"
    ] = new_val
    return sd_dict


def new_hvac_p_res_w_per_k(sd_dict, new_val) -> dict:
    sd_dict["veh"]["hvac"][hvac_type_var]["p_res_watts_per_kelvin"] = new_val
    return sd_dict


def new_hvac_i_res(sd_dict, new_val) -> dict:
    """Set `new_val` for HVAC integral control gain"""
    sd_dict["veh"]["hvac"][hvac_type_var]["i_res"] = new_val
    return sd_dict


def new_hvac_d_res(sd_dict, new_val) -> dict:
    """Set `new_val` for HVAC derivative control gain"""
    sd_dict["veh"]["hvac"][hvac_type_var]["d_res"] = new_val
    return sd_dict


def new_hvac_p_cabin_w_per_k(sd_dict, new_val) -> dict:
    sd_dict["veh"]["hvac"][hvac_type_var]["p_cabin_watts_per_kelvin"] = new_val
    return sd_dict


def new_hvac_i_cabin(sd_dict, new_val) -> dict:
    """Set `new_val` for HVAC integral control gain"""
    sd_dict["veh"]["hvac"][hvac_type_var]["i_cabin"] = new_val
    return sd_dict


def new_hvac_d_cabin(sd_dict, new_val) -> dict:
    """Set `new_val` for HVAC derivative control gain"""
    sd_dict["veh"]["hvac"][hvac_type_var]["d_cabin"] = new_val
    return sd_dict


def new_hvac_frac_of_ideal_cop(sd_dict, new_val) -> dict:
    sd_dict["veh"]["hvac"][hvac_type_var]["frac_of_ideal_cop"] = new_val
    return sd_dict


# Objective Functions -- `obj_fns`
def get_mod_soc(sd_dict):
    return np.array(sd_dict["veh"]["pt_type"][pt_type_var]["res"]["history"]["soc"])


def get_exp_soc(df):
    return df["HVBatt_SOC_CAN4__per"] / 100


def get_mod_cab_temp_celsius(sd_dict):
    return (
        np.array(sd_dict["veh"]["cabin"][cabin_type_var]
                 ["history"]["temperature_kelvin"])
        - celsius_to_kelvin_offset
    )


def get_exp_cab_temp_celsius(df):
    return df[cabin_temp_column]


def get_mod_batt_temp_celsius(sd_dict):
    mod_batt_temp_celsiusi_float = np.array(
        sd_dict["veh"]["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"]["history"]["temperature_kelvin"]
    ) - celsius_to_kelvin_offset
    # the test data temperature is quantized
    mod_batt_temp_celsius_int = np.array(
        [int(temp) for temp in mod_batt_temp_celsiusi_float])
    return mod_batt_temp_celsius_int


def get_exp_batt_temp_celsius(df):
    return df[batt_temp_column]


def get_mod_pwr_hvac_kw(sd_dict):
    return (
        np.array(sd_dict["veh"]["hvac"][cabin_type_var]
                 ["history"]["pwr_aux_for_hvac_watts"]) / 1e3
    )


def get_exp_pwr_hvac_kw(df):
    if df[cell_temp_column].mean() < 15:
        pwr_hvac = [0] * len(df)
    else:
        pwr_hvac = df["HVAC_Power_Hioki_P3[W]"] / 1e3
    return pwr_hvac


# post-processing functions
def get_mod_soc_delta(sd_dict: dict) -> float:
    soc = np.array(sd_dict["veh"]["pt_type"]
                   [pt_type_var]["res"]["history"]["soc"])
    soc_delta: float = soc[-1] - soc[0]
    return soc_delta


def get_exp_soc_delta(df: pd.DataFrame) -> float:
    soc = df["HVBatt_SOC_CAN4__per"].to_numpy() / 100
    soc_delta: float = soc[-1] - soc[0]
    return soc_delta


save_path = Path(__file__).parent / "pymoo_res" / Path(__file__).stem
save_path.mkdir(exist_ok=True, parents=True)

# Model Objectives
cal_mod_obj = pymoo_api.ModelObjectives(
    models=sds_for_cal,
    dfs=dfs_for_cal,
    obj_fns=(
        (get_mod_soc, get_exp_soc),
        (get_mod_cab_temp_celsius, get_exp_cab_temp_celsius),
        (get_mod_batt_temp_celsius, get_exp_batt_temp_celsius),
        # TODO: add objectives for:
        # - achieved and cycle speed
        # - battery temperature -- BEV only, if available
        # - HVAC power for cabin, if available
        # - HVAC power for res, if available
    ),
    param_fns_and_bounds=(
        (new_em_eff_max, (0.80, 0.99)),  # new_em_eff_max,
        (new_em_eff_range, (0.1, 0.6)),  # new_em_eff_range,
        # new_cab_shell_htc_w_per_m2_k,
        (new_cab_shell_htc_w_per_m2_k, (10, 350)),
        # new_cab_htc_to_amb_stop_w_per_m2_k,
        (new_cab_htc_to_amb_stop_w_per_m2_k, (10, 250)),
        (new_cab_tm_j_per_k, (50e3, 350e3)),  # new_cab_tm_j_per_k,
        (new_cab_length_m, (1.5, 7)),  # new_cab_length_m,
        (new_res_cndctnc_to_amb, (1, 60)),  # new_res_cndctnc_to_amb,
        (new_res_cndctnc_to_cab, (1, 60)),  # new_res_cndctnc_to_cab,
        (new_res_tm_j_per_k, (30e3, 200e3)),  # new_res_tm_j_per_k,
        (new_hvac_p_res_w_per_k, (5, 1_000)),  # new_hvac_p_res_w_per_k,
        (new_hvac_i_res, (1, 100)),  # new_hvac_i_res,
        # (new_hvac_d_res, (1, 100)),  # new_hvac_d_res,
        (new_hvac_p_cabin_w_per_k, (5, 1_000)),  # new_hvac_p_cabin_w_per_k,
        (new_hvac_i_cabin, (1, 100)),  # new_hvac_i_cabin,
        # (new_hvac_d_cabin, (1, 100)),  # new_hvac_d_cabin,
        # new_hvac_frac_of_ideal_cop,
        (new_hvac_frac_of_ideal_cop, (0.15, 0.35)),
    ),
    constr_fns=(),
    verbose=False,
)

val_mod_obj = deepcopy(cal_mod_obj)
val_mod_obj.dfs = dfs_for_val
val_mod_obj.models = sds_for_val


# TODO: put in parameter perturbation here
def perturb_params(pos_perturb_dec: float = 0.05, neg_perturb_dec: float = 0.1):
    """
    # Arguments:
    # - `pos_perturb_doc`: perturbation percentage added to all params.  Can be overridden invididually
    # - `neg_perturb_doc`: perturbation percentage subtracted from all params.  Can be overridden invididually
    """
    em = fsim.ElectricMachine.from_pydict(
        veh_dict["pt_type"][pt_type_var]["em"], skip_init=False)
    baseline_params_and_bounds = [
        (em.eff_fwd_max, None),
        (em.eff_fwd_range, None),
        (
            veh_dict["cabin"][cabin_type_var]["cab_shell_htc_to_amb_watts_per_square_meter_kelvin"],
            None,
        ),
        (
            veh_dict["cabin"][cabin_type_var]["cab_htc_to_amb_stop_watts_per_square_meter_kelvin"],
            None,
        ),
        (veh_dict["cabin"][cabin_type_var]
         ["heat_capacitance_joules_per_kelvin"], None),
        (veh_dict["cabin"][cabin_type_var]["length_meters"], None),
        (
            veh_dict["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"][
                "conductance_to_amb_watts_per_kelvin"
            ],
            None,
        ),
        (
            veh_dict["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"][
                "conductance_to_cab_watts_per_kelvin"
            ],
            None,
        ),
        (
            veh_dict["pt_type"][pt_type_var]["res"]["thrml"]["RESLumpedThermal"][
                "heat_capacitance_joules_per_kelvin"
            ],
            None,
        ),
        (veh_dict["hvac"][hvac_type_var]["p_res_watts_per_kelvin"], None),
        (veh_dict["hvac"][hvac_type_var]["i_res"], None),
        # (veh_dict["hvac"][hvac_type_var]["d_res"], None),
        (veh_dict["hvac"][hvac_type_var]["p_cabin_watts_per_kelvin"], None),
        (veh_dict["hvac"][hvac_type_var]["i_cabin"], None),
        # (veh_dict["hvac"][hvac_type_var]["d_cabin"], None),
        (veh_dict["hvac"][hvac_type_var]["frac_of_ideal_cop"], None),
    ]

    baseline_params = [bpb[0] for bpb in baseline_params_and_bounds]

    print(
        "Verifying that model responds to input parameter changes by individually perturbing parameters"
    )
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
        perturbed_errors = cal_mod_obj.get_errors(
            cal_mod_obj.update_params(perturbed_params))
        if np.all(perturbed_errors == baseline_errors):
            print("\nperturbed_errros:")
            pprint.pp(perturbed_errors)
            print("baseline_errors")
            pprint.pp(baseline_errors)
            print("")
            raise Exception(
                f"+{100 * param_pos_perturb_dec}% perturbation failed for param {cal_mod_obj.param_fns[i].__name__}"
            )

        # -5%
        perturbed_params = baseline_params.copy()
        perturbed_params[i] = param * (1 - param_neg_perturb_dec)
        perturbed_errors = cal_mod_obj.get_errors(
            cal_mod_obj.update_params(perturbed_params))
        if np.all(perturbed_errors == baseline_errors):
            print("\nperturbed_errros:")
            pprint.pp(perturbed_errors)
            print("baseline_errors")
            pprint.pp(baseline_errors)
            print("")
            raise Exception(
                f"-{100 * param_neg_perturb_dec}% perturbation failed for param {cal_mod_obj.param_fns[i].__name__}"
            )

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

    print(f"Starting calibration with: {args}.")
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
        ftol=args.ftol,
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
