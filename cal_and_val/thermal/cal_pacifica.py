# %%
# Imports

from copy import deepcopy

import numpy as np
from scipy.optimize import differential_evolution

import fastsim as fsim


def relative_error(actual: float, expected: float) -> float:
    return (actual - expected) / expected


SHOW_INTERMEDIATE_OUTPUT = False


# %%
# Setters
# The 'set_' prefix is stripped when parsing parameter names


def set_fc_thrml_heat_capacitance_j_per_k(veh_dict: dict, new_val: float) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"][
        "heat_capacitance_joules_per_kelvin"
    ] = new_val
    return veh_dict


def set_fc_thrml_length_for_convection_m(veh_dict: dict, new_val: float) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"][
        "length_for_convection_meters"
    ] = new_val
    return veh_dict


def set_fc_thrml_htc_to_amb_stop_w_per_m2_k(veh_dict: dict, new_val: float) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"][
        "htc_to_amb_stop_watts_per_square_meter_kelvin"
    ] = new_val
    return veh_dict


def set_fc_thrml_conductance_from_comb_w_per_k(veh_dict: dict, new_val: float) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"][
        "conductance_from_comb_watts_per_kelvin"
    ] = new_val
    return veh_dict


def set_fc_thrml_radiator_effectiveness(veh_dict: dict, new_val: float) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"][
        "radiator_effectiveness"
    ] = new_val
    return veh_dict


def set_cab_shell_htc_w_per_m2_k(veh_dict: dict, new_val: float) -> dict:
    veh_dict["cabin"]["LumpedCabin"][
        "cab_shell_htc_to_amb_watts_per_square_meter_kelvin"
    ] = new_val
    return veh_dict


def set_hvac_frac_of_ideal_cop(veh_dict: dict, new_val: float) -> dict:
    veh_dict["hvac"]["LumpedCabin"]["frac_of_ideal_cop"] = new_val
    return veh_dict


# fc_eff_model parameters
def set_fc_thrml_fc_eff_model_exponential_offset(
    veh_dict: dict, new_val: float
) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"]["fc_eff_model"][
        "Exponential"
    ]["offset"] = new_val
    return veh_dict


def set_fc_thrml_fc_eff_model_exponential_lag(veh_dict: dict, new_val: float) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"]["fc_eff_model"][
        "Exponential"
    ]["lag"] = new_val
    return veh_dict


def set_fc_thrml_fc_eff_model_exponential_minimum(
    veh_dict: dict, new_val: float
) -> dict:
    veh_dict["pt_type"][PT_TYPE]["fc"]["thrml"]["FuelConverterThermal"]["fc_eff_model"][
        "Exponential"
    ]["minimum"] = new_val
    return veh_dict


# %%
# Targets
TARGET_CITY_MPG = 18.9954
TARGET_HWY_MPG = 28.1266

TARGET_CITY_GAL_PER_MILE = 1 / TARGET_CITY_MPG
TARGET_HWY_GAL_PER_MILE = 1 / TARGET_HWY_MPG

PT_TYPE = "Conv"

# Load the base vehicle once; all evaluations copy from this dict.
veh_base = fsim.Vehicle.from_resource(
    "2026_Chrysler_Pacifica_Select_Thermal_DFCO_StopStart.yaml"
)
veh_base_dict = veh_base.to_pydict()

# %%
# Parameters
#   (parameter setter function, lower bound, upper bound)
PARAM_REGISTRY: list[tuple] = [
    (set_fc_thrml_heat_capacitance_j_per_k, 50_000.0, 300_000.0),
    # (set_fc_thrml_length_for_convection_m, 1.0, 3.0),  # TODO: causes error
    (set_fc_thrml_htc_to_amb_stop_w_per_m2_k, 5.0, 100.0),
    (set_fc_thrml_conductance_from_comb_w_per_k, 5.0, 5_000.0),
    (set_fc_thrml_radiator_effectiveness, 3.0, 300.0),
    (set_cab_shell_htc_w_per_m2_k, 10.0, 250.0),
    (set_hvac_frac_of_ideal_cop, 0.05, 0.35),
    # fc_eff_model parameters
    (set_fc_thrml_fc_eff_model_exponential_offset, 220.0, 350.0),
    (set_fc_thrml_fc_eff_model_exponential_lag, 10.0, 60.0),
    (set_fc_thrml_fc_eff_model_exponential_minimum, 0.15, 0.35),
]
PARAM_NAMES = [fn.__name__.removeprefix("set_") for fn, *_ in PARAM_REGISTRY]
BOUNDS = [(lb, ub) for _, lb, ub in PARAM_REGISTRY]

# %%
# Objective function


def run_five_cycle(
    veh: fsim.Vehicle, logging: bool = False
) -> tuple[float, float, float, float]:
    city_mpg, hwy_mpg, comb_mpg = fsim.five_cycle(veh, logging=logging)

    # Calculate error in terms of gal/mile
    city_gal_per_mi = 1 / city_mpg
    hwy_gal_per_mi = 1 / hwy_mpg

    err_city = relative_error(city_gal_per_mi, TARGET_CITY_GAL_PER_MILE)
    err_hwy = relative_error(hwy_gal_per_mi, TARGET_HWY_GAL_PER_MILE)
    err_comb = relative_error(
        1 / comb_mpg, 1 / ((TARGET_CITY_MPG + TARGET_HWY_MPG) / 2)
    )
    # Calculate error using only city and highway error
    obj_val = err_city**2 + err_hwy**2

    return city_mpg, hwy_mpg, comb_mpg, err_city, err_hwy, err_comb, obj_val


def calculate_error(xs: np.ndarray) -> float:
    """
    Run five_cycle function and return sum of square errors
    """
    veh_dict = deepcopy(veh_base_dict)
    for (setter_fn, _, _), val in zip(PARAM_REGISTRY, xs):
        veh_dict = setter_fn(veh_dict, val)

    veh = fsim.Vehicle.from_pydict(veh_dict, skip_init=False)
    city_mpg, hwy_mpg, comb_mpg, err_city, err_hwy, err_comb, obj_val = run_five_cycle(
        veh
    )

    param_strs = ", ".join(f"{name} = {v:.4g}" for name, v in zip(PARAM_NAMES, xs))
    if SHOW_INTERMEDIATE_OUTPUT:
        print(
            f"  [{param_strs}]  "
            f"city = {city_mpg:.4f} mpg ({100 * err_city:+.2f}%)  "
            f"highway = {hwy_mpg:.4f} mpg ({100 * err_hwy:+.2f}%)  "
            f"combined = {comb_mpg:.4f} mpg ({100 * err_comb:+.2f}%)  "
            f"obj = {obj_val:.6f}"
        )
    return obj_val


# %%
# Main function

if __name__ == "__main__":
    # Print initial info
    print(f"Targets: city = {TARGET_CITY_MPG} mpg, hwy = {TARGET_HWY_MPG} mpg")
    print(f"Optimizing {len(PARAM_REGISTRY)} parameter(s):")
    for name, (lb, ub) in zip(PARAM_NAMES, BOUNDS):
        print(f"  {name}: [{lb:.4g}, {ub:.4g}]")
    print()

    # Optimize
    print("\n── Optimization Output ──────────────────────────────────────────────────")
    if not SHOW_INTERMEDIATE_OUTPUT:
        print("(output suppressed)...")
    result = differential_evolution(
        calculate_error,
        bounds=BOUNDS,
        seed=42,
        tol=1e-4,
        atol=1e-6,
        maxiter=200,
        popsize=5,
        disp=SHOW_INTERMEDIATE_OUTPUT,
        workers=1,
    )

    # Show parameter with asterisk * if parameter is close to either bound
    BOUND_TOL = 0.05  # within 5% of bound
    near_bound = []
    for val, (lb, ub) in zip(result.x, BOUNDS):
        bound_range = ub - lb
        if (val - lb) / bound_range < BOUND_TOL or (ub - val) / bound_range < BOUND_TOL:
            near_bound.append(True)
        else:
            near_bound.append(False)
    asterisks = ["*" if near else "" for near in near_bound]

    # Print results
    print("\n── Optimization Result ──────────────────────────────────────────────────")
    print(f"Message: {result.message}")
    (
        _city_mpg,
        _hwy_mpg,
        _combined_mpg,
        _err_city,
        _err_hwy,
        _err_comb,
        unopt_obj_val,
    ) = run_five_cycle(veh_base)
    print(f"Objective value (unoptimized): {unopt_obj_val:.6f}")
    print(f"Objective value (final): {result.fun:.6f}")
    print("\nOptimal parameters:")
    for name, val, (lb, ub), asterisk in zip(PARAM_NAMES, result.x, BOUNDS, asterisks):
        print(f"  {asterisk}{name}: {val:.6g}  (bounds: [{lb:.4g}, {ub:.4g}])")

    # Final evaluation with logging enabled
    print("\nFinal five_cycle run:")
    veh_dict_final = deepcopy(veh_base_dict)
    for (setter_fn, *_), val in zip(PARAM_REGISTRY, result.x):
        veh_dict_final = setter_fn(veh_dict_final, val)
    veh_final = fsim.Vehicle.from_pydict(veh_dict_final, skip_init=False)
    city_mpg, hwy_mpg, combined_mpg, err_city, err_hwy, err_comb, obj_val = (
        run_five_cycle(veh_final)
    )
    print(f"  city = {city_mpg:.4f} mpg ({100 * err_city:+.2f}%)")
    print(f"  hwy = {hwy_mpg:.4f} mpg ({100 * err_hwy:+.2f}%)")
    print(f"  combined = {combined_mpg:.4f} mpg ({100 * err_comb:+.2f}%)")


# %%
# TODO:
# - [ ] revisit base model - apply script to non-thermal model & run simdrivelabel
# - [ ] sanity check parameter bounds
# - [ ] bug fix for characteristic length - nonzero fuel energy during soak

# full size buses
# - [ ] dust off Altoona pipeline work to get base model
# - goal is to calibrate thermal model to match FleetDNA data
