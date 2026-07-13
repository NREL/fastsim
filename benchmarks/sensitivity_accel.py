#!/usr/bin/env python
"""
Sensitivity analysis: effect of FC ramp lag and wheel friction coefficient
on 0-60 mph acceleration times for FASTSim-3 reference vehicles.

Usage:
    pixi run python benchmarks/sensitivity_accel.py
"""

import json
from pathlib import Path

import yaml
import fastsim as fsim

VEH_DIR = Path(__file__).resolve().parent.parent / "cal_and_val" / "f3-vehicles"

VEHICLES: dict[str, float] = {
    "2016 TESLA Model S60 2WD": 5.0,
    "2017 CHEVROLET Bolt": 6.5,
    "2016 FORD Escape 4cyl 2WD": 9.0,
    "2016 TOYOTA Camry 4cyl 2WD": 7.9,
    "2016 TOYOTA Corolla 4cyl 2WD": 9.2,
    "2016 Toyota Prius Two FWD": 10.1,
    "2016 TOYOTA Highlander Hybrid": 7.9,
}

# Sweep values
RAMP_LAG_VALUES = [1.0, 2.0, 3.0, 4.0, 6.0]  # seconds
WHEEL_FRIC_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # dimensionless


def get_0_to_60(veh: fsim.Vehicle) -> float | None:
    """Run label FE and extract the 0-60 time."""
    try:
        lfe = fsim.get_label_fe(veh)
        return json.loads(lfe.to_json())["net_accel"]
    except Exception:
        return None


def modify_vehicle(veh_dict: dict, ramp_lag: float | None = None, wheel_fric: float | None = None) -> dict:
    """Return a modified copy of the vehicle dict."""
    import copy
    d = copy.deepcopy(veh_dict)

    # Modify wheel friction coefficient
    if wheel_fric is not None:
        d["chassis"]["wheel_fric_coef"] = wheel_fric

    # Modify FC ramp lag (only for vehicles that have an FC)
    if ramp_lag is not None:
        pt = d.get("pt_type", {})
        # Conventional vehicle
        if "Conv" in pt:
            pt["Conv"]["fc"]["pwr_ramp_lag_seconds"] = ramp_lag
        # HEV
        elif "HEV" in pt:
            pt["HEV"]["fc"]["pwr_ramp_lag_seconds"] = ramp_lag

    return d


def load_veh_dict(name: str) -> dict:
    """Load vehicle YAML as a dict."""
    yaml_path = VEH_DIR / f"{name}.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def veh_from_dict(d: dict) -> fsim.Vehicle:
    """Create a Vehicle from a dict via YAML round-trip."""
    return fsim.Vehicle.from_yaml(yaml.dump(d))


def has_fc(veh_dict: dict) -> bool:
    """Check if vehicle has a fuel converter."""
    pt = veh_dict.get("pt_type", {})
    return "Conv" in pt or "HEV" in pt


def run_ramp_lag_sweep():
    """Sweep FC ramp lag values and print results."""
    print("=" * 100)
    print("SENSITIVITY: FC Ramp Lag (seconds) vs 0-60 mph time")
    print("=" * 100)

    # Header
    lag_strs = [f"{v:>6.1f}s" for v in RAMP_LAG_VALUES]
    header = f"{'Vehicle':<42} {'Ref':>6}  " + "  ".join(lag_strs) + "  {'Default lag'}"
    print(f"\n{'Vehicle':<42} {'Ref':>6}  " + "  ".join(lag_strs) + f"  {'Orig lag':>10}")
    print("-" * (55 + 8 * len(RAMP_LAG_VALUES) + 12))

    for veh_name, ref_time in VEHICLES.items():
        veh_dict = load_veh_dict(veh_name)

        if not has_fc(veh_dict):
            # BEV — no FC ramp lag to sweep
            base_veh = veh_from_dict(veh_dict)
            t = get_0_to_60(base_veh)
            t_str = f"{t:.2f}" if t else "N/A"
            print(f"{veh_name:<42} {ref_time:>6.1f}  " + "  ".join([f"{'  ---':>8}"] * len(RAMP_LAG_VALUES)) + f"  {'(BEV)':>10}")
            continue

        # Get the original ramp lag
        pt = veh_dict.get("pt_type", {})
        if "Conv" in pt:
            orig_lag = pt["Conv"]["fc"]["pwr_ramp_lag_seconds"]
        else:
            orig_lag = pt["HEV"]["fc"]["pwr_ramp_lag_seconds"]

        results = []
        for lag in RAMP_LAG_VALUES:
            mod = modify_vehicle(veh_dict, ramp_lag=lag)
            veh = veh_from_dict(mod)
            t = get_0_to_60(veh)
            results.append(t)

        result_strs = [f"{t:>8.2f}" if t else f"{'N/A':>8}" for t in results]
        print(f"{veh_name:<42} {ref_time:>6.1f}  " + "  ".join(result_strs) + f"  {orig_lag:>8.1f}s")

    print()


def run_wheel_fric_sweep():
    """Sweep wheel friction coefficient values and print results."""
    print("=" * 100)
    print("SENSITIVITY: Wheel Friction Coefficient vs 0-60 mph time")
    print("=" * 100)

    # Header
    fric_strs = [f"{v:>6.1f}" for v in WHEEL_FRIC_VALUES]
    print(f"\n{'Vehicle':<42} {'Ref':>6}  " + "  ".join([f"{'μ='+s:>8}" for s in fric_strs]))
    print("-" * (55 + 8 * len(WHEEL_FRIC_VALUES)))

    for veh_name, ref_time in VEHICLES.items():
        veh_dict = load_veh_dict(veh_name)

        results = []
        for fric in WHEEL_FRIC_VALUES:
            mod = modify_vehicle(veh_dict, wheel_fric=fric)
            veh = veh_from_dict(mod)
            t = get_0_to_60(veh)
            results.append(t)

        result_strs = [f"{t:>8.2f}" if t else f"{'N/A':>8}" for t in results]
        print(f"{veh_name:<42} {ref_time:>6.1f}  " + "  ".join(result_strs))

    print()


if __name__ == "__main__":
    print(f"fastsim version: {fsim.__version__}\n")
    run_ramp_lag_sweep()
    run_wheel_fric_sweep()
