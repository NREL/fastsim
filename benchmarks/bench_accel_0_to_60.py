#!/usr/bin/env python
"""
Benchmark script: 0-60 mph acceleration tests for FASTSim-3 vehicles.

Loads vehicles from cal_and_val/f3-vehicles/, runs the EPA label fuel economy
calculation (which includes the acceleration test), and prints a table of
simulated 0-60 times alongside published reference values.

Usage:
    pixi run python benchmarks/bench_accel_0_to_60.py
"""

import json
import time
from pathlib import Path

import fastsim as fsim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path to vehicle YAML files (relative to repo root)
VEH_DIR = Path(__file__).resolve().parent.parent / "cal_and_val" / "f3-vehicles"

# Vehicles to test.  Keys = YAML filename stems, values = published 0-60 mph
# reference times in seconds (from zeroto60times.com or manufacturer data).
# Use None when a published number isn't available.
VEHICLES: dict[str, float | None] = {
    "2016 TESLA Model S60 2WD": 5.0,
    "2017 CHEVROLET Bolt": 6.5,
    "2016 FORD Escape 4cyl 2WD": 9.0,
    "2016 TOYOTA Camry 4cyl 2WD": 7.9,
    "2016 TOYOTA Corolla 4cyl 2WD": 9.2,
    "2016 Toyota Prius Two FWD": 10.1,
    "2016 TOYOTA Highlander Hybrid": 7.9,
}


def run_accel_benchmark() -> list[dict]:
    """Run 0-60 acceleration test for each vehicle and return results."""
    results: list[dict] = []

    for veh_name, ref_time in VEHICLES.items():
        yaml_path = VEH_DIR / f"{veh_name}.yaml"
        if not yaml_path.exists():
            print(f"  WARNING: {yaml_path} not found — skipping")
            results.append(
                {
                    "vehicle": veh_name,
                    "sim_0_60_s": None,
                    "ref_0_60_s": ref_time,
                    "delta_s": None,
                    "elapsed_s": None,
                    "error": "file not found",
                }
            )
            continue

        t0 = time.perf_counter()
        try:
            veh = fsim.Vehicle.from_file(str(yaml_path))
            # get_label_fe internally runs the acceleration test (run_accel)
            # and stores the 0-60 time in LabelFe.net_accel
            label_fe = fsim.get_label_fe(veh)
            elapsed = time.perf_counter() - t0

            lfe_dict = json.loads(label_fe.to_json())
            sim_time = lfe_dict["net_accel"]
            delta = (sim_time - ref_time) if ref_time is not None else None

            results.append(
                {
                    "vehicle": veh_name,
                    "sim_0_60_s": sim_time,
                    "ref_0_60_s": ref_time,
                    "delta_s": delta,
                    "elapsed_s": elapsed,
                    "error": None,
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append(
                {
                    "vehicle": veh_name,
                    "sim_0_60_s": None,
                    "ref_0_60_s": ref_time,
                    "delta_s": None,
                    "elapsed_s": elapsed,
                    "error": str(exc),
                }
            )

    return results


def print_table(results: list[dict]) -> None:
    """Pretty-print a results table to stdout."""
    header = f"{'Vehicle':<42} {'Sim 0-60 (s)':>13} {'Ref 0-60 (s)':>13} {'Delta (s)':>10} {'Wall (s)':>9}  {'Status'}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("FASTSim-3  0-60 mph Acceleration Benchmark")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        sim_str = f"{r['sim_0_60_s']:.2f}" if r["sim_0_60_s"] is not None else "N/A"
        ref_str = f"{r['ref_0_60_s']:.1f}" if r["ref_0_60_s"] is not None else "N/A"
        delta_str = f"{r['delta_s']:+.2f}" if r["delta_s"] is not None else "N/A"
        wall_str = f"{r['elapsed_s']:.2f}" if r["elapsed_s"] is not None else "N/A"
        status = "OK" if r["error"] is None else f"ERR: {r['error'][:40]}"
        print(
            f"{r['vehicle']:<42} {sim_str:>13} {ref_str:>13} {delta_str:>10} {wall_str:>9}  {status}"
        )

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"fastsim version: {fsim.__version__}")
    print(f"Vehicle YAML dir: {VEH_DIR}")
    print(f"Vehicles to test: {len(VEHICLES)}")

    results = run_accel_benchmark()
    print_table(results)
