#!/usr/bin/env python
"""
Fuel economy comparison script: runs get_label_fe for a set of vehicles and
outputs results as JSON. Designed to be run in two different worktrees (baseline
vs. current) to compare fuel consumption impact of code/model changes.

Usage:
    # In baseline worktree:
    pixi run python benchmarks/bench_fe_comparison.py --output baseline_fe.json

    # In current worktree:
    pixi run python benchmarks/bench_fe_comparison.py --output current_fe.json

    # Compare:
    pixi run python benchmarks/bench_fe_comparison.py --compare baseline_fe.json current_fe.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import fastsim as fsim

# ---------------------------------------------------------------------------
# Vehicles to test (broad cross-section of powertrains)
# ---------------------------------------------------------------------------
VEHICLES = [
    # Conv
    "2012 Ford Focus",
    "2016 AUDI A3 4cyl 2WD",
    "2016 CHEVROLET Malibu 4cyl 2WD",
    "2016 FORD Escape 4cyl 2WD",
    "2016 TOYOTA Camry 4cyl 2WD",
    "2016 TOYOTA Corolla 4cyl 2WD",
    "2017 Toyota Highlander 3.5 L",
    "2020 Chevrolet Colorado 2WD Diesel",
    # HEV
    "2016 FORD C-MAX HEV",
    "2016 KIA Optima Hybrid",
    "2016 TOYOTA Highlander Hybrid",
    "2016 Toyota Prius Two FWD",
    "2022 Toyota RAV4 Hybrid LE",
    "Toyota Corolla Cross Hybrid",
    "2016 Hyundai Tucson Fuel Cell",
    "Toyota Mirai",
    # PHEV
    "2016 BMW i3 REx PHEV",
    "2016 CHEVROLET Volt",
    "2016 FORD C-MAX (PHEV)",
    "2016 HYUNDAI Sonata PHEV",
    "2017 Prius Prime",
    # BEV
    "2016 CHEVROLET Spark EV",
    "2016 Leaf 24 kWh",
    "2016 TESLA Model S60 2WD",
    "2017 CHEVROLET Bolt",
    "2022 Ford F-150 Lightning 4WD",
    "2022 Tesla Model 3 RWD",
    "2022 Tesla Model Y RWD",
    "2022 Volvo XC40 Recharge twin",
    "2023 Polestar 2 Long range Dual motor",
]


def run_label_fe(veh_dir: Path) -> list[dict]:
    """Run get_label_fe for each vehicle and collect results."""
    results = []
    for name in VEHICLES:
        yaml_path = veh_dir / f"{name}.yaml"
        if not yaml_path.exists():
            results.append({"vehicle": name, "error": "file not found"})
            continue

        t0 = time.perf_counter()
        try:
            veh = fsim.Vehicle.from_file(str(yaml_path))
            label_fe = fsim.get_label_fe(veh)
            elapsed = time.perf_counter() - t0
            lfe = json.loads(label_fe.to_json())

            results.append({
                "vehicle": name,
                "adj_comb_mpgge": lfe.get("adj_comb_mpgge"),
                "adj_udds_mpgge": lfe.get("adj_udds_mpgge"),
                "adj_hwfet_mpgge": lfe.get("adj_hwfet_mpgge"),
                "adj_comb_kwh_per_mi": lfe.get("adj_comb_kwh_per_mi"),
                "net_accel": lfe.get("net_accel"),
                "net_range_miles": lfe.get("net_range_miles"),
                "elapsed_s": round(elapsed, 2),
                "error": None,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append({
                "vehicle": name,
                "elapsed_s": round(elapsed, 2),
                "error": str(exc)[:200],
            })

    return results


def compare(baseline_path: Path, current_path: Path) -> None:
    """Load two JSON result files and print a comparison table."""
    with open(baseline_path) as f:
        baseline = {r["vehicle"]: r for r in json.load(f)}
    with open(current_path) as f:
        current = {r["vehicle"]: r for r in json.load(f)}

    print()
    print("=" * 100)
    print("Fuel Economy Comparison: Baseline vs. Current")
    print("=" * 100)
    header = (f"{'Vehicle':<42} {'Base MPGge':>10} {'Curr MPGge':>10} "
              f"{'Δ MPGge':>9} {'Δ%':>7}  {'Base 0-60':>9} {'Curr 0-60':>9}")
    print(header)
    print("-" * 100)

    deltas_pct = []
    for name in VEHICLES:
        b = baseline.get(name, {})
        c = current.get(name, {})

        b_mpg = b.get("adj_comb_mpgge")
        c_mpg = c.get("adj_comb_mpgge")
        b_accel = b.get("net_accel")
        c_accel = c.get("net_accel")

        # For BEVs, use kWh/mi if mpgge is None
        if b_mpg is None:
            b_mpg = b.get("adj_comb_kwh_per_mi")
            c_mpg = c.get("adj_comb_kwh_per_mi") if c else None
            unit = "kWh/mi"
        else:
            unit = "MPGge"

        if b_mpg is not None and c_mpg is not None:
            delta = c_mpg - b_mpg
            pct = 100.0 * delta / b_mpg if b_mpg != 0 else 0
            deltas_pct.append(pct)
            mpg_b_str = f"{b_mpg:.1f}"
            mpg_c_str = f"{c_mpg:.1f}"
            delta_str = f"{delta:+.2f}"
            pct_str = f"{pct:+.1f}%"
        else:
            mpg_b_str = "ERR" if b.get("error") else "N/A"
            mpg_c_str = "ERR" if c.get("error") else "N/A"
            delta_str = "—"
            pct_str = "—"

        accel_b_str = f"{b_accel:.2f}" if b_accel is not None else "ERR"
        accel_c_str = f"{c_accel:.2f}" if c_accel is not None else "ERR"

        print(f"{name:<42} {mpg_b_str:>10} {mpg_c_str:>10} "
              f"{delta_str:>9} {pct_str:>7}  {accel_b_str:>9} {accel_c_str:>9}")

    print("-" * 100)
    if deltas_pct:
        import statistics
        print(f"{'SUMMARY':<42} {'':>10} {'':>10} "
              f"{'':>9} {statistics.mean(deltas_pct):>+6.2f}%")
        print(f"  Mean FE change: {statistics.mean(deltas_pct):+.2f}%  |  "
              f"Median: {statistics.median(deltas_pct):+.2f}%  |  "
              f"Max: {max(deltas_pct):+.2f}%  |  Min: {min(deltas_pct):+.2f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FE comparison tool")
    parser.add_argument("--output", "-o", type=Path,
                        help="Run label FE and save results to this JSON file")
    parser.add_argument("--compare", "-c", nargs=2, type=Path, metavar=("BASELINE", "CURRENT"),
                        help="Compare two result JSON files")
    parser.add_argument("--veh-dir", type=Path, default=None,
                        help="Override vehicle YAML directory")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    elif args.output:
        veh_dir = args.veh_dir or (
            Path(__file__).resolve().parent.parent / "cal_and_val" / "f3-vehicles"
        )
        print(f"fastsim version: {fsim.__version__}")
        print(f"Vehicle dir: {veh_dir}")
        print(f"Vehicles: {len(VEHICLES)}")
        print()

        results = run_label_fe(veh_dir)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        ok = sum(1 for r in results if r.get("error") is None)
        err = len(results) - ok
        print(f"\nDone: {ok} succeeded, {err} failed")
        print(f"Results saved to: {args.output}")
    else:
        parser.print_help()
