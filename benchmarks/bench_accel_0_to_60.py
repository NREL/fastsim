#!/usr/bin/env python
"""
Benchmark script: 0-60 mph acceleration tests for FASTSim-3 vehicles.

Loads US-market vehicles from cal_and_val/f3-vehicles/, runs the EPA label
fuel economy calculation (which includes the acceleration test), and prints a
table of simulated 0-60 times alongside published reference values, plus a
summary grouped by powertrain type so systematic biases can be spotted.

Reference 0-60 mph times were sourced for the production trim that most
closely matches the FASTSim model's powertrain (peak engine power, motor
power, and battery capacity). Sources are primarily Car & Driver instrumented
tests, with manufacturer specs and zeroto60times.com used as fallbacks.

Usage:
    pixi run python benchmarks/bench_accel_0_to_60.py
"""

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import fastsim as fsim

# Colors for powertrain categories (consistent across all plots)
PT_COLORS = {
    "Conv": "#d62728",   # red
    "HEV":  "#2ca02c",   # green
    "PHEV": "#ff7f0e",   # orange
    "BEV":  "#1f77b4",   # blue
    "Unknown": "#7f7f7f",  # gray
}
PT_ORDER = ["Conv", "HEV", "PHEV", "BEV"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path to vehicle YAML files (relative to repo root)
VEH_DIR = Path(__file__).resolve().parent.parent / "cal_and_val" / "f3-vehicles"

# US-market vehicles only.  Keys = YAML filename stems.
# Values: (published_0_60_s, matched_trim_note)
# Sourced from Car & Driver, MotorTrend, and manufacturer specs for the
# trim that best matches the FASTSim model's powertrain sizing.
VEHICLES: dict[str, tuple[float, str]] = {
    # --- Conventional (ICE) ---
    "2012 Ford Focus":                          (9.0,  "SE 2.0L hatch FWD"), # more like 8.3s
    "2012 Ford Fusion":                         (8.6,  "SE 2.5L FWD"), # 8.9s according to a couple sources
    "2016 AUDI A3 4cyl 2WD":                    (7.2,  "A3 1.8T FWD"),
    "2016 BMW 328d 4cyl 2WD":                   (7.2,  "328d sedan diesel"),
    "2016 CHEVROLET Malibu 4cyl 2WD":           (8.5,  "1.5T LT"),
    "2016 FORD Escape 4cyl 2WD":                (9.0,  "2.5L S FWD"),
    "2016 FORD Explorer 4cyl 2WD":              (8.2,  "2.3L EcoBoost FWD"),
    "2016 HYUNDAI Elantra 4cyl 2WD":            (9.0,  "2.0L SE"),
    "2016 TOYOTA Camry 4cyl 2WD":               (7.9,  "LE 2.5L"),
    "2016 TOYOTA Corolla 4cyl 2WD":             (9.2,  "LE 1.8L"),
    "2017 Toyota Highlander 3.5 L":             (7.3,  "LE V6 FWD"),
    "2020 Chevrolet Colorado 2WD Diesel":       (9.5,  "2.8L Duramax 2WD"),
    # --- HEV ---
    "2016 FORD C-MAX HEV":                      (8.4,  "C-Max Hybrid SE"),
    "2016 KIA Optima Hybrid":                   (9.2,  "EX Hybrid"),
    "2016 TOYOTA Highlander Hybrid":            (7.5,  "Limited Platinum"),
    "2016 Toyota Prius Two FWD":                (10.1, "Prius Two"),
    "2022 Toyota RAV4 Hybrid LE":               (7.4,  "LE Hybrid"),
    "Toyota Corolla Cross Hybrid":              (7.3,  "SE Hybrid"),
    # --- HEV / FCEV (modeled as HEV in FASTSim) ---
    "2016 Hyundai Tucson Fuel Cell":            (12.5, "FCEV base"),
    "Toyota Mirai":                             (9.0,  "Mirai XLE"),
    # --- PHEV ---
    "2016 BMW i3 REx PHEV":                     (8.0,  "i3 REx"),
    "2016 CHEVROLET Volt":                      (8.4,  "Gen2 Premier"),
    "2016 FORD C-MAX (PHEV)":                   (7.9,  "C-Max Energi"), # maybe more like 7.9s
    "2016 HYUNDAI Sonata PHEV":                 (8.5,  "Sonata Plug-In"), # more like 8.5s
    "2017 Prius Prime":                         (10.5, "Prime Plus"),
    # --- BEV ---
    "2016 CHEVROLET Spark EV":                  (7.2,  "Spark EV base"),
    "2016 Leaf 24 kWh":                         (10.2, "Leaf S 24 kWh"),
    "2016 MITSUBISHI i-MiEV":                   (13.4, "ES base"),
    "2016 Nissan Leaf 30 kWh":                  (9.9,  "SV 30 kWh"),
    "2016 TESLA Model S60 2WD":                 (5.5,  "Model S 60 RWD"),
    "2017 CHEVROLET Bolt":                      (6.5,  "Bolt LT"),
    "2021 BMW iX xDrive40":                     (6.1,  "iX xDrive40"),
    "2022 Ford F-150 Lightning 4WD":            (4.0,  "Standard Range Pro 4WD"),
    "2022 MINI Cooper SE Hardtop 2 door":       (6.9,  "Cooper SE"),
    "2022 Tesla Model 3 RWD":                   (5.8,  "Model 3 RWD"),
    "2022 Tesla Model Y RWD":                   (5.8,  "Model Y RWD"),
    "2022 Volvo XC40 Recharge twin":            (4.7,  "Recharge Twin AWD"),
    "2023 Polestar 2 Long range Dual motor":    (4.5,  "LR Dual Motor"),
    "2023 Volvo C40 Recharge":                  (4.5,  "Recharge Twin AWD"),
}


def detect_powertrain(yaml_path: Path) -> str:
    """Return BEV / Conv / HEV / PHEV by inspecting the YAML pt_type tag."""
    with yaml_path.open() as f:
        doc = yaml.safe_load(f)
    pt_type = doc.get("pt_type")
    if isinstance(pt_type, dict) and pt_type:
        return next(iter(pt_type.keys()))
    if isinstance(pt_type, str):
        return pt_type
    return "Unknown"


def run_accel_benchmark() -> list[dict]:
    """Run 0-60 acceleration test for each vehicle and return results."""
    results: list[dict] = []

    for veh_name, (ref_time, trim) in VEHICLES.items():
        yaml_path = VEH_DIR / f"{veh_name}.yaml"
        if not yaml_path.exists():
            print(f"  WARNING: {yaml_path} not found — skipping")
            results.append({
                "vehicle": veh_name, "trim": trim, "powertrain": "Unknown",
                "sim_0_60_s": None, "ref_0_60_s": ref_time,
                "delta_s": None, "elapsed_s": None, "error": "file not found",
            })
            continue

        powertrain = detect_powertrain(yaml_path)
        t0 = time.perf_counter()
        try:
            veh = fsim.Vehicle.from_file(str(yaml_path))
            label_fe = fsim.get_label_fe(veh)
            elapsed = time.perf_counter() - t0
            lfe_dict = json.loads(label_fe.to_json())
            sim_time = lfe_dict["net_accel"]
            results.append({
                "vehicle": veh_name, "trim": trim, "powertrain": powertrain,
                "sim_0_60_s": sim_time, "ref_0_60_s": ref_time,
                "delta_s": sim_time - ref_time, "elapsed_s": elapsed, "error": None,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append({
                "vehicle": veh_name, "trim": trim, "powertrain": powertrain,
                "sim_0_60_s": None, "ref_0_60_s": ref_time,
                "delta_s": None, "elapsed_s": elapsed, "error": str(exc),
            })

    return results


def print_table(results: list[dict]) -> None:
    """Pretty-print a per-vehicle results table grouped by powertrain."""
    header = (f"{'Vehicle':<42} {'PT':<5} {'Sim':>6} {'Ref':>6} "
              f"{'Δ (s)':>8} {'Δ %':>7}  Status")
    sep = "-" * len(header)
    print("\n" + sep)
    print("FASTSim-3  0-60 mph Acceleration Benchmark  (US-market vehicles)")
    print(sep)
    print(header)
    print(sep)

    # Group output by powertrain
    by_pt: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_pt[r["powertrain"]].append(r)

    for pt in ("Conv", "HEV", "PHEV", "BEV", "Unknown"):
        rows = by_pt.get(pt, [])
        if not rows:
            continue
        print(f"--- {pt} ({len(rows)}) ".ljust(len(header), "-"))
        for r in rows:
            sim = f"{r['sim_0_60_s']:.2f}" if r["sim_0_60_s"] is not None else "N/A"
            ref = f"{r['ref_0_60_s']:.1f}"
            delta = (f"{r['delta_s']:+.2f}"
                     if r["delta_s"] is not None else "N/A")
            pct = (f"{100.0 * r['delta_s'] / r['ref_0_60_s']:+6.1f}%"
                   if r["delta_s"] is not None else "N/A")
            status = "OK" if r["error"] is None else f"ERR: {r['error'][:30]}"
            print(f"{r['vehicle']:<42} {pt:<5} {sim:>6} {ref:>6} "
                  f"{delta:>8} {pct:>7}  {status}")

    print(sep + "\n")


def print_summary(results: list[dict]) -> None:
    """Aggregate stats per powertrain type."""
    print("Summary by powertrain type")
    print("-" * 78)
    print(f"{'PT':<6} {'N':>3} {'Mean Δ (s)':>12} {'Median Δ (s)':>14} "
          f"{'MAE (s)':>10} {'Mean Δ %':>10}")
    print("-" * 78)

    by_pt: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["delta_s"] is not None:
            by_pt[r["powertrain"]].append(r)

    overall_deltas = []
    overall_pcts = []
    for pt in ("Conv", "HEV", "PHEV", "BEV"):
        rows = by_pt.get(pt, [])
        if not rows:
            continue
        deltas = [r["delta_s"] for r in rows]
        pcts = [100.0 * r["delta_s"] / r["ref_0_60_s"] for r in rows]
        overall_deltas.extend(deltas)
        overall_pcts.extend(pcts)
        print(f"{pt:<6} {len(rows):>3} "
              f"{statistics.mean(deltas):>+12.2f} "
              f"{statistics.median(deltas):>+14.2f} "
              f"{statistics.mean(abs(d) for d in deltas):>10.2f} "
              f"{statistics.mean(pcts):>+9.1f}%")

    if overall_deltas:
        print("-" * 78)
        print(f"{'ALL':<6} {len(overall_deltas):>3} "
              f"{statistics.mean(overall_deltas):>+12.2f} "
              f"{statistics.median(overall_deltas):>+14.2f} "
              f"{statistics.mean(abs(d) for d in overall_deltas):>10.2f} "
              f"{statistics.mean(overall_pcts):>+9.1f}%")
    print()
    print("Sign convention: Δ = sim - reference. Positive Δ → FASTSim is slower")
    print("                 (predicts a longer 0-60 time) than the published value.")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(results: list[dict], outdir: Path) -> None:
    """Generate scatter + bar charts comparing sim vs. published 0-60 times."""
    ok = [r for r in results if r["sim_0_60_s"] is not None]
    if not ok:
        print("No successful results to plot.")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Plot 1: Sim vs. Published scatter, colored by powertrain ----
    fig, ax = plt.subplots(figsize=(8, 8))
    all_vals = []
    for pt in PT_ORDER:
        rows = [r for r in ok if r["powertrain"] == pt]
        if not rows:
            continue
        ref = [r["ref_0_60_s"] for r in rows]
        sim = [r["sim_0_60_s"] for r in rows]
        all_vals.extend(ref + sim)
        ax.scatter(ref, sim, s=70, c=PT_COLORS[pt],
                   edgecolors="black", linewidths=0.5,
                   label=f"{pt} (n={len(rows)})", alpha=0.85, zorder=3)

    lo = min(all_vals) - 0.5
    hi = max(all_vals) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6,
            label="y = x (perfect)", zorder=1)
    # ±10% bands
    xs = np.linspace(lo, hi, 50)
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color="gray", alpha=0.12,
                    label="±10% band", zorder=0)

    ax.set_xlabel("Published 0-60 mph (s)")
    ax.set_ylabel("FASTSim-3 simulated 0-60 mph (s)")
    ax.set_title("FASTSim-3 vs. Published 0-60 mph times")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    p1 = outdir / "accel_scatter.png"
    fig.savefig(p1, dpi=150)
    print(f"  saved: {p1}")

    # ---- Plot 2: Side-by-side bar chart of sim vs. ref per vehicle ----
    # Sort within each powertrain by reference time, then concatenate by PT
    ordered: list[dict] = []
    for pt in PT_ORDER:
        ordered.extend(sorted(
            (r for r in ok if r["powertrain"] == pt),
            key=lambda r: r["ref_0_60_s"],
        ))

    n = len(ordered)
    idx = np.arange(n)
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(12, n * 0.35), 7))
    ref_vals = [r["ref_0_60_s"] for r in ordered]
    sim_vals = [r["sim_0_60_s"] for r in ordered]
    bar_colors = [PT_COLORS[r["powertrain"]] for r in ordered]

    ax.bar(idx - width / 2, ref_vals, width,
           color="lightgray", edgecolor="black", linewidth=0.5,
           label="Published")
    ax.bar(idx + width / 2, sim_vals, width,
           color=bar_colors, edgecolor="black", linewidth=0.5,
           label="FASTSim-3 (color = powertrain)")

    ax.set_xticks(idx)
    ax.set_xticklabels([r["vehicle"] for r in ordered],
                       rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("0-60 mph time (s)")
    ax.set_title("Per-vehicle 0-60 mph: Published vs. FASTSim-3")
    ax.grid(True, axis="y", alpha=0.3)

    # Build a combined legend with one swatch per powertrain + Published
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="lightgray", edgecolor="black",
                     label="Published")]
    handles += [Patch(facecolor=PT_COLORS[pt], edgecolor="black",
                      label=f"FASTSim - {pt}")
                for pt in PT_ORDER if any(r["powertrain"] == pt for r in ordered)]
    ax.legend(handles=handles, loc="upper left")

    # Vertical separators between powertrain groups
    cum = 0
    for pt in PT_ORDER:
        count = sum(1 for r in ordered if r["powertrain"] == pt)
        if count == 0:
            continue
        cum += count
        if cum < n:
            ax.axvline(cum - 0.5, color="black", lw=0.6, alpha=0.4)
    fig.tight_layout()
    p2 = outdir / "accel_bars_per_vehicle.png"
    fig.savefig(p2, dpi=150)
    print(f"  saved: {p2}")

    # ---- Plot 3: Mean Δ (s) and Mean Δ% summary bars by powertrain ----
    by_pt: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_pt[r["powertrain"]].append(r)

    pts = [pt for pt in PT_ORDER if by_pt.get(pt)]
    mean_delta = [statistics.mean(r["delta_s"] for r in by_pt[pt]) for pt in pts]
    mean_pct = [statistics.mean(100.0 * r["delta_s"] / r["ref_0_60_s"]
                                for r in by_pt[pt]) for pt in pts]
    counts = [len(by_pt[pt]) for pt in pts]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [PT_COLORS[pt] for pt in pts]

    bars = axL.bar(pts, mean_delta, color=colors,
                   edgecolor="black", linewidth=0.5)
    axL.axhline(0, color="black", lw=0.8)
    axL.set_ylabel("Mean Δ = sim − published (s)")
    axL.set_title("Mean absolute error by powertrain")
    axL.grid(True, axis="y", alpha=0.3)
    for bar, val, n_ in zip(bars, mean_delta, counts):
        axL.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.05 if val >= 0 else -0.05),
                 f"{val:+.2f}\n(n={n_})",
                 ha="center",
                 va="bottom" if val >= 0 else "top",
                 fontsize=9)

    bars = axR.bar(pts, mean_pct, color=colors,
                   edgecolor="black", linewidth=0.5)
    axR.axhline(0, color="black", lw=0.8)
    axR.set_ylabel("Mean Δ% = (sim − published) / published × 100")
    axR.set_title("Mean percent error by powertrain")
    axR.grid(True, axis="y", alpha=0.3)
    for bar, val, n_ in zip(bars, mean_pct, counts):
        axR.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.5 if val >= 0 else -0.5),
                 f"{val:+.1f}%\n(n={n_})",
                 ha="center",
                 va="bottom" if val >= 0 else "top",
                 fontsize=9)

    fig.suptitle("FASTSim-3  0-60 mph error summary by powertrain  "
                 "(positive → sim slower than published)")
    fig.tight_layout()
    p3 = outdir / "accel_summary_by_powertrain.png"
    fig.savefig(p3, dpi=150)
    print(f"  saved: {p3}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"fastsim version: {fsim.__version__}")
    print(f"Vehicle YAML dir: {VEH_DIR}")
    print(f"Vehicles to test: {len(VEHICLES)}")

    results = run_accel_benchmark()
    print_table(results)
    print_summary(results)

    print("Generating plots...")
    make_plots(results, Path(__file__).resolve().parent / "plots")
    plt.show()
