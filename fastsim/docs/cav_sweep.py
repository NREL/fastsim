import sys
from pathlib import Path
import csv
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import fastsim
from fastsim.tests.test_coasting import make_coasting_plot


DO_SHOW = False
FRACTION_EXTENDED_TIME = 0.25
ABSOLUTE_EXTENDED_TIME_S = 20.0 # 180.0
OUTPUT_DIR = Path(__file__).parent.absolute() / 'test_output'
MIN_ECO_CRUISE_TARGET_SPEED_m_per_s = 8.0
ECO_COAST_ALLOW_PASSING = False
RESAMPLE_TO_1HZ = True
CSV_KEYS = [
    'powertrain', 'cycle', 'veh',
    'use:base (kWh/mi)', 'use:eco-coast (kWh/mi)', 'use:eco-cruise (kWh/mi)', 'use:all-eco (kWh/mi)',
    'savings:eco-coast (%)', 'savings:eco-cruise (%)', 'savings:all-eco (%)',
    'dist:base (mi)', 'dist:eco-coast (mi)', 'dist:eco-cruise (mi)', 'dist:all-eco (mi)',
    'dist-short:eco-coast (%)', 'dist-short:eco-cruise (%)', 'dist-short:all-eco (%)',
    'soc-delta:base', 'soc-delta:eco-coast', 'soc-delta:eco-cruise', 'soc-delta:all-eco',
]


def make_distance_by_time_plot(cyc0, cyc, save_file=None, do_show=False):
    (fig, ax) = plt.subplots()
    ax.plot(cyc0.time_s, np.cumsum(cyc0.dist_m), 'gray', label='lead')
    ax.plot(cyc.time_s, np.cumsum(cyc.dist_m), 'b-', lw=2, label='cav')
    ax.plot(cyc.time_s, np.cumsum(cyc.dist_m), 'r.', ms=1)
    ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.legend(loc=0)
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=300)
    if do_show:
        plt.show()
    plt.close()


def make_debug_plot(sd: fastsim.simdrive.SimDrive, save_file:Optional[str]=None, do_show:bool=False):
    """
    """
    (fig, axs) = plt.subplots(nrows=2)
    axs[0].plot(sd.cyc0.time_s, sd.cyc0.mps, 'gray', lw=2.5, label='lead')
    axs[0].plot(sd.cyc.time_s, sd.cyc.mps, 'b-', lw=2, label='cav')
    axs[0].plot(sd.cyc.time_s, sd.cyc.mps, 'r.', ms=1)
    axs[0].set_xlabel('Elapsed Time (s)')
    axs[0].set_ylabel('Speed (m/s)')
    axs[0].legend(loc=0)
    ax2 = axs[1].twinx()
    color = 'tab:red'
    ax2.plot(sd.cyc.time_s, sd.impose_coast, 'r.', ms=1, label='impose coast')
    ax2.set_ylabel('Impose Coast', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(False)
    color = 'tab:blue'
    axs[1].plot(sd.cyc.time_s, sd.coast_delay_index.tolist(), 'b.', lw=2, label='coast delay')
    axs[1].set_ylabel('Coast Delay', color=color)
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].grid(False)
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=300)
    if do_show:
        plt.show()
    plt.close()


def make_save_file(prefix, postfix, save_dir=None, use_rust=False):
    if save_dir is not None:
        prefix_addition = ''
        if use_rust:
            prefix_addition = '_rust'
        return save_dir / f'{prefix}{prefix_addition}_{postfix}'
    return None


PREMADE_CYCLES = {
    'trapz': fastsim.cycle.Cycle.from_dict(
        fastsim.cycle.resample(
            fastsim.cycle.make_cycle(
                ts=[0.0, 10.0, 30.0, 34.0, 40.0],
                vs=[0.0, 10.0, 10.0, 0.0, 0.0],
            ),
            new_dt=1.0
        )
    ),
    'trapz-x2': fastsim.cycle.Cycle.from_dict(
        fastsim.cycle.resample(
            fastsim.cycle.make_cycle(
                ts=[0.0, 10.0, 30.0, 34.0, 40.0, 50.0, 70.0, 74.0, 80.0],
                vs=[0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
            ),
            new_dt=1.0
        )
    ),
    'stacked-trapz': fastsim.cycle.Cycle.from_dict(
        fastsim.cycle.resample(
            fastsim.cycle.make_cycle(
                ts=[0.0, 10.0, 20.0, 24.0, 50.0, 54.0, 60.0],
                vs=[0.0, 20.0, 20.0, 8.0, 8.0, 0.0, 0.0],
            ),
            new_dt=1.0
        )
    ),
}


def load_cycle(cyc_name: str, use_rust: bool=False) -> fastsim.cycle.Cycle:
    """
    Load the given cycle and return
    """
    if cyc_name in PREMADE_CYCLES:
        raw_cycle = PREMADE_CYCLES.get(cyc_name)
    else:
        raw_cycle = fastsim.cycle.Cycle.from_file(cyc_name)
        if RESAMPLE_TO_1HZ:
            raw_cycle = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    raw_cycle.get_cyc_dict(),
                    new_dt=1.0,
                    hold_keys_next={'grade'},
                )
            )
    return fastsim.cycle.extend_cycle(
        raw_cycle,
        time_fraction=FRACTION_EXTENDED_TIME,
        absolute_time_s=ABSOLUTE_EXTENDED_TIME_S,
        use_rust=use_rust
    )


def create_simdrive(cyc: fastsim.cycle.Cycle, veh: fastsim.vehicle.Vehicle, use_rust:bool=False) -> fastsim.simdrive.SimDrive:
    if use_rust:
        return fastsim.simdrive.RustSimDrive(cyc.to_rust(), veh.to_rust())
    return fastsim.simdrive.SimDrive(cyc, veh)


def no_eco_driving(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None, use_rust=False, verbose=True):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name, use_rust=use_rust)
    sim = create_simdrive(cyc, veh, use_rust=use_rust)
    sim.sim_drive(init_soc=init_soc)
    if verbose:
        print(f"NO ECO-DRIVING: {sim.mpgge:.3f} mpg", flush=True)
    if save_dir is not None:
        make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'base.png', save_dir, use_rust), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
        make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'base_dist_by_time.png', save_dir, use_rust))
        make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'base_debug.png', save_dir, use_rust))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sum(sim.dist_mi),
        sum(sim.dist_mi),
        sim.soc[-1] - sim.soc[0]
    )


def eco_coast(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None, use_rust=False, verbose=True):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name, use_rust=use_rust)
    sim = create_simdrive(cyc, veh, use_rust=use_rust)
    params = sim.sim_params
    if use_rust:
        params.reset_orphaned()
    params.coast_allow = True
    params.coast_allow_passing = ECO_COAST_ALLOW_PASSING
    params.coast_start_speed_m_per_s = -1.0
    params.coast_time_horizon_for_adjustment_s = 20.0
    sim.sim_params = params
    sim.sim_drive(init_soc=init_soc)
    if verbose:
        print(f"ECO-COAST: {sim.mpgge:.3f} mpg", flush=True)
    if save_dir is not None:
        make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocoast.png', save_dir, use_rust), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
        make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocoast_dist_by_time.png', save_dir, use_rust))
        make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'ecocoast_debug.png', save_dir, use_rust))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sum(sim.dist_mi),
        sum(sim.dist_mi),
        sim.soc[-1] - sim.soc[0]
    )


def eco_cruise(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None, blend_factor=1.0, use_rust=False, verbose=True):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name)
    sim = create_simdrive(cyc, veh, use_rust=use_rust)
    # NOTE: load_cycle already extends the cycle in time so extend_fraction is not needed
    sim.activate_eco_cruise(
        by_microtrip=True,
        extend_fraction=0.0,
        blend_factor=blend_factor,
        min_target_speed_m_per_s=MIN_ECO_CRUISE_TARGET_SPEED_m_per_s)
    # Eco-cruise parameters
    # NOTE: activate_eco_cruise() already sets idm params. To set
    # them to values # other than the default, they must be modified
    # AFTER a call # to activate_eco_cruise()
    params = sim.sim_params
    if use_rust:
        params.reset_orphaned()
    params.idm_accel_m_per_s2 = 0.5
    params.idm_decel_m_per_s2 = 2.5
    params.idm_dt_headway_s = 2.0
    params.idm_minimum_gap_m = 10.0
    sim.sim_params = params
    # Initialize Electric Drive System
    sim.sim_drive(init_soc=init_soc)
    if verbose:
        print(f"ECO-CRUISE: {sim.mpgge:.3f} mpg", flush=True)
    if save_dir is not None:
        make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocruise.png', save_dir, use_rust), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
        make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocruise_dist_by_time.png', save_dir, use_rust))
        make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'ecocruise_debug.png', save_dir, use_rust))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sum(sim.dist_mi),
        sum(sim.dist_mi),
        sim.soc[-1] - sim.soc[0]
    )

def eco_coast_and_cruise(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None, blend_factor=1.0, use_rust=False, verbose=True):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name, use_rust=use_rust)
    sim = create_simdrive(cyc, veh, use_rust=use_rust)
    # NOTE: load_cycle already extends the cycle in time so extend_fraction is not needed
    sim.activate_eco_cruise(
        by_microtrip=True,
        extend_fraction=0.0,
        blend_factor=blend_factor,
        min_target_speed_m_per_s=MIN_ECO_CRUISE_TARGET_SPEED_m_per_s)
    params = sim.sim_params
    if use_rust:
        params.reset_orphaned()
    # Eco-coast parameters
    params.coast_allow = True
    params.coast_allow_passing = False
    params.coast_start_speed_m_per_s = -1.0
    params.coast_time_horizon_for_adjustment_s = 20.0
    # Eco-cruise parameters
    # NOTE: activate_eco_cruise() already sets idm params. To set
    # them to values # other than the default, they must be modified
    # AFTER a call # to activate_eco_cruise()
    params.idm_accel_m_per_s2 = 0.5
    params.idm_decel_m_per_s2 = 2.5
    params.idm_dt_headway_s = 2.0
    params.idm_minimum_gap_m = 10.0
    sim.sim_params = params
    # Initialize Electric Drive System
    sim.sim_drive(init_soc=init_soc)
    if verbose:
        print(f"ECO-COAST + ECO-CRUISE: {sim.mpgge:.3f} mpg", flush=True)
    if save_dir is not None:
        make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'alleco.png', save_dir, use_rust), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
        make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'alleco_dist_by_time.png', save_dir, use_rust))
        make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'alleco_debug.png', save_dir, use_rust))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sum(sim.dist_mi),
        sum(sim.dist_mi),
        sim.soc[-1] - sim.soc[0]
    )


def calc_percentage(base, other):
    return (base - other) * 100.0 / base


def run_for_powertrain(save_dir, outputs, cyc_name, veh, powertrain, init_soc=None, do_show=None, use_rust=False, verbose=True, maneuver=None):
    output = {'powertrain': powertrain, 'cycle': cyc_name, 'veh': veh.scenario_name}
    args = {
        'init_soc': init_soc,
        'cyc_name': cyc_name,
        'do_show': do_show if do_show is not None else DO_SHOW,
        'use_rust': use_rust,
        'verbose': verbose,
    }
    tag = f'{cyc_name}_{powertrain}'
    if maneuver is None or maneuver == "base":
        (output['use:base (kWh/mi)'], output['dist:base (mi)'], output['soc-delta:base']) = no_eco_driving(veh, save_dir=save_dir, tag=tag, **args)
    if maneuver is None or maneuver == "eco-approach" or maneuver == "eco-coast":
        (output['use:eco-coast (kWh/mi)'], output['dist:eco-coast (mi)'], output['soc-delta:eco-coast']) = eco_coast(veh, save_dir=save_dir, tag=tag, **args)
    if maneuver is None or maneuver == "eco-cruise":
        (output['use:eco-cruise (kWh/mi)'], output['dist:eco-cruise (mi)'], output['soc-delta:eco-cruise']) = eco_cruise(veh, save_dir=save_dir, tag=tag, **args)
    if maneuver is None or maneuver == "all-eco":
        (output['use:all-eco (kWh/mi)'], output['dist:all-eco (mi)'], output['soc-delta:all-eco']) = eco_coast_and_cruise(veh, save_dir=save_dir, tag=tag, **args)
    if maneuver is None:
        output['savings:eco-coast (%)'] = calc_percentage(output['use:base (kWh/mi)'], output['use:eco-coast (kWh/mi)'])
        output['savings:eco-cruise (%)'] = calc_percentage(output['use:base (kWh/mi)'], output['use:eco-cruise (kWh/mi)'])
        output['savings:all-eco (%)'] = calc_percentage(output['use:base (kWh/mi)'], output['use:all-eco (kWh/mi)'])
        output['dist-short:eco-coast (%)'] = calc_percentage(output['dist:base (mi)'], output['dist:eco-coast (mi)'])
        output['dist-short:eco-cruise (%)'] = calc_percentage(output['dist:base (mi)'], output['dist:eco-cruise (mi)'])
        output['dist-short:all-eco (%)'] = calc_percentage(output['dist:base (mi)'], output['dist:all-eco (mi)'])
    outputs.append(output)


def main(cycle_name=None, powertrain=None, do_show=None, use_rust=False, verbose=True, save_dir=None, maneuver=None):
    """
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    cyc_names = [cycle_name] if cycle_name is not None else [
        "hwfet", "udds", "us06", "NREL13", "trapz", "trapz-x2", "stacked-trapz", "TSDC_tripno_42648_cycle",
    ]
    outputs = []
    for cyc_name in cyc_names:
        if verbose:
            print("." * 20)
            print(f"CYCLE: {cyc_name}")
        if powertrain is None or powertrain == "conv":
            veh_conv = fastsim.vehicle.Vehicle.from_vehdb(1)
            if verbose:
                print(f"CONV: {veh_conv.scenario_name}", flush=True)
            run_for_powertrain(save_dir, outputs, cyc_name, veh_conv, 'conv', init_soc=None, do_show=do_show, use_rust=use_rust, verbose=verbose, maneuver=maneuver)

        if powertrain is None or powertrain == "hev":
            veh_hev = fastsim.vehicle.Vehicle.from_vehdb(9)
            if verbose:
                print(f"HEV: {veh_hev.scenario_name}", flush=True)
            run_for_powertrain(save_dir, outputs, cyc_name, veh_hev, 'hev', init_soc=None, do_show=do_show, use_rust=use_rust, verbose=verbose, maneuver=maneuver)

        if powertrain is None or powertrain == "phev":
            veh_phev = fastsim.vehicle.Vehicle.from_vehdb(12)
            if verbose:
                print(f"PHEV: {veh_phev.scenario_name}", flush=True)
            run_for_powertrain(save_dir, outputs, cyc_name, veh_phev, 'phev', init_soc=None, do_show=do_show, use_rust=use_rust, verbose=verbose, maneuver=maneuver)

        if powertrain is None or powertrain == "bev":
            veh_bev = fastsim.vehicle.Vehicle.from_vehdb(17)
            if verbose:
                print(f"BEV: {veh_bev.scenario_name}", flush=True)
            run_for_powertrain(save_dir, outputs, cyc_name, veh_bev, 'bev', init_soc=None, do_show=do_show, use_rust=use_rust, verbose=verbose, maneuver=maneuver)

    keys = CSV_KEYS
    if save_dir is not None:
        with open(save_dir / 'output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(keys)
            for item in outputs:
                writer.writerow([str(item.get(k, '--')) for k in keys])
    if verbose:
        print("Done!", flush=True)
    return outputs


if __name__ == "__main__":
    cycle_name = None
    if len(sys.argv) >= 2 and sys.argv[1] != 'None':
        cycle_name = sys.argv[1]
    powertrain = None
    if len(sys.argv) >= 3 and sys.argv[2] in ("conv", "hev", "phev", "bev"):
        powertrain = sys.argv[2]
    do_show = None
    if len(sys.argv) >= 4 and sys.argv[3] == 'show':
        do_show = True
    use_rust = False
    if len(sys.argv) >= 5 and sys.argv[4] == 'rust':
        use_rust = True
    print("** CAV SWEEP **")
    print("-" * 40)
    print(f"CYCLE     : {cycle_name if cycle_name is not None else 'all'}")
    print(f"POWERTRAIN: {powertrain if powertrain is not None else 'all'}")
    print(f"SHOW FIGS : {do_show if do_show is not None else 'default'}")
    if use_rust:
        print("PLATFORM  : RUST BACKEND")
    else:
        print("PLATFORM  : PURE PYTHON")
    sys.stdout.flush()
    main(cycle_name, powertrain=powertrain, do_show=do_show, use_rust=use_rust, save_dir=OUTPUT_DIR)
