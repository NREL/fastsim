from pathlib import Path
import csv

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


def extend_cycle(cyc, absolute_time_s=0, time_fraction=0):
    """
    cyc: fastsim.cycle.Cycle
    absolute_time_s: float, the seconds to extend
    time_fraction: float, the fraction of the original cycle time to add on
    RETURNS: fastsim.cycle.Cycle, the new cycle with stopped time appended
    NOTE: additional time is rounded to the nearest second
    """
    cyc0 = cyc.get_cyc_dict()
    extra_time_s = absolute_time_s + float(int(round(time_fraction * cyc.time_s[-1])))
    # Zero-velocity cycle segment so simulation doesn't end while moving
    cyc_stop = fastsim.cycle.resample(
        fastsim.cycle.make_cycle([0.0, extra_time_s], [0.0, 0.0]),
        new_dt=1.0,
    )
    return fastsim.cycle.Cycle.from_dict(
        fastsim.cycle.concat([cyc0, cyc_stop])
    )


def make_distance_by_time_plot(cyc0, cyc, save_file=None, do_show=False):
    (fig, ax) = plt.subplots()
    ax.plot(cyc0.time_s, cyc0.dist_m.cumsum(), 'gray', label='lead')
    ax.plot(cyc.time_s, cyc.dist_m.cumsum(), 'b-', lw=2, label='cav')
    ax.plot(cyc.time_s, cyc.dist_m.cumsum(), 'r.', ms=1)
    ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.legend(loc=0)
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=300)
    if do_show:
        plt.show()
    plt.close()

def make_debug_plot(sd, save_file=None, do_show=False):
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
    axs[1].plot(sd.cyc.time_s, sd.coast_delay_index, 'b.', lw=2, label='coast delay')
    axs[1].set_ylabel('Coast Delay', color=color)
    axs[1].tick_params(axis='y', labelcolor=color)
    axs[1].grid(False)
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=300)
    if do_show:
        plt.show()
    plt.close()



def make_save_file(prefix, postfix, save_dir=None):
    if save_dir is not None:
        return save_dir / f'{prefix}_{postfix}'
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


def load_cycle(cyc_name: str) -> fastsim.cycle.Cycle:
    """
    """
    if cyc_name in PREMADE_CYCLES:
        raw_cycle = PREMADE_CYCLES.get(cyc_name)
    else:
        raw_cycle = fastsim.cycle.Cycle.from_file(cyc_name)
    return extend_cycle(
        raw_cycle,
        time_fraction=FRACTION_EXTENDED_TIME,
        absolute_time_s=ABSOLUTE_EXTENDED_TIME_S
    )


def no_eco_driving(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name)
    sim = fastsim.simdrive.SimDrive(cyc, veh)
    sim.sim_drive(init_soc=init_soc)
    print(f"NO ECO-DRIVING: {sim.mpgge:.3f} mpg")
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'base.png', save_dir), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'base_dist_by_time.png', save_dir))
    make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'base_debug.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )


def eco_coast(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name)
    sim = fastsim.simdrive.SimDrive(cyc, veh)
    params = sim.sim_params
    params.coast_allow = True
    params.coast_allow_passing = ECO_COAST_ALLOW_PASSING
    params.coast_start_speed_m_per_s = -1.0
    params.coast_time_horizon_for_adjustment_s = 20.0
    sim.sim_params = params
    sim.sim_drive(init_soc=init_soc)
    print(f"ECO-COAST: {sim.mpgge:.3f} mpg")
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocoast.png', save_dir), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocoast_dist_by_time.png', save_dir))
    make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'ecocoast_debug.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )


def eco_coast_by_microtrip(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None):
    min_gap_threshold = -0.1 # meters
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc_mts = fastsim.cycle.to_microtrips(load_cycle(cyc_name).get_cyc_dict())
    fuel_kj = 0.0
    ess_dischg_kj = 0.0
    dist_mi = 0.0
    base_traces = []
    traces = []
    for mt in cyc_mts:
        cyc = fastsim.cycle.Cycle.from_dict(mt)
        base_traces.append(cyc.get_cyc_dict())
        sim = fastsim.simdrive.SimDrive(cyc, veh)
        sim_base = fastsim.simdrive.copy_sim_drive(sim)
        params = sim.sim_params
        params.coast_allow = True
        params.coast_allow_passing = ECO_COAST_ALLOW_PASSING
        params.coast_start_speed_m_per_s = -1.0
        params.coast_time_horizon_for_adjustment_s = 20.0
        sim.sim_params = params
        sim.sim_drive(init_soc=init_soc)
        # check gap
        min_gap = np.min(sim.gap_to_lead_vehicle_m)
        # if min gap < threshold OR we end with non-zero speed, simulate using normal simdrive (i.e., no-eco), else use coasting
        if min_gap < min_gap_threshold or sim.cyc.mps[-1] > 0.1:
            sim_base.sim_drive(init_soc=init_soc)
            sim = sim_base
            traces.append(cyc.get_cyc_dict())
        else:
            traces.append(sim.cyc.get_cyc_dict())
        fuel_kj += sim.fuel_kj
        ess_dischg_kj += sim.ess_dischg_kj
        dist_mi += sim.dist_mi.sum()
    fuel_kwh_per_mi = ((fuel_kj + ess_dischg_kj) / 3.6e3) / dist_mi
    mpgge = 1.0 / (fuel_kwh_per_mi / sim.props.kwh_per_gge)
    print(f"ECO-COAST: {mpgge:.3f} mpg")
    cyc0 = fastsim.cycle.Cycle.from_dict(fastsim.cycle.concat(base_traces))
    cyc = fastsim.cycle.Cycle.from_dict(fastsim.cycle.concat(traces))
    make_coasting_plot(cyc0, cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocoast.png', save_dir), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
    make_distance_by_time_plot(cyc0, cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocoast_dist_by_time.png', save_dir))
    # make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'ecocoast_debug.png', save_dir))
    return (fuel_kwh_per_mi, dist_mi)



def time_spent_moving(cycle):
    """
    Calculates the time in seconds spent moving.

    Arguments:
    ----------
    cycle: drive cycle converted to dictionary by cycle.get_cycl_dict()

    RETURN: float, the time in seconds spent moving
    """
    t_move_s = 0.0
    for (dt, vavg) in zip(np.diff(cycle['time_s']), np.array(cycle['mps'][1:] + cycle['mps'][:-1]) / 2.0):
        if vavg > 0:
            t_move_s += dt
    return t_move_s


def create_dist_and_avg_speeds_by_microtrip(cyc, use_moving_time=True, verbose=False):
    dist_and_avg_speeds = []
    # Split cycle into microtrips
    microtrips = fastsim.cycle.to_microtrips(cyc.get_cyc_dict())
    dist_at_start_of_microtrip_m = 0.0
    for mt in microtrips:
        mt_cyc = fastsim.cycle.Cycle.from_dict(mt)
        mt_dist_m = sum(mt_cyc.dist_m)
        mt_time_s = time_spent_moving(mt_cyc.get_cyc_dict()) if use_moving_time else mt_cyc.time_s[-1]
        mt_avg_spd_m_per_s = mt_dist_m / mt_time_s if mt_time_s > 0.0 else 0.0
        if mt_dist_m > 0.0:
            dist_and_avg_speeds.append(
                (dist_at_start_of_microtrip_m, max(mt_avg_spd_m_per_s, MIN_ECO_CRUISE_TARGET_SPEED_m_per_s))
            )
            dist_at_start_of_microtrip_m += mt_dist_m
    if verbose:
        print('Microtrip distances and average speeds:')
        for (d, vavg) in dist_and_avg_speeds:
            print(f'- dist: {d:.3f} m / vavg: {vavg:.3f}')
    return dist_and_avg_speeds


def eco_cruise(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name)
    sim = fastsim.simdrive.SimDrive(cyc, veh)
    params = sim.sim_params
    dist_and_avg_speeds = create_dist_and_avg_speeds_by_microtrip(cyc)
    # Eco-cruise parameters
    params = sim.sim_params
    params.follow_allow = True
    params.idm_accel_m_per_s2 = 0.5
    params.idm_decel_m_per_s2 = 2.5
    params.idm_dt_headway_s = 2.0
    params.idm_minimum_gap_m = 10.0
    params.idm_v_desired_m_per_s = dist_and_avg_speeds[0][1]
    sim.sim_params = params
    # Initialize Electric Drive System
    init_soc = sim.veh.max_soc if init_soc is None else init_soc
    sim.init_for_step(init_soc=init_soc)
    # Simulate
    current_mt_idx = 0
    dist_traveled_m = 0.0
    while sim.i < len(cyc.time_s):
        idx = max(0, sim.i - 1)
        dist_traveled_m += sim.cyc.dist_m[idx]
        if current_mt_idx < len(dist_and_avg_speeds):
            mt_start_dist_m, mt_avg_spd_m_per_s = dist_and_avg_speeds[current_mt_idx]
            if dist_traveled_m >= mt_start_dist_m:
                params.idm_v_desired_m_per_s = mt_avg_spd_m_per_s
                sim.sim_params = params
                current_mt_idx += 1
        sim.sim_drive_step()
    sim.set_post_scalars()
    print(f"ECO-CRUISE: {sim.mpgge:.3f} mpg")
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocruise.png', save_dir), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'ecocruise_dist_by_time.png', save_dir))
    make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'ecocruise_debug.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )

def eco_coast_and_cruise(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None, do_show=None):
    do_show = DO_SHOW if do_show is None else do_show
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = load_cycle(cyc_name)
    sim = fastsim.simdrive.SimDrive(cyc, veh)
    params = sim.sim_params
    dist_and_avg_speeds = create_dist_and_avg_speeds_by_microtrip(cyc)
    params = sim.sim_params
    # Eco-coast parameters
    params.coast_allow = True
    params.coast_allow_passing = False
    params.coast_start_speed_m_per_s = -1.0
    params.coast_time_horizon_for_adjustment_s = 20.0
    # Eco-cruise parameters
    params.follow_allow = True
    params.idm_accel_m_per_s2 = 0.5
    params.idm_decel_m_per_s2 = 2.5
    params.idm_dt_headway_s = 2.0
    params.idm_minimum_gap_m = 10.0
    params.idm_v_desired_m_per_s = dist_and_avg_speeds[0][1]
    sim.sim_params = params
    # Initialize Electric Drive System
    init_soc = sim.veh.max_soc if init_soc is None else init_soc
    sim.init_for_step(init_soc=init_soc)
    # Simulate
    current_mt_idx = 0
    dist_traveled_m = 0.0
    while sim.i < len(cyc.time_s):
        idx = max(0, sim.i - 1)
        dist_traveled_m += sim.cyc.dist_m[idx]
        if current_mt_idx < len(dist_and_avg_speeds):
            mt_start_dist_m, mt_avg_spd_m_per_s = dist_and_avg_speeds[current_mt_idx]
            if dist_traveled_m >= mt_start_dist_m:
                params.idm_v_desired_m_per_s = mt_avg_spd_m_per_s
                sim.sim_params = params
                current_mt_idx += 1
        sim.sim_drive_step()
    sim.set_post_scalars()
    print(f"ECO-COAST + ECO-CRUISE: {sim.mpgge:.3f} mpg")
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'alleco.png', save_dir), coast_brake_start_speed_m_per_s=sim.sim_params.coast_brake_start_speed_m_per_s)
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=do_show, save_file=make_save_file(tag, 'alleco_dist_by_time.png', save_dir))
    make_debug_plot(sim, do_show=do_show, save_file=make_save_file(tag, 'alleco_debug.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )


def calc_percentage(base, other):
    return (base - other) * 100.0 / base


def run_for_powertrain(save_dir, outputs, cyc_name, veh, powertrain, init_soc=None, do_show=None):
    use_eco_coast_by_mt = False
    output = {'powertrain': powertrain, 'cycle': cyc_name, 'veh': veh.scenario_name}
    args = {
        'init_soc': init_soc,
        'cyc_name': cyc_name,
        'do_show': do_show if do_show is not None else DO_SHOW,
    }
    tag = f'{cyc_name}_{powertrain}'
    (output['use:base (kWh/mi)'], output['dist:base (mi)']) = no_eco_driving(veh, save_dir=save_dir, tag=tag, **args)
    if use_eco_coast_by_mt:
        (output['use:eco-coast (kWh/mi)'], output['dist:eco-coast (mi)']) = eco_coast_by_microtrip(veh, save_dir=save_dir, tag=tag, **args)
    else:
        (output['use:eco-coast (kWh/mi)'], output['dist:eco-coast (mi)']) = eco_coast(veh, save_dir=save_dir, tag=tag, **args)
    (output['use:eco-cruise (kWh/mi)'], output['dist:eco-cruise (mi)']) = eco_cruise(veh, save_dir=save_dir, tag=tag, **args)
    (output['use:all-eco (kWh/mi)'], output['dist:all-eco (mi)']) = eco_coast_and_cruise(veh, save_dir=save_dir, tag=tag, **args)
    output['savings:eco-coast (%)'] = calc_percentage(output['use:base (kWh/mi)'], output['use:eco-coast (kWh/mi)'])
    output['savings:eco-cruise (%)'] = calc_percentage(output['use:base (kWh/mi)'], output['use:eco-cruise (kWh/mi)'])
    output['savings:all-eco (%)'] = calc_percentage(output['use:base (kWh/mi)'], output['use:all-eco (kWh/mi)'])
    output['dist-short:eco-coast (%)'] = calc_percentage(output['dist:base (mi)'], output['dist:eco-coast (mi)'])
    output['dist-short:eco-cruise (%)'] = calc_percentage(output['dist:base (mi)'], output['dist:eco-cruise (mi)'])
    output['dist-short:all-eco (%)'] = calc_percentage(output['dist:base (mi)'], output['dist:all-eco (mi)'])
    outputs.append(output)


def main(cycle_name=None, powertrain=None, do_show=None):
    save_dir = OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    cyc_names = [cycle_name] if cycle_name is not None else [
        "hwfet", "udds", "us06", "NREL13", "trapz", "trapz-x2", "stacked-trapz"
    ]
    outputs = []
    for cyc_name in cyc_names:
        print(f"CYCLE: {cyc_name}")
        if powertrain is None or powertrain == "conv":
            veh_conv = fastsim.vehicle.Vehicle.from_vehdb(1)
            print(f"CONV: {veh_conv.scenario_name}")
            run_for_powertrain(save_dir, outputs, cyc_name, veh_conv, 'conv', init_soc=None, do_show=do_show)

        if powertrain is None or powertrain == "hev":
            veh_hev = fastsim.vehicle.Vehicle.from_vehdb(9)
            print(f"HEV: {veh_hev.scenario_name}")
            run_for_powertrain(save_dir, outputs, cyc_name, veh_hev, 'hev', init_soc=None, do_show=do_show)

        if powertrain is None or powertrain == "phev":
            veh_phev = fastsim.vehicle.Vehicle.from_vehdb(12)
            print(f"PHEV: {veh_phev.scenario_name}")
            run_for_powertrain(save_dir, outputs, cyc_name, veh_phev, 'phev', init_soc=None, do_show=do_show)

        if powertrain is None or powertrain == "bev":
            veh_bev = fastsim.vehicle.Vehicle.from_vehdb(17)
            print(f"BEV: {veh_bev.scenario_name}")
            run_for_powertrain(save_dir, outputs, cyc_name, veh_bev, 'bev', init_soc=None, do_show=do_show)

    keys = [
        'powertrain', 'cycle', 'veh',
        'use:base (kWh/mi)', 'use:eco-coast (kWh/mi)', 'use:eco-cruise (kWh/mi)', 'use:all-eco (kWh/mi)',
        'savings:eco-coast (%)', 'savings:eco-cruise (%)', 'savings:all-eco (%)',
        'dist:base (mi)', 'dist:eco-coast (mi)', 'dist:eco-cruise (mi)', 'dist:all-eco (mi)',
        'dist-short:eco-coast (%)', 'dist-short:eco-cruise (%)', 'dist-short:all-eco (%)',
    ]
    with open(save_dir / 'output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)
        for item in outputs:
            writer.writerow([str(item[k]) for k in keys])
    print("Done!")


if __name__ == "__main__":
    import sys
    cycle_name = None
    if len(sys.argv) >= 2:
        cycle_name = sys.argv[1]
    powertrain = None
    if len(sys.argv) >= 3 and sys.argv[2] in ("conv", "hev", "phev", "bev"):
        powertrain = sys.argv[2]
    do_show = None
    if len(sys.argv) >= 4 and sys.argv[3] == 'show':
        do_show = True
    main(cycle_name, powertrain=powertrain, do_show=do_show)
