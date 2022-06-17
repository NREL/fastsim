from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

import fastsim
from fastsim.tests.test_coasting import make_coasting_plot


DO_SHOW = False
FRACTION_EXTENDED_TIME = 0.1
ABSOLUTE_EXTENDED_TIME_S = 180.0
OUTPUT_DIR = Path(__file__).parent.absolute() / 'test_output'


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
    ax.plot(cyc0.time_s, cyc0.dist_m.cumsum(), 'k-', label='lead')
    ax.plot(cyc.time_s, cyc.dist_m.cumsum(), 'b-', label='cav')
    ax.plot(cyc.time_s, cyc.dist_m.cumsum(), 'r.', markersize=0.5)
    ax.set_xlabel('Elapsed Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.legend(loc=0)
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


def no_eco_driving(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None):
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = extend_cycle(fastsim.cycle.Cycle.from_file(cyc_name), time_fraction=FRACTION_EXTENDED_TIME, absolute_time_s=ABSOLUTE_EXTENDED_TIME_S)
    sim = fastsim.simdrive.SimDrive(cyc, veh)
    sim.sim_drive(init_soc=init_soc)
    print(f"NO ECO-DRIVING: {sim.mpgge:.3f} mpg")
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'base.png', save_dir))
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'base_dist_by_time.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )


def eco_coast(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None):
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = extend_cycle(fastsim.cycle.Cycle.from_file(cyc_name), time_fraction=FRACTION_EXTENDED_TIME, absolute_time_s=ABSOLUTE_EXTENDED_TIME_S)
    sim = fastsim.simdrive.SimDrive(cyc, veh)
    params = sim.sim_params
    params.coast_allow = True
    params.coast_allow_passing = False
    params.coast_start_speed_m_per_s = -1.0
    sim.sim_params = params
    sim.sim_drive(init_soc=init_soc)
    print(f"ECO-COAST: {sim.mpgge:.3f} mpg")
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'ecocoast.png', save_dir))
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'ecocoast_dist_by_time.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )


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
        if mt_dist_m > 0.0 and mt_avg_spd_m_per_s > 1.0:
            dist_and_avg_speeds.append(
                (dist_at_start_of_microtrip_m, mt_avg_spd_m_per_s)
            )
            dist_at_start_of_microtrip_m += mt_dist_m
    if verbose:
        print('Microtrip distances and average speeds:')
        for (d, vavg) in dist_and_avg_speeds:
            print(f'- dist: {d:.3f} m / vavg: {vavg:.3f}')
    return dist_and_avg_speeds


def eco_cruise(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None):
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = extend_cycle(fastsim.cycle.Cycle.from_file(cyc_name), time_fraction=FRACTION_EXTENDED_TIME, absolute_time_s=ABSOLUTE_EXTENDED_TIME_S)
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
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'ecocruise.png', save_dir))
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'ecocruise_dist_by_time.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )

def eco_coast_and_cruise(veh, init_soc=None, save_dir=None, tag=None, cyc_name=None):
    cyc_name = "udds" if cyc_name is None else cyc_name
    cyc = extend_cycle(fastsim.cycle.Cycle.from_file(cyc_name), time_fraction=FRACTION_EXTENDED_TIME, absolute_time_s=ABSOLUTE_EXTENDED_TIME_S)
    sim = fastsim.simdrive.SimDrive(cyc, veh)
    params = sim.sim_params
    dist_and_avg_speeds = create_dist_and_avg_speeds_by_microtrip(cyc)
    params = sim.sim_params
    # Eco-coast parameters
    params.coast_allow = True
    params.coast_allow_passing = False
    params.coast_start_speed_m_per_s = -1.0
    # Eco-cruise parameters
    params.follow_allow = True
    params.idm_accel_m_per_s2 = 0.5
    params.idm_decel_m_per_s2 = 2.5
    params.idm_dt_headway_s = 2.0
    params.idm_minimum_gap_m = 10.0
    params.idm_v_desired_m_per_s = dist_and_avg_speeds[0][1]
    sim.sim_params = params
    # Initialize Electric Drive System
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
    make_coasting_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'alleco.png', save_dir))
    make_distance_by_time_plot(sim.cyc0, sim.cyc, do_show=DO_SHOW, save_file=make_save_file(tag, 'alleco_dist_by_time.png', save_dir))
    return (
        ((sim.fuel_kj + sim.ess_dischg_kj) / 3.6e3) / sim.dist_mi.sum(),
        sim.dist_mi.sum()
    )


def calc_percentage(base, other):
    return (base - other) * 100.0 / base


def run_for_powertrain(save_dir, outputs, cyc_name, veh, powertrain, init_soc=None):
    output = {'powertrain': powertrain, 'cycle': cyc_name, 'veh': veh.scenario_name}
    args = {
        'init_soc': init_soc,
        'cyc_name': cyc_name,
    }
    tag = f'{cyc_name}_{powertrain}'
    (output['use:base (kWh/mi)'], output['dist:base (mi)']) = no_eco_driving(veh, save_dir=save_dir, tag=tag, **args)
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


def main():
    save_dir = OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    cyc_names = ["hwfet", "udds", "us06"]
    outputs = []
    for cyc_name in cyc_names:
        veh_conv = fastsim.vehicle.Vehicle.from_vehdb(1)
        print(f"CONV: {veh_conv.scenario_name}")
        run_for_powertrain(save_dir, outputs, cyc_name, veh_conv, 'conv', init_soc=None)

        veh_hev = fastsim.vehicle.Vehicle.from_vehdb(9)
        print(f"HEV: {veh_hev.scenario_name}")
        run_for_powertrain(save_dir, outputs, cyc_name, veh_hev, 'hev', init_soc=0.8)

        veh_phev = fastsim.vehicle.Vehicle.from_vehdb(12)
        print(f"PHEV: {veh_phev.scenario_name}")
        run_for_powertrain(save_dir, outputs, cyc_name, veh_phev, 'phev', init_soc=1.0)

        veh_bev = fastsim.vehicle.Vehicle.from_vehdb(17)
        print(f"BEV: {veh_bev.scenario_name}")
        run_for_powertrain(save_dir, outputs, cyc_name, veh_bev, 'bev', init_soc=1.0)

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
    main()
