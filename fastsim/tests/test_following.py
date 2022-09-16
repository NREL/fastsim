"""
Tests that check the drive cycle modification functionality.
"""
import unittest

import numpy as np

import fastsim
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable
from fastsim.auxiliaries import set_nested_values

DO_PLOTS = False
USE_PYTHON = True
USE_RUST = True

if USE_RUST and not RUST_AVAILABLE:
    warn_rust_unavailable(__file__)


class TestFollowing(unittest.TestCase):
    def setUp(self) -> None:
        fastsim.utils.disable_logging()

        self.initial_gap_m = 5.0
        trapz = fastsim.cycle.make_cycle(
            [0.0, 10.0, 45.0, 55.0, 150.0],
            [0.0, 20.0, 20.0, 0.0, 0.0],
        )
        trapz = fastsim.cycle.resample(trapz, new_dt=1.0)
        if USE_PYTHON:
            self.trapz = fastsim.cycle.Cycle.from_dict(trapz)
            self.veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            # sd0 is for reference to an unchanged, no-following simdrive
            self.sd0 = fastsim.simdrive.SimDrive(self.trapz, self.veh)
            self.sd = fastsim.simdrive.SimDrive(self.trapz, self.veh)
            self.sd.sim_params.idm_allow = True
            self.sd.sim_params.idm_minimum_gap_m = self.initial_gap_m
        if RUST_AVAILABLE and USE_RUST:
            self.ru_trapz = fastsim.cycle.Cycle.from_dict(trapz).to_rust()
            self.ru_veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            # sd0 is for reference to an unchanged, no-following simdrive
            self.ru_sd0 = fastsim.simdrive.RustSimDrive(self.ru_trapz, self.ru_veh)
            self.ru_sd0.sim_params = set_nested_values(self.ru_sd0.sim_params)
            self.ru_sd = fastsim.simdrive.RustSimDrive(self.ru_trapz, self.ru_veh)
            self.ru_sd.sim_params = set_nested_values(self.ru_sd.sim_params,
                idm_allow=True,
                idm_minimum_gap_m=self.initial_gap_m
            )
        return super().setUp()

    def test_that_we_have_a_gap_between_us_and_the_lead_vehicle(self):
        "A positive gap should exist between us and the lead vehicle"
        if USE_PYTHON:
            self.assertTrue(self.sd.sim_params.idm_allow)
            self.sd.sim_drive()
            self.assertTrue(self.sd.sim_params.idm_allow)
            gaps_m = self.sd.gap_to_lead_vehicle_m
            if DO_PLOTS:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(self.sd.cyc0.time_s, gaps_m, 'k.')
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Gap (m)')
                fig.tight_layout()
                save_file = "test_that_we_have_a_gap_between_us_and_the_lead_vehicle__0.png"
                fig.savefig(save_file, dpi=300)
                plt.close()
            self.assertTrue((gaps_m > 0.0).any())
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(self.sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(self.sd.cyc).sum(),
                places=-1)
        if RUST_AVAILABLE and USE_RUST:
            self.assertTrue(self.ru_sd.sim_params.idm_allow)
            self.ru_sd.sim_drive()
            self.assertTrue(self.ru_sd.sim_params.idm_allow)
            gaps_m = np.array(self.ru_sd.gap_to_lead_vehicle_m())
            if DO_PLOTS:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(self.ru_sd.cyc0.time_s, gaps_m, 'k.')
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Gap (m)')
                fig.tight_layout()
                save_file = "test_that_we_have_a_gap_between_us_and_the_lead_vehicle__0-rust.png"
                fig.savefig(save_file, dpi=300)
                plt.close()
            self.assertTrue((gaps_m > 0.0).any())
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(self.ru_sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(self.ru_sd.cyc).sum(),
                places=-1)

    def test_that_the_gap_changes_over_the_cycle(self):
        "Ensure that our gap calculation is doing something"
        if USE_PYTHON:
            self.sd.sim_drive()
            gaps_m = self.sd.gap_to_lead_vehicle_m
            if DO_PLOTS:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(self.sd.cyc0.time_s, gaps_m, 'k.')
                bad_gap_mask = gaps_m < self.initial_gap_m
                if bad_gap_mask.sum() > 0:
                    if False:
                        print(f"MAX GAP: {gaps_m.max()}")
                        print(f"MIN GAP: {gaps_m.min()}")
                    ax.plot(self.sd.cyc0.time_s[bad_gap_mask], gaps_m[bad_gap_mask], 'ro')
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Gap (m)')
                fig.tight_layout()
                save_file = "test_that_the_gap_changes_over_the_cycle__0.png"
                fig.savefig(save_file, dpi=300)
                plt.close()
                from fastsim.tests.test_coasting import make_coasting_plot
                make_coasting_plot(
                    self.sd.cyc0, self.sd.cyc,
                    title='Test that Gap Changes Over Cycle (Python)',
                    save_file="test_that_the_gap_changes_over_the_cycle__1.png",
                )
            self.assertFalse((gaps_m == 5.0).all())
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(self.sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(self.sd.cyc).sum(),
                places=-1,
                msg='Distance should be about the same')
            self.assertTrue((gaps_m > 0.0).all(), msg='We cannot pass the lead vehicle')
            self.assertTrue(
                (gaps_m > (self.initial_gap_m - 1.0)).all(),
                msg='We cannot get closer than the initial gap distance')
        if RUST_AVAILABLE and USE_RUST:
            self.ru_sd.sim_drive()
            gaps_m = np.array(self.ru_sd.gap_to_lead_vehicle_m())
            if DO_PLOTS:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(self.ru_sd.cyc0.time_s, gaps_m, 'k.')
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Gap (m)')
                fig.tight_layout()
                save_file = "test_that_the_gap_changes_over_the_cycle__0-rust.png"
                fig.savefig(save_file, dpi=300)
                plt.close()
                from fastsim.tests.test_coasting import make_coasting_plot
                make_coasting_plot(
                    self.ru_sd.cyc0, self.ru_sd.cyc,
                    title='Test that Gap Changes Over Cycle (Rust)',
                    save_file="test_that_the_gap_changes_over_the_cycle__1-rust.png",
                )
            self.assertFalse((gaps_m == 5.0).all())
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(self.ru_sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(self.ru_sd.cyc).sum(),
                places=-1,
                msg='Distance should be about the same')
            self.assertTrue((gaps_m > 0.0).all(), msg='We cannot pass the lead vehicle')
            self.assertTrue(
                (gaps_m > (self.initial_gap_m - 1.0)).all(),
                msg='We cannot get closer than the initial gap distance')

    def test_that_following_works_over_parameter_sweep(self):
        "We're going to sweep through all of the parameters and see how it goes"
        if DO_PLOTS:
            if USE_PYTHON:
                import matplotlib.pyplot as plt
                import pandas as pd
                import seaborn as sns
                sns.set()

                accel_coefs = np.linspace(-10.0, 10.0, num=5, endpoint=True)
                speed_coefs = np.linspace(-10.0, 10.0, num=5, endpoint=True)
                init_offsets= np.linspace(0.0, 100.0, num=5, endpoint=True)
                udds = fastsim.cycle.Cycle.from_file('udds')
                veh = fastsim.vehicle.Vehicle.from_vehdb(5)
                sd = fastsim.simdrive.SimDrive(udds, veh)
                sd.sim_drive()
                base_distance_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                base_fuel_consumption_gal__100mi = 100.0 / sd.mpgge
                results = {
                    'accel_coef_s2': [0.0],
                    'speed_coef_s': [0.0],
                    'init_offset_m': [0.0],
                    'fuel_economy_mpgge': [sd.mpgge],
                    'fuel_consumption_gal__100mi': [base_fuel_consumption_gal__100mi],
                    'distance_m': [base_distance_m],
                    'distance_short_m': [0.0],
                }
                idx = 0
                for accel_coef_s2 in accel_coefs:
                    for speed_coef_s in speed_coefs:
                        for initial_offset_m in init_offsets:
                            idx += 1
                            if idx % 10 == 0:
                                print(f"Running {idx}...")
                            udds = fastsim.cycle.Cycle.from_file('udds')
                            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
                            sd = fastsim.simdrive.SimDrive(udds, veh)
                            sd.sim_params.idm_allow = True
                            sd.sim_drive()
                            results['accel_coef_s2'].append(accel_coef_s2)
                            results['speed_coef_s'].append(speed_coef_s)
                            results['init_offset_m'].append(initial_offset_m)
                            results['fuel_economy_mpgge'].append(sd.mpgge)
                            results['fuel_consumption_gal__100mi'].append(100.0 / sd.mpgge)
                            dist_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                            results['distance_m'].append(dist_m)
                            results['distance_short_m'].append(base_distance_m - dist_m)
                results = pd.DataFrame(results)
                # Sweep of Speed Coef, holding accel_coef_s2
                save_file = 'test_that_following_works_over_parameter_sweep_SC.png'
                fig, ax = plt.subplots()
                ac_range = [accel_coefs[0], accel_coefs[len(accel_coefs)//4], accel_coefs[len(accel_coefs)//2], accel_coefs[3*len(accel_coefs)//4], accel_coefs[-1]]
                for ac in ac_range:
                    mask = results['accel_coef_s2'] == ac
                    ax.plot(
                        results['speed_coef_s'][mask],
                        results['fuel_consumption_gal__100mi'][mask],
                        linestyle='-',
                        marker='.',
                        label=f'AC={ac} s2')
                ax.plot([speed_coefs[0], speed_coefs[-1]], [base_fuel_consumption_gal__100mi]*2, 'y--', label=f'baseline')
                ax.set_xlabel('Speed Coefficient (s)')
                ax.set_ylabel('Fuel Consumption (gallons/100 miles)')
                ax.set_title('Fuel Consumption by Speed Coefficient')
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_file, dpi=300)
                plt.close()

                # Sweep of Accel Coef, holding speed_coef_s
                save_file = 'test_that_following_works_over_parameter_sweep_AC.png'
                fig, ax = plt.subplots()
                sc_range = [speed_coefs[0], speed_coefs[len(speed_coefs)//4], speed_coefs[len(speed_coefs)//2], speed_coefs[3*len(speed_coefs)//4], speed_coefs[-1]]
                for sc in sc_range:
                    mask = results['speed_coef_s'] == sc
                    ax.plot(
                        results['accel_coef_s2'][mask],
                        results['fuel_consumption_gal__100mi'][mask],
                        linestyle='-',
                        marker='.',
                        label=f'SC={sc} s')
                ax.plot([accel_coefs[0], accel_coefs[-1]], [base_fuel_consumption_gal__100mi]*2, 'y--', label=f'baseline')
                ax.set_xlabel('Accel Coefficient (s2)')
                ax.set_ylabel('Fuel Consumption (gallons/100 miles)')
                ax.set_title('Fuel Consumption by Accel Coefficient')
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_file, dpi=300)
                plt.close()

                # Sweep of Init Offset
                save_file = 'test_that_following_works_over_parameter_sweep_OFFSET.png'
                fig, ax = plt.subplots()
                ax.plot(
                    results['init_offset_m'],
                    results['fuel_consumption_gal__100mi'],
                    linestyle='none',
                    marker='.',
                    label=f'init-offset')
                ax.plot([init_offsets[0], init_offsets[-1]], [base_fuel_consumption_gal__100mi]*2, 'y--', label=f'baseline')
                ax.set_xlabel('Lead Vehicle Offset (m)')
                ax.set_ylabel('Fuel Consumption (gallons/100 miles)')
                ax.set_title('Fuel Consumption by Lead Vehicle Offset')
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_file, dpi=300)
                plt.close()
            if RUST_AVAILABLE and USE_RUST:
                import matplotlib.pyplot as plt
                import pandas as pd
                import seaborn as sns
                sns.set()

                accel_coefs = np.linspace(-10.0, 10.0, num=5, endpoint=True)
                speed_coefs = np.linspace(-10.0, 10.0, num=5, endpoint=True)
                init_offsets= np.linspace(0.0, 100.0, num=5, endpoint=True)
                udds = fastsim.cycle.Cycle.from_file('udds').to_rust()
                veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
                sd = fastsim.simdrive.RustSimDrive(udds, veh)
                sd.sim_drive()
                base_distance_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                base_fuel_consumption_gal__100mi = 100.0 / sd.mpgge
                results = {
                    'accel_coef_s2': [0.0],
                    'speed_coef_s': [0.0],
                    'init_offset_m': [0.0],
                    'fuel_economy_mpgge': [sd.mpgge],
                    'fuel_consumption_gal__100mi': [base_fuel_consumption_gal__100mi],
                    'distance_m': [base_distance_m],
                    'distance_short_m': [0.0],
                }
                idx = 0
                for accel_coef_s2 in accel_coefs:
                    for speed_coef_s in speed_coefs:
                        for initial_offset_m in init_offsets:
                            idx += 1
                            if idx % 10 == 0:
                                print(f"Running {idx}...")
                            udds = fastsim.cycle.Cycle.from_file('udds').to_rust()
                            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
                            sd = fastsim.simdrive.RustSimDrive(udds, veh)
                            sim_params = sd.sim_params
                            sim_params.idm_allow = True
                            sd.sim_params = sim_params
                            sd.sim_drive()
                            results['accel_coef_s2'].append(accel_coef_s2)
                            results['speed_coef_s'].append(speed_coef_s)
                            results['init_offset_m'].append(initial_offset_m)
                            results['fuel_economy_mpgge'].append(sd.mpgge)
                            results['fuel_consumption_gal__100mi'].append(100.0 / sd.mpgge)
                            dist_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                            results['distance_m'].append(dist_m)
                            results['distance_short_m'].append(base_distance_m - dist_m)
                results = pd.DataFrame(results)
                # Sweep of Speed Coef, holding accel_coef_s2
                save_file = 'test_that_following_works_over_parameter_sweep_SC-rust.png'
                fig, ax = plt.subplots()
                ac_range = [accel_coefs[0], accel_coefs[len(accel_coefs)//4], accel_coefs[len(accel_coefs)//2], accel_coefs[3*len(accel_coefs)//4], accel_coefs[-1]]
                for ac in ac_range:
                    mask = results['accel_coef_s2'] == ac
                    ax.plot(
                        results['speed_coef_s'][mask],
                        results['fuel_consumption_gal__100mi'][mask],
                        linestyle='-',
                        marker='.',
                        label=f'AC={ac} s2')
                ax.plot([speed_coefs[0], speed_coefs[-1]], [base_fuel_consumption_gal__100mi]*2, 'y--', label=f'baseline')
                ax.set_xlabel('Speed Coefficient (s)')
                ax.set_ylabel('Fuel Consumption (gallons/100 miles)')
                ax.set_title('Fuel Consumption by Speed Coefficient')
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_file, dpi=300)
                plt.close()

                # Sweep of Accel Coef, holding speed_coef_s
                save_file = 'test_that_following_works_over_parameter_sweep_AC-rust.png'
                fig, ax = plt.subplots()
                sc_range = [speed_coefs[0], speed_coefs[len(speed_coefs)//4], speed_coefs[len(speed_coefs)//2], speed_coefs[3*len(speed_coefs)//4], speed_coefs[-1]]
                for sc in sc_range:
                    mask = results['speed_coef_s'] == sc
                    ax.plot(
                        results['accel_coef_s2'][mask],
                        results['fuel_consumption_gal__100mi'][mask],
                        linestyle='-',
                        marker='.',
                        label=f'SC={sc} s')
                ax.plot([accel_coefs[0], accel_coefs[-1]], [base_fuel_consumption_gal__100mi]*2, 'y--', label=f'baseline')
                ax.set_xlabel('Accel Coefficient (s2)')
                ax.set_ylabel('Fuel Consumption (gallons/100 miles)')
                ax.set_title('Fuel Consumption by Accel Coefficient')
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_file, dpi=300)
                plt.close()

                # Sweep of Init Offset
                save_file = 'test_that_following_works_over_parameter_sweep_OFFSET-rust.png'
                fig, ax = plt.subplots()
                ax.plot(
                    results['init_offset_m'],
                    results['fuel_consumption_gal__100mi'],
                    linestyle='none',
                    marker='.',
                    label=f'init-offset')
                ax.plot([init_offsets[0], init_offsets[-1]], [base_fuel_consumption_gal__100mi]*2, 'y--', label=f'baseline')
                ax.set_xlabel('Lead Vehicle Offset (m)')
                ax.set_ylabel('Fuel Consumption (gallons/100 miles)')
                ax.set_title('Fuel Consumption by Lead Vehicle Offset')
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_file, dpi=300)
                plt.close()
    
    def test_that_we_can_use_the_idm(self):
        "Tests use of the IDM model for following"
        if USE_PYTHON:
            self.sd.sim_drive()
            gaps_m = self.sd.gap_to_lead_vehicle_m
            self.assertTrue((gaps_m > self.initial_gap_m).any())
            if DO_PLOTS:
                from fastsim.tests.test_coasting import make_coasting_plot
                make_coasting_plot(
                    self.sd.cyc0, self.sd.cyc,
                    title='test_that_we_can_use_the_idm__1.png',
                    save_file="test_that_we_can_use_the_idm__1.png")
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(self.sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(self.sd.cyc).sum(),
                places=-1,
                msg='Distance traveled should be fairly close')
        if RUST_AVAILABLE and USE_RUST:
            self.ru_sd.sim_drive()
            gaps_m = np.array(self.ru_sd.gap_to_lead_vehicle_m())
            self.assertTrue((gaps_m > self.initial_gap_m).any())
            if DO_PLOTS:
                from fastsim.tests.test_coasting import make_coasting_plot
                make_coasting_plot(
                    self.ru_sd.cyc0, self.ru_sd.cyc,
                    title='Test That We Can use the IDM (RUST)',
                    save_file="test_that_we_can_use_the_idm__1-rust.png")
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(self.ru_sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(self.ru_sd.cyc).sum(),
                places=-1,
                msg='Distance traveled should be fairly close')

    def test_sweeping_idm_parameters(self):
        "Tests use of the IDM model for following"
        if DO_PLOTS:
            if USE_PYTHON:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import pandas as pd
                import seaborn as sns
                import pathlib as pl
                sns.set()

                datafile_name = 'test_sweeping_idm_parameters.csv'
                save_dir = pl.Path('test_output')
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)
                datafile_path = save_dir / datafile_name

                results = None
                # CYCLE STATISICS
                stopped_dict = fastsim.cycle.make_cycle([0, 120], [0, 0])
                udds_dict = fastsim.cycle.Cycle.from_file('udds').get_cyc_dict()
                udds = fastsim.cycle.Cycle.from_dict(
                    fastsim.cycle.resample(
                        fastsim.cycle.concat([udds_dict, stopped_dict]),
                        new_dt=1.0,
                    )
                )
                max_speed_m__s = np.max(udds.mps)
                id_by_pt_type = {
                    'CONV': 2,
                    'HEV': 10,
                    'BEV': 19,
                }
                mask = fastsim.cycle.trapz_step_distances(udds) > 0
                avg_speed_when_moving_m__s = fastsim.cycle.trapz_step_distances(udds)[mask].sum() / udds.dt_s[mask].sum()
                # DESIGN VARIABLES
                veh_pt_types = ['CONV', 'HEV', 'BEV']
                dt_headways = [0.5, 1.0, 4.0, 5.0, 10.0]
                v_desireds = [avg_speed_when_moving_m__s, max_speed_m__s]
                s_mins = np.linspace(1.0, 10.0, num=5, endpoint=True)
                deltas = [1.0, 2.0, 4.0, 6.0]
                a_accels = np.linspace(0.5, 2.5, num=5, endpoint=True)
                b_accels = np.linspace(1.0, 3.0, num=5, endpoint=True)
                # GENERATE DATA (and PLOTS)
                if not pl.Path(datafile_path).exists():
                    num_runs = len(veh_pt_types) + (
                        len(veh_pt_types) * len(dt_headways) * len(v_desireds)
                        * len(s_mins) * len(deltas) * len(a_accels) * len(b_accels)
                    )
                    print(f"TOTAL RUNS: {num_runs}")
                    results = {
                        'pt_type': [],
                        'dt_headway_s': [],
                        'v_desired_m__s': [],
                        's_min_m': [],
                        'delta': [],
                        'a_m__s2': [],
                        'b_m__s2': [],
                        'fuel_economy_mpgge': [],
                        'fuel_consumption_gal__100mi': [],
                        'distance_m': [],
                        'distance_short_m': [],
                        'battery_kWh_per_mi': [],
                    }
                    idx = 0
                    fmt_str = "{:.2f}"
                    def fmt(n):
                        return fmt_str.format(n).replace('.', '_')
                    for pt_type in veh_pt_types:
                        veh = fastsim.vehicle.Vehicle.from_vehdb(id_by_pt_type[pt_type])
                        sd = fastsim.simdrive.SimDrive(udds, veh)
                        sd.sim_drive()
                        base_distance_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                        if sd.mpgge == 0.0 or np.isnan(sd.mpgge):
                            base_fuel_consumption_gal__100mi = 0.0
                        else: 
                            base_fuel_consumption_gal__100mi = 100.0 / sd.mpgge
                        results['pt_type'].append(pt_type)
                        results['dt_headway_s'].append(0.0)
                        results['v_desired_m__s'].append(0.0)
                        results['s_min_m'].append(0.0)
                        results['delta'].append(0.0)
                        results['a_m__s2'].append(0.0)
                        results['b_m__s2'].append(0.0)
                        results['fuel_economy_mpgge'].append(sd.mpgge)
                        results['battery_kWh_per_mi'].append(sd.battery_kwh_per_mi)
                        if sd.mpgge > 0.0:
                            results['fuel_consumption_gal__100mi'].append(base_fuel_consumption_gal__100mi)
                        elif sd.battery_kwh_per_mi > 0.0:
                            gge__mi = sd.battery_kwh_per_mi / sd.props.kwh_per_gge
                            results['fuel_consumption_gal__100mi'].append(100.0 * gge__mi)
                        else:
                            results['fuel_consumption_gal__100mi'].append(-1.0)
                        results['distance_m'].append(base_distance_m)
                        results['distance_short_m'].append(0.0)
                        for dt_h_s in dt_headways:
                            for v_d_m__s in v_desireds:
                                for s_min_m in s_mins:
                                    for delta in deltas:
                                        for a_m__s2 in a_accels:
                                            for b_m__s2 in b_accels:
                                                key = (
                                                    "run_"
                                                    + f"pt{pt_type}_"
                                                    + f"dth{fmt(dt_h_s)}_"
                                                    + f"vd{fmt(v_d_m__s)}_"
                                                    + f"s0{fmt(s_min_m)}_"
                                                    + f"delta{fmt(delta)}_"
                                                    + f"a{fmt(a_m__s2)}_"
                                                    + f"b{fmt(b_m__s2)}"
                                                )
                                                idx += 1
                                                if idx % 10 == 0:
                                                    print(f"Running {idx}...")
                                                sd = fastsim.simdrive.SimDrive(udds, veh)
                                                sd.sim_params.idm_allow = True
                                                sd.sim_params.idm_minimum_gap_m = s_min_m
                                                sd.sim_params.idm_delta = delta
                                                sd.sim_params.idm_accel_m_per_s2 = a_m__s2
                                                sd.sim_params.idm_decel_m_per_s2 = b_m__s2
                                                sd.sim_params.idm_v_desired_m_per_s = v_d_m__s
                                                sd.sim_params.idm_dt_headway_s = dt_h_s
                                                sd.sim_drive()
                                                results['pt_type'].append(pt_type)
                                                results['dt_headway_s'].append(dt_h_s)
                                                results['v_desired_m__s'].append(v_d_m__s)
                                                results['s_min_m'].append(s_min_m)
                                                results['delta'].append(delta)
                                                results['a_m__s2'].append(a_m__s2)
                                                results['b_m__s2'].append(b_m__s2)
                                                results['fuel_economy_mpgge'].append(sd.mpgge)
                                                results['battery_kWh_per_mi'].append(sd.battery_kwh_per_mi)
                                                if not np.isnan(sd.mpgge) and sd.mpgge > 0.0:
                                                    results['fuel_consumption_gal__100mi'].append(100.0 / sd.mpgge)
                                                elif sd.battery_kwh_per_mi > 0.0:
                                                    pass
                                                    gge__mi = sd.battery_kwh_per_mi / sd.props.kwh_per_gge
                                                    results['fuel_consumption_gal__100mi'].append(100.0 * gge__mi)
                                                else:
                                                    results['fuel_consumption_gal__100mi'].append(-1.0)
                                                dist_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                                                results['distance_m'].append(dist_m)
                                                results['distance_short_m'].append(base_distance_m - dist_m)
                                                # create a plot by key and save it
                                                from fastsim.tests.test_coasting import make_coasting_plot 
                                                save_path = save_dir / f'{key}.png'
                                                try:
                                                    make_coasting_plot(
                                                        sd.cyc0,
                                                        sd.cyc,
                                                        use_mph=False,
                                                        title=key,
                                                        save_file=save_path,
                                                        do_show=False,
                                                        gap_offset_m=s_min_m,
                                                    )
                                                except Exception as err: 
                                                    with open(save_dir / 'run-log.txt', 'a') as f:
                                                        f.write(f'issue creating plot for {key}: {err}')
                                                if False:
                                                    # Pickles the runs; note: takes a lot of disk space...
                                                    import pickle
                                                    sd_path = save_dir / f'{key}.pickle'
                                                    with open(sd_path, 'wb') as f:
                                                        sd_copy = fastsim.simdrive.copy_sim_drive(sd)
                                                        pickle.dump(sd_copy, f)
                    print(f"idx: {idx}")
                    results = pd.DataFrame(results)
                    results.to_csv(datafile_path)
                else:
                    results = pd.read_csv(datafile_path)
                self.assertTrue(results is not None)
                results['distance_short_pct'] = (
                    (results['distance_short_m'] * 100.0)
                    / (results['distance_short_m'] + results['distance_m'])
                )
                # seaborn plots
                SM_DELTA = '\u03b4'
                param_labels = [
                    #'pt_type', 'v_desired_m__s', 'dt_headway_s', 's_min_m', 'delta', 'a_m__s2', 'b_m__s2'
                    'PT', 'Vd', 'dth', 's', SM_DELTA, 'a', 'b'
                ]
                RR = results.rename(columns={
                    'pt_type':'PT',
                    'v_desired_m__s': 'Vd',
                    'dt_headway_s':'dth',
                    's_min_m': 's',
                    'delta': SM_DELTA,
                    'a_m__s2': 'a',
                    'b_m__s2': 'b',
                })
                for hue_item in param_labels:
                    g = sns.displot(
                        data=RR,
                        x="fuel_consumption_gal__100mi",
                        hue=hue_item,
                        multiple="stack",
                    )
                    g.ax.plot([RR['fuel_consumption_gal__100mi'][0]]*2, [0, 1000], 'r-', linewidth=2)
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_distplot_{hue_item}.png', dpi=300)
                    plt.close()
                    g = None
                # PERCENTAGE DISTANCE SHORT PLOT
                g = sns.displot(
                    data=RR,
                    x='distance_short_pct',
                    multiple='stack'
                )
                g.savefig(save_dir / 'test_sweeping_idm_parameters_distplot_dist_short.png', dpi=300)
                plt.close()
                g = None
                # FacetGrid HistPlot of pt_type x v_desired_m__s for fuel_consumption
                g = sns.FacetGrid(RR, row="PT", col="Vd", hue='dth')
                g.map(sns.histplot, "fuel_consumption_gal__100mi")
                g.savefig(
                    save_dir / 'test_sweeping_idm_parameters_facetgrid_row_pt_type_col_v_desired_for_fuel_consumption.png',
                    dpi=600)
                plt.close()
                g = None
                # FacetGrid HistPlot of pt_type x dt_headway_s for fuel_consumption
                g = sns.FacetGrid(RR, row="PT", col="dth", hue='Vd')
                g.map(sns.histplot, "fuel_consumption_gal__100mi")
                g.savefig(
                    save_dir / 'test_sweeping_idm_parameters_facetgrid_row_pt_type_col_dt_headway_for_fuel_consumption.png',
                    dpi=600)
                plt.close()
                g = None
                # FacetGrid HistPlot of pt_type x dt_headway_s for fuel_consumption
                g = sns.FacetGrid(RR, row="Vd", col="dth", hue='PT')
                g.map(sns.histplot, "fuel_consumption_gal__100mi")
                g.savefig(
                    save_dir / 'test_sweeping_idm_parameters_facetgrid_row_v_desired_col_dt_headway_for_fuel_consumption.png',
                    dpi=600)
                plt.close()
                g = None
                for pt_type in veh_pt_types:
                    param_labels = ['Vd', 'dth', 's', SM_DELTA, 'a', 'b']
                    R = RR[RR['PT'] == pt_type]
                    r0 = R[R["Vd"] == 0.0]
                    base_fuel_consumption_gal__100mi = float(r0["fuel_consumption_gal__100mi"].values[0])
                    print(f'Baseline Fuel Consumption for {pt_type}: {base_fuel_consumption_gal__100mi} gal/100mi')
                    r = R[R["Vd"] != 0.0]
                    g = sns.FacetGrid(r, row="Vd", col="dth")
                    g.map(sns.histplot, "fuel_consumption_gal__100mi")
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_facetgrid_row_vd_col_dtheadway_{pt_type}.png', dpi=600)
                    plt.close()
                    g = None

                    g = sns.PairGrid(r, x_vars=param_labels, y_vars=["fuel_consumption_gal__100mi"])
                    g.map(sns.scatterplot)
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_pair_plot_{pt_type}.png', dpi=600)
                    plt.close()
                    g = None

                    def my_violin(x, y, **kwargs):
                        ax = plt.gca()
                        ax.plot([np.min(x), np.max(x)], [base_fuel_consumption_gal__100mi] * 2, 'r:', label='baseline')
                        sns.violinplot(x=x, y=y, **kwargs)
                    g = sns.PairGrid(r, x_vars=param_labels, y_vars=["fuel_consumption_gal__100mi"])
                    g.map(my_violin) # sns.violinplot)
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_violin_{pt_type}.png', dpi=600)
                    plt.close()
                    g = None
                
                import plotly.express as px
                another = (
                    RR[[
                        "PT", "Vd", "dth",
                        "s", SM_DELTA, "a", "b",
                        "fuel_consumption_gal__100mi",
                    ]]
                ).copy()
                fig = px.parallel_coordinates(
                    another,
                    color="fuel_consumption_gal__100mi",
                    labels={
                        'PT': 'Powertrain',
                        'Vd': 'Desired Speed (m/s)',
                        'dth': 'Headway (s)',
                        's': 'Min Gap (m)',
                    }
                )
                fig.show()
                plt.close()

                print('DONE!')
            if RUST_AVAILABLE and USE_RUST:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import pandas as pd
                import seaborn as sns
                import pathlib as pl
                sns.set()

                datafile_name = 'test_sweeping_idm_parameters.csv'
                save_dir = pl.Path('test_output') / 'rust'
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)
                datafile_path = save_dir / datafile_name

                results = None
                # CYCLE STATISICS
                stopped_dict = fastsim.cycle.make_cycle([0, 120], [0, 0])
                udds_dict = fastsim.cycle.Cycle.from_file('udds').get_cyc_dict()
                udds = fastsim.cycle.Cycle.from_dict(
                    fastsim.cycle.resample(
                        fastsim.cycle.concat([udds_dict, stopped_dict]),
                        new_dt=1.0,
                    )
                ).to_rust()
                max_speed_m__s = np.max(udds.mps)
                id_by_pt_type = {
                    'CONV': 2,
                    'HEV': 10,
                    'BEV': 19,
                }
                mask = fastsim.cycle.trapz_step_distances(udds) > 0
                avg_speed_when_moving_m__s = fastsim.cycle.trapz_step_distances(udds)[mask].sum() / np.array(udds.dt_s)[mask].sum()
                # DESIGN VARIABLES
                veh_pt_types = ['CONV', 'HEV', 'BEV']
                dt_headways = [0.5, 1.0, 4.0, 5.0, 10.0]
                v_desireds = [avg_speed_when_moving_m__s, max_speed_m__s]
                s_mins = np.linspace(1.0, 10.0, num=5, endpoint=True)
                deltas = [1.0, 2.0, 4.0, 6.0]
                a_accels = np.linspace(0.5, 2.5, num=5, endpoint=True)
                b_accels = np.linspace(1.0, 3.0, num=5, endpoint=True)
                # GENERATE DATA (and PLOTS)
                if not pl.Path(datafile_path).exists():
                    num_runs = len(veh_pt_types) + (
                        len(veh_pt_types) * len(dt_headways) * len(v_desireds)
                        * len(s_mins) * len(deltas) * len(a_accels) * len(b_accels)
                    )
                    print(f"TOTAL RUNS: {num_runs}")
                    results = {
                        'pt_type': [],
                        'dt_headway_s': [],
                        'v_desired_m__s': [],
                        's_min_m': [],
                        'delta': [],
                        'a_m__s2': [],
                        'b_m__s2': [],
                        'fuel_economy_mpgge': [],
                        'fuel_consumption_gal__100mi': [],
                        'distance_m': [],
                        'distance_short_m': [],
                        'battery_kWh_per_mi': [],
                    }
                    idx = 0
                    fmt_str = "{:.2f}"
                    def fmt(n):
                        return fmt_str.format(n).replace('.', '_')
                    for pt_type in veh_pt_types:
                        veh = fastsim.vehicle.Vehicle.from_vehdb(id_by_pt_type[pt_type]).to_rust()
                        sd = fastsim.simdrive.RustSimDrive(udds, veh)
                        sd.sim_drive()
                        base_distance_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                        if sd.mpgge == 0.0 or np.isnan(sd.mpgge):
                            base_fuel_consumption_gal__100mi = 0.0
                        else: 
                            base_fuel_consumption_gal__100mi = 100.0 / sd.mpgge
                        results['pt_type'].append(pt_type)
                        results['dt_headway_s'].append(0.0)
                        results['v_desired_m__s'].append(0.0)
                        results['s_min_m'].append(0.0)
                        results['delta'].append(0.0)
                        results['a_m__s2'].append(0.0)
                        results['b_m__s2'].append(0.0)
                        results['fuel_economy_mpgge'].append(sd.mpgge)
                        results['battery_kWh_per_mi'].append(sd.battery_kwh_per_mi)
                        if sd.mpgge > 0.0:
                            results['fuel_consumption_gal__100mi'].append(base_fuel_consumption_gal__100mi)
                        elif sd.battery_kwh_per_mi > 0.0:
                            gge__mi = sd.battery_kwh_per_mi / sd.props.kwh_per_gge
                            results['fuel_consumption_gal__100mi'].append(100.0 * gge__mi)
                        else:
                            results['fuel_consumption_gal__100mi'].append(-1.0)
                        results['distance_m'].append(base_distance_m)
                        results['distance_short_m'].append(0.0)
                        for dt_h_s in dt_headways:
                            for v_d_m__s in v_desireds:
                                for s_min_m in s_mins:
                                    for delta in deltas:
                                        for a_m__s2 in a_accels:
                                            for b_m__s2 in b_accels:
                                                key = (
                                                    "run_"
                                                    + f"pt{pt_type}_"
                                                    + f"dth{fmt(dt_h_s)}_"
                                                    + f"vd{fmt(v_d_m__s)}_"
                                                    + f"s0{fmt(s_min_m)}_"
                                                    + f"delta{fmt(delta)}_"
                                                    + f"a{fmt(a_m__s2)}_"
                                                    + f"b{fmt(b_m__s2)}"
                                                )
                                                idx += 1
                                                if idx % 10 == 0:
                                                    print(f"Running {idx}...")
                                                sd = fastsim.simdrive.RustSimDrive(udds, veh)
                                                sim_params = sd.sim_params
                                                sim_params.idm_allow = True
                                                sim_params.idm_minimum_gap_m = s_min_m
                                                sim_params.idm_delta = delta
                                                sim_params.idm_accel_m_per_s2 = a_m__s2
                                                sim_params.idm_decel_m_per_s2 = b_m__s2
                                                sim_params.idm_v_desired_m_per_s = v_d_m__s
                                                sim_params.idm_dt_headway_s = dt_h_s
                                                sd.sim_params = sim_params
                                                sd.sim_drive()
                                                results['pt_type'].append(pt_type)
                                                results['dt_headway_s'].append(dt_h_s)
                                                results['v_desired_m__s'].append(v_d_m__s)
                                                results['s_min_m'].append(s_min_m)
                                                results['delta'].append(delta)
                                                results['a_m__s2'].append(a_m__s2)
                                                results['b_m__s2'].append(b_m__s2)
                                                results['fuel_economy_mpgge'].append(sd.mpgge)
                                                results['battery_kWh_per_mi'].append(sd.battery_kwh_per_mi)
                                                if not np.isnan(sd.mpgge) and sd.mpgge > 0.0:
                                                    results['fuel_consumption_gal__100mi'].append(100.0 / sd.mpgge)
                                                elif sd.battery_kwh_per_mi > 0.0:
                                                    pass
                                                    gge__mi = sd.battery_kwh_per_mi / sd.props.kwh_per_gge
                                                    results['fuel_consumption_gal__100mi'].append(100.0 * gge__mi)
                                                else:
                                                    results['fuel_consumption_gal__100mi'].append(-1.0)
                                                dist_m = fastsim.cycle.trapz_step_distances(sd.cyc).sum()
                                                results['distance_m'].append(dist_m)
                                                results['distance_short_m'].append(base_distance_m - dist_m)
                                                # create a plot by key and save it
                                                from fastsim.tests.test_coasting import make_coasting_plot 
                                                save_path = save_dir / f'{key}.png'
                                                try:
                                                    make_coasting_plot(
                                                        sd.cyc0,
                                                        sd.cyc,
                                                        use_mph=False,
                                                        title=key,
                                                        save_file=save_path,
                                                        do_show=False,
                                                        gap_offset_m=s_min_m,
                                                    )
                                                except Exception as err: 
                                                    with open(save_dir / 'run-log.txt', 'a') as f:
                                                        f.write(f'issue creating plot for {key}: {err}')
                                                if False:
                                                    # Pickles the runs; note: takes a lot of disk space...
                                                    import pickle
                                                    sd_path = save_dir / f'{key}.pickle'
                                                    with open(sd_path, 'wb') as f:
                                                        sd_copy = fastsim.simdrive.copy_sim_drive(sd)
                                                        pickle.dump(sd_copy, f)
                    print(f"idx: {idx}")
                    results = pd.DataFrame(results)
                    results.to_csv(datafile_path)
                else:
                    results = pd.read_csv(datafile_path)
                self.assertTrue(results is not None)
                results['distance_short_pct'] = (
                    (results['distance_short_m'] * 100.0)
                    / (results['distance_short_m'] + results['distance_m'])
                )
                # seaborn plots
                SM_DELTA = '\u03b4'
                param_labels = [
                    #'pt_type', 'v_desired_m__s', 'dt_headway_s', 's_min_m', 'delta', 'a_m__s2', 'b_m__s2'
                    'PT', 'Vd', 'dth', 's', SM_DELTA, 'a', 'b'
                ]
                RR = results.rename(columns={
                    'pt_type':'PT',
                    'v_desired_m__s': 'Vd',
                    'dt_headway_s':'dth',
                    's_min_m': 's',
                    'delta': SM_DELTA,
                    'a_m__s2': 'a',
                    'b_m__s2': 'b',
                })
                for hue_item in param_labels:
                    g = sns.displot(
                        data=RR,
                        x="fuel_consumption_gal__100mi",
                        hue=hue_item,
                        multiple="stack",
                    )
                    g.ax.plot([RR['fuel_consumption_gal__100mi'][0]]*2, [0, 1000], 'r-', linewidth=2)
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_distplot_{hue_item}.png', dpi=300)
                    plt.close()
                    g = None
                # PERCENTAGE DISTANCE SHORT PLOT
                g = sns.displot(
                    data=RR,
                    x='distance_short_pct',
                    multiple='stack'
                )
                g.savefig(save_dir / 'test_sweeping_idm_parameters_distplot_dist_short.png', dpi=300)
                plt.close()
                g = None
                # FacetGrid HistPlot of pt_type x v_desired_m__s for fuel_consumption
                g = sns.FacetGrid(RR, row="PT", col="Vd", hue='dth')
                g.map(sns.histplot, "fuel_consumption_gal__100mi")
                g.savefig(
                    save_dir / 'test_sweeping_idm_parameters_facetgrid_row_pt_type_col_v_desired_for_fuel_consumption.png',
                    dpi=600)
                plt.close()
                g = None
                # FacetGrid HistPlot of pt_type x dt_headway_s for fuel_consumption
                g = sns.FacetGrid(RR, row="PT", col="dth", hue='Vd')
                g.map(sns.histplot, "fuel_consumption_gal__100mi")
                g.savefig(
                    save_dir / 'test_sweeping_idm_parameters_facetgrid_row_pt_type_col_dt_headway_for_fuel_consumption.png',
                    dpi=600)
                plt.close()
                g = None
                # FacetGrid HistPlot of pt_type x dt_headway_s for fuel_consumption
                g = sns.FacetGrid(RR, row="Vd", col="dth", hue='PT')
                g.map(sns.histplot, "fuel_consumption_gal__100mi")
                g.savefig(
                    save_dir / 'test_sweeping_idm_parameters_facetgrid_row_v_desired_col_dt_headway_for_fuel_consumption.png',
                    dpi=600)
                plt.close()
                g = None
                for pt_type in veh_pt_types:
                    param_labels = ['Vd', 'dth', 's', SM_DELTA, 'a', 'b']
                    R = RR[RR['PT'] == pt_type]
                    r0 = R[R["Vd"] == 0.0]
                    base_fuel_consumption_gal__100mi = float(r0["fuel_consumption_gal__100mi"].values[0])
                    print(f'Baseline Fuel Consumption for {pt_type}: {base_fuel_consumption_gal__100mi} gal/100mi')
                    r = R[R["Vd"] != 0.0]
                    g = sns.FacetGrid(r, row="Vd", col="dth")
                    g.map(sns.histplot, "fuel_consumption_gal__100mi")
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_facetgrid_row_vd_col_dtheadway_{pt_type}.png', dpi=600)
                    plt.close()
                    g = None

                    g = sns.PairGrid(r, x_vars=param_labels, y_vars=["fuel_consumption_gal__100mi"])
                    g.map(sns.scatterplot)
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_pair_plot_{pt_type}.png', dpi=600)
                    plt.close()
                    g = None

                    def my_violin(x, y, **kwargs):
                        ax = plt.gca()
                        ax.plot([np.min(x), np.max(x)], [base_fuel_consumption_gal__100mi] * 2, 'r:', label='baseline')
                        sns.violinplot(x=x, y=y, **kwargs)
                    g = sns.PairGrid(r, x_vars=param_labels, y_vars=["fuel_consumption_gal__100mi"])
                    g.map(my_violin) # sns.violinplot)
                    g.savefig(save_dir / f'test_sweeping_idm_parameters_violin_{pt_type}.png', dpi=600)
                    plt.close()
                    g = None
                
                import plotly.express as px
                another = (
                    RR[[
                        "PT", "Vd", "dth",
                        "s", SM_DELTA, "a", "b",
                        "fuel_consumption_gal__100mi",
                    ]]
                ).copy()
                fig = px.parallel_coordinates(
                    another,
                    color="fuel_consumption_gal__100mi",
                    labels={
                        'PT': 'Powertrain',
                        'Vd': 'Desired Speed (m/s)',
                        'dth': 'Headway (s)',
                        's': 'Min Gap (m)',
                    }
                )
                fig.show()
                plt.close()

                print('DONE!')

    def test_distance_based_grade_on_following(self):
        "Tests use of the IDM model for following"
        if USE_PYTHON:
            cyc = fastsim.cycle.make_cycle(
                [0.0 , 10.0 , 20.0 , 30.0 , 40.0 , 50.0 , 99.0],
                [0.0 , 10.0 , 10.0 , 10.0 , 10.0 ,  0.0 ,  0.0],
                [0.00,  0.00,  0.01, 0.00, -0.01 ,  0.0 ,  0.0],
            )
            cyc = fastsim.cycle.resample(cyc, new_dt=0.1, hold_keys_next={'grade'})
            cyc = fastsim.cycle.Cycle.from_dict(cyc)
            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            sd = fastsim.simdrive.SimDrive(cyc, veh)
            sd.sim_params.idm_allow = True
            sd.sim_params.idm_minimum_gap_m = 5.0
            sd.sim_params.idm_v_desired_m_per_s = 5.0
            sd.sim_drive()
            ts0 = sd.cyc0.time_s
            dds0 = sd.cyc0.dist_m
            # TODO: should use sd.cyc0.delta_elev_m but that currently is not as accurate
            hs0 = np.cumsum(np.cos(np.arctan(sd.cyc0.grade)) * fastsim.cycle.trapz_step_distances(sd.cyc0) * sd.cyc0.grade)
            ts1 = sd.cyc.time_s
            dds1 = sd.cyc.dist_m
            # TODO: should use sd.cyc0.delta_elev_m but that currently is not as accurate
            hs1 = np.cumsum(np.cos(np.arctan(sd.cyc.grade)) * fastsim.cycle.trapz_step_distances(sd.cyc) * sd.cyc.grade)
            self.assertAlmostEqual(sum(dds0), sum(dds1), places=-1)
            self.assertAlmostEqual(hs0[-1], hs1[-1], places=6)
            def calc_dist_of_peak_elev(dds, hs):
                d = 0.0
                d_for_peak = 0.0
                h_max = None
                for dd, h in zip(dds, hs):
                    d += dd
                    if h_max is None or h > h_max:
                        d_for_peak = d
                        h_max = h
                return d_for_peak
            def calc_time_of_peak_elev(ts, hs):
                t = 0.0
                t_for_peak = 0.0
                h_max = None
                for t, h in zip(ts, hs):
                    if h_max is None or h > h_max:
                        t_for_peak = t
                        h_max = h
                return t_for_peak
            dist_at_peak0 = calc_dist_of_peak_elev(dds0, hs0)
            dist_at_peak1 = calc_dist_of_peak_elev(dds1, hs1)
            self.assertAlmostEqual(dist_at_peak0, dist_at_peak1, places=0)
            time_at_peak0 = calc_time_of_peak_elev(ts0, hs0)
            time_at_peak1 = calc_time_of_peak_elev(ts1, hs1)
            self.assertTrue((time_at_peak0 + 10.0) < time_at_peak1)
            if False:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.time_s, sd.cyc0.delta_elev_m, 'r-', label='cyc0')
                ax.plot(sd.cyc.time_s, sd.cyc.delta_elev_m, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Elevation (m)')
                fig.savefig('junk-elev.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.time_s, sd.cyc0.mps, 'r-', label='cyc0')
                ax.plot(sd.cyc.time_s, sd.cyc.mps, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Speed (m/s)')
                fig.savefig('junk-mps.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.dist_m.cumsum(), sd.cyc0.delta_elev_m, 'r-', label='cyc0')
                ax.plot(sd.cyc.dist_m.cumsum(), sd.cyc.delta_elev_m, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Distance Traveled (s)')
                ax.set_ylabel('Elevation (m)')
                fig.savefig('junk-elev-by-dist.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.dist_m.cumsum(), sd.cyc0.mps, 'r-', label='cyc0')
                ax.plot(sd.cyc.dist_m.cumsum(), sd.cyc.mps, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Distance Traveled (m)')
                ax.set_ylabel('Speed (m/s)')
                fig.savefig('junk-mps-by-dist.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.time_s, sd.cyc0.dist_m.cumsum(), 'r-', label='cyc0')
                ax.plot(sd.cyc.time_s, sd.cyc.dist_m.cumsum(), 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Distance Traveled (m)')
                fig.savefig('junk-distance-by-time.png', dpi=600)
                plt.close()
                self.assertTrue((sd.cyc.dist_m == sd.dist_m).all())
                self.assertTrue((sd.mps_ach == sd.cyc.mps).all())
        if RUST_AVAILABLE and USE_RUST:
            cyc = fastsim.cycle.make_cycle(
                [0.0 , 10.0 , 20.0 , 30.0 , 40.0 , 50.0 , 99.0],
                [0.0 , 10.0 , 10.0 , 10.0 , 10.0 ,  0.0 ,  0.0],
                [0.01,  0.01,  0.0 , -0.01, -0.01,  0.0 ,  0.0],
            )
            cyc = fastsim.cycle.resample(cyc, new_dt=0.1, hold_keys={'grade'})
            cyc = fastsim.cycle.Cycle.from_dict(cyc).to_rust()
            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            sd = fastsim.simdrive.RustSimDrive(cyc, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                idm_allow=True,
                idm_minimum_gap_m=5.0,
                idm_v_desired_m_per_s=5.0,
            )
            sd.sim_drive()
            ts0 = sd.cyc0.time_s
            dds0 = sd.cyc0.dist_m
            # TODO: should use sd.cyc0.delta_elev_m but that currently is not as accurate
            hs0 = np.cumsum(np.cos(np.arctan(sd.cyc0.grade)) * fastsim.cycle.trapz_step_distances(sd.cyc0) * sd.cyc0.grade)
            ts1 = sd.cyc.time_s
            dds1 = sd.cyc.dist_m
            # TODO: should use sd.cyc0.delta_elev_m but that currently is not as accurate
            hs1 = np.cumsum(np.cos(np.arctan(sd.cyc.grade)) * fastsim.cycle.trapz_step_distances(sd.cyc) * sd.cyc.grade)
            self.assertAlmostEqual(sum(dds0), sum(dds1), places=-1)
            self.assertAlmostEqual(hs0[-1], hs1[-1], places=6)
            def calc_dist_of_peak_elev(dds, hs):
                d = 0.0
                d_for_peak = 0.0
                h_max = None
                for dd, h in zip(dds, hs):
                    d += dd
                    if h_max is None or h > h_max:
                        d_for_peak = d
                        h_max = h
                return d_for_peak
            def calc_time_of_peak_elev(ts, hs):
                t = 0.0
                t_for_peak = 0.0
                h_max = None
                for t, h in zip(ts, hs):
                    if h_max is None or h > h_max:
                        t_for_peak = t
                        h_max = h
                return t_for_peak
            dist_at_peak0 = calc_dist_of_peak_elev(dds0, hs0)
            dist_at_peak1 = calc_dist_of_peak_elev(dds1, hs1)
            self.assertAlmostEqual(dist_at_peak0, dist_at_peak1, places=0)
            time_at_peak0 = calc_time_of_peak_elev(ts0, hs0)
            time_at_peak1 = calc_time_of_peak_elev(ts1, hs1)
            self.assertTrue((time_at_peak0 + 10.0) < time_at_peak1)
            if False:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.time_s, sd.cyc0.delta_elev_m, 'r-', label='cyc0')
                ax.plot(sd.cyc.time_s, sd.cyc.delta_elev_m, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Elevation (m)')
                fig.savefig('junk-rust-elev.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.time_s, sd.cyc0.mps, 'r-', label='cyc0')
                ax.plot(sd.cyc.time_s, sd.cyc.mps, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Speed (m/s)')
                fig.savefig('junk-rust-mps.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(np.array(sd.cyc0.dist_m).cumsum(), sd.cyc0.delta_elev_m, 'r-', label='cyc0')
                ax.plot(np.array(sd.cyc.dist_m).cumsum(), sd.cyc.delta_elev_m, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Distance Traveled (s)')
                ax.set_ylabel('Elevation (m)')
                fig.savefig('junk-rust-elev-by-dist.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(np.array(sd.cyc0.dist_m).cumsum(), sd.cyc0.mps, 'r-', label='cyc0')
                ax.plot(np.array(sd.cyc.dist_m).cumsum(), sd.cyc.mps, 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Distance Traveled (m)')
                ax.set_ylabel('Speed (m/s)')
                fig.savefig('junk-rust-mps-by-dist.png', dpi=600)
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(sd.cyc0.time_s, np.array(sd.cyc0.dist_m).cumsum(), 'r-', label='cyc0')
                ax.plot(sd.cyc.time_s, np.array(sd.cyc.dist_m).cumsum(), 'k.', label='cyc')
                ax.legend()
                ax.set_xlabel('Elapsed Time (s)')
                ax.set_ylabel('Distance Traveled (m)')
                fig.savefig('junk-rust-distance-by-time.png', dpi=600)
                plt.close()
                self.assertTrue((np.array(sd.cyc.dist_m) == np.array(sd.dist_m)).all())
                self.assertTrue((np.array(sd.mps_ach) == np.array(sd.cyc.mps)).all())

if __name__ == '__main__':
    unittest.main()
