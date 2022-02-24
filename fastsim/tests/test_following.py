"""
Tests that check the drive cycle modification functionality.
"""
import unittest

import numpy as np

import fastsim


DO_PLOTS = False


class TestFollowing(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_gap_m = 5.0
        trapz = fastsim.cycle.make_cycle(
            [0.0, 10.0, 45.0, 55.0, 150.0],
            [0.0, 20.0, 20.0, 0.0, 0.0],
        )
        trapz = fastsim.cycle.resample(trapz, new_dt=1.0)
        self.trapz = fastsim.cycle.Cycle(cyc_dict=trapz)
        self.veh = fastsim.vehicle.Vehicle(5, verbose=False)
        self.veh.lead_offset_m = self.initial_gap_m
        # sd0 is for reference to an unchanged, no-following simdrive
        self.sd0 = fastsim.simdrive.SimDriveClassic(self.trapz, self.veh)
        self.sd0.sim_params.verbose = False
        self.sd = fastsim.simdrive.SimDriveClassic(self.trapz, self.veh)
        self.sd.sim_params.follow_allow = True
        self.sd.sim_params.verbose = False
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_that_we_have_a_gap_between_us_and_the_lead_vehicle(self):
        "A positive gap should exist between us and the lead vehicle"
        self.assertTrue(self.sd.sim_params.follow_allow)
        self.veh.lead_speed_coef_s = 0.0
        self.veh.lead_offset_m = self.initial_gap_m
        self.veh.lead_accel_coef_s2 = 0.0
        self.sd.sim_drive()
        self.assertTrue(self.sd.sim_params.follow_allow)
        gaps_m = self.sd.gap_to_lead_vehicle_m
        if DO_PLOTS:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(self.sd.cyc0.cycSecs, gaps_m, 'k.')
            ax.set_xlabel('Elapsed Time (s)')
            ax.set_ylabel('Gap (m)')
            fig.tight_layout()
            save_file = "test_that_we_have_a_gap_between_us_and_the_lead_vehicle__0.png"
            fig.savefig(save_file, dpi=300)
            plt.close()
        self.assertTrue((gaps_m == 5.0).all())
        self.assertAlmostEqual(self.sd.cyc0.cycDistMeters_v2.sum(), self.sd.cyc.cycDistMeters_v2.sum())

    def test_that_the_gap_changes_over_the_cycle(self):
        "Ensure that our gap calculation is doing something"
        self.sd.sim_drive()
        gaps_m = self.sd.gap_to_lead_vehicle_m
        if DO_PLOTS:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(self.sd.cyc0.cycSecs, gaps_m, 'k.')
            ax.set_xlabel('Elapsed Time (s)')
            ax.set_ylabel('Gap (m)')
            fig.tight_layout()
            save_file = "test_that_the_gap_changes_over_the_cycle__0.png"
            fig.savefig(save_file, dpi=300)
            plt.close()
            from fastsim.tests.test_coasting import make_coasting_plot
            make_coasting_plot(
                self.sd.cyc0, self.sd.cyc,
                gap_offset_m=self.veh.lead_offset_m,
                title='test_that_the_gap_changes_over_the_cycle__1.png',
                save_file="test_that_the_gap_changes_over_the_cycle__1.png",
            )
        self.assertFalse((gaps_m == 5.0).all())
        self.assertAlmostEqual(
            self.sd.cyc0.cycDistMeters_v2.sum(),
            self.sd.cyc.cycDistMeters_v2.sum(),
            places=-1,
            msg='Distance should be about the same')
        self.assertTrue((gaps_m > 0.0).all(), msg='We cannot pass the lead vehicle')
        self.assertTrue(
            (gaps_m > (self.initial_gap_m - 0.5)).all(),
            msg='We cannot get closer than the initial gap distance')


    def test_that_following_works_over_parameter_sweep(self):
        "We're going to sweep through all of the parameters and see how it goes"
        if DO_PLOTS:
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            sns.set()

            accel_coefs = np.linspace(-10.0, 10.0, num=5, endpoint=True)
            speed_coefs = np.linspace(-10.0, 10.0, num=5, endpoint=True)
            init_offsets= np.linspace(0.0, 100.0, num=5, endpoint=True)
            udds = fastsim.cycle.Cycle('udds')
            veh = fastsim.vehicle.Vehicle(5, verbose=False)
            sd = fastsim.simdrive.SimDriveClassic(udds, veh)
            sd.sim_drive()
            base_distance_m = sd.cyc.cycDistMeters_v2.sum()
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
                        udds = fastsim.cycle.Cycle('udds')
                        veh = fastsim.vehicle.Vehicle(5, verbose=False)
                        veh.lead_accel_coef_s2 = accel_coef_s2
                        veh.lead_speed_coef_s = speed_coef_s
                        veh.lead_offset_m = initial_offset_m
                        sd = fastsim.simdrive.SimDriveClassic(udds, veh)
                        sd.sim_params.follow_allow = True
                        sd.sim_drive()
                        results['accel_coef_s2'].append(accel_coef_s2)
                        results['speed_coef_s'].append(speed_coef_s)
                        results['init_offset_m'].append(initial_offset_m)
                        results['fuel_economy_mpgge'].append(sd.mpgge)
                        results['fuel_consumption_gal__100mi'].append(100.0 / sd.mpgge)
                        dist_m = sd.cyc.cycDistMeters_v2.sum()
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