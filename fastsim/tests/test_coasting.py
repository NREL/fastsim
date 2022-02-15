"""
Tests that check the drive cycle modification functionality.
"""
import unittest

import numpy as np

import fastsim


MPH_TO_MPS = 1.0 / fastsim.params.mphPerMps


class TestCoasting(unittest.TestCase):
    def setUp(self) -> None:
        # create a trapezoidal trip shape
        # initial ramp: d(t=10s) = 100 meters distance
        # distance by time in constant speed region = d(t) = 100m + (t - 10s) * 20m/s 
        # distance of stop: 100m + (45s - 10s) * 20m/s + 0.5 * (55s - 45s) * 20m/s = 900m
        self.distance_of_stop_m = 900.0
        trapz = fastsim.cycle.make_cycle(
            [0.0, 10.0, 45.0, 55.0, 100.0],
            #[0.0, 40.0 * MPH_TO_MPS, 40.0 * MPH_TO_MPS, 0.0, 0.0],
            [0.0, 20.0, 20.0, 0.0, 0.0],
        )
        trapz = fastsim.cycle.resample(trapz, new_dt=1.0)
        self.trapz = fastsim.cycle.Cycle(cyc_dict=trapz)
        self.veh = fastsim.vehicle.Vehicle(5)
        self.sim_drive = fastsim.simdrive.SimDriveClassic(self.trapz, self.veh)
        self.sim_drive_coast = fastsim.simdrive.SimDriveClassic(self.trapz, self.veh)
        self.sim_drive_coast.sim_params.allow_coast = True
        self.sim_drive_coast.sim_params.coast_start_speed_m__s = 17.0
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_cycle_reported_distance_traveled_m(self):
        ""
        # At the entering of constant-speed region
        idx = 10
        expected_time_s = 10.0
        t = self.trapz.cycSecs[idx]
        self.assertAlmostEqual(expected_time_s, t)
        expected_distance_m = 100.0
        dist_m = self.trapz.cycDistMeters_v2[:(idx + 1)].sum()
        self.assertAlmostEqual(expected_distance_m, dist_m)
        # At t=20s
        idx = 20
        expected_time_s = 20.0
        t = self.trapz.cycSecs[idx]
        self.assertAlmostEqual(expected_time_s, t)
        expected_distance_m = 300.0 # 100m + (20s - 10s) * 20m/s
        dist_m = self.trapz.cycDistMeters_v2[:(idx + 1)].sum()
        self.assertAlmostEqual(expected_distance_m, dist_m)
        dts = fastsim.cycle.calc_distance_to_next_stop(dist_m, self.trapz)
        dts_expected_m = 900 - dist_m
        self.assertAlmostEqual(dts_expected_m, dts)

    def test_cycle_modifications_with_constant_jerk(self):
        ""
        idx = 20
        n = 10
        accel = -1.0
        jerk = 0.1
        trapz = self.trapz.copy()
        fastsim.cycle.modify_cycle_with_trajectory(
            trapz, idx, n, jerk, -1.0
        )
        self.assertNotEqual(self.trapz.cycMps[idx], trapz.cycMps[idx])
        self.assertEqual(len(self.trapz.cycMps), len(trapz.cycMps))
        self.assertTrue(self.trapz.cycMps[idx] > trapz.cycMps[idx])
        v0 = trapz.cycMps[idx-1]
        v = v0
        a = accel
        for i in range(len(self.trapz.cycSecs)):
            msg = f"i: {i}; idx: {idx}; idx+n: {idx+n}"
            if i < idx or i >= idx+n:
                self.assertEqual(self.trapz.cycMps[i], trapz.cycMps[i], msg)
            else:
                dt = trapz.secs[idx]
                a_expected = fastsim.cycle.accel_for_constant_jerk(i - idx, accel, jerk, dt)
                a = accel + (i - idx) * jerk * dt
                v += a * dt
                msg += f" a: {a}, v: {v}, dt: {dt}"
                self.assertAlmostEqual(a_expected, a, msg=msg)
                self.assertAlmostEqual(v, trapz.cycMps[i], msg=msg)
    
    def test_that_cycle_modifications_work_as_expected(self):
        ""
        idx = 20
        n = 10
        accel = -1.0
        jerk = 0.0
        trapz = self.trapz.copy()
        fastsim.cycle.modify_cycle_with_trajectory(
            trapz, idx, n, jerk, -1.0
        )
        self.assertNotEqual(self.trapz.cycMps[idx], trapz.cycMps[idx])
        self.assertEqual(len(self.trapz.cycMps), len(trapz.cycMps))
        self.assertTrue(self.trapz.cycMps[idx] > trapz.cycMps[idx])
        for i in range(len(self.trapz.cycSecs)):
            msg = f"i: {i}; idx: {idx}; idx+n: {idx+n}"
            if i < idx or i >= idx+n:
                self.assertEqual(self.trapz.cycMps[i], trapz.cycMps[i])
            else:
                self.assertAlmostEqual(
                    self.trapz.cycMps[idx-1] + (accel * (i - idx + 1)),
                    trapz.cycMps[i]
                )
    
    def test_that_we_can_coast(self):
        "Test the standard interface to Eco-Approach for 'free coasting'"
        self.assertFalse(self.sim_drive.impose_coast.any(), "All impose_coast starts out False")
        while self.sim_drive_coast.i < len(self.trapz.cycSecs):
            #self.sim_drive.sim_drive_step()
            self.sim_drive_coast.sim_drive_step()
        #max_trace_miss_m__s = np.absolute(self.trapz.cycMps - self.sim_drive.mpsAch).max()
        max_trace_miss_coast_m__s = np.absolute(self.trapz.cycMps - self.sim_drive_coast.mpsAch).max()
        #self.assertTrue(max_trace_miss_m__s < 0.01, f"Max trace miss: {max_trace_miss_m__s} m/s")
        self.assertTrue(max_trace_miss_coast_m__s > 1.0, f"Max trace miss: {max_trace_miss_coast_m__s} m/s")
        #self.assertFalse(self.sim_drive.impose_coast.any())
        self.assertFalse(self.sim_drive_coast.impose_coast[0])
        if True:
            do_show = False
            import matplotlib.pyplot as plt
            (fig, ax) = plt.subplots()
            ax.plot(self.sim_drive_coast.cyc0.cycSecs, self.sim_drive_coast.cyc0.cycMps, 'gray', label='shadow-trace')
            ax.plot(self.sim_drive_coast.cyc.cycSecs, self.sim_drive_coast.cyc.cycMps, 'blue', label='coast')
            ax.plot(self.sim_drive_coast.cyc.cycSecs, self.sim_drive_coast.cyc.cycMps, 'r.')
            ax.set_xlabel('Elapsed Time (s)')
            ax.set_ylabel('Speed (m/s)')
            ax.legend()
            fig.tight_layout()
            fig.savefig('coasting-by-time.png', dpi=300)
            if not do_show:
                plt.close()
            (fig, ax) = plt.subplots()
            ax.plot(self.sim_drive_coast.cyc0.cycDistMeters.cumsum(), self.sim_drive_coast.cyc0.cycMps, 'gray', label='shadow-trace')
            ax.plot(self.sim_drive_coast.cyc.cycDistMeters.cumsum(), self.sim_drive_coast.cyc.cycMps, 'blue', label='coast')
            ax.plot(self.sim_drive_coast.cyc.cycDistMeters.cumsum(), self.sim_drive_coast.cyc.cycMps, 'r.')
            ax.set_title('Coasting over Trapezoidal Cycle')
            ax.set_xlabel('Distance Traveled (m)')
            ax.set_ylabel('Speed (m/s)')
            ax.legend()
            fig.tight_layout()
            fig.savefig('coasting-by-distance.png', dpi=300)
            if do_show:
                plt.show()
            else:
                plt.close()
            fig = None
            ax = None
            (fig, ax) = plt.subplots()
            ax.plot(self.sim_drive_coast.cyc0.cycSecs, self.sim_drive_coast.cyc0.cycDistMeters.cumsum(), 'gray', label='shadow-trace')
            ax.plot(self.sim_drive_coast.cyc.cycSecs, self.sim_drive_coast.cyc.cycDistMeters.cumsum(), 'blue', label='coast')
            ax.plot(self.sim_drive_coast.cyc.cycSecs, self.sim_drive_coast.cyc.cycDistMeters.cumsum(), 'r.')
            ax.set_title('Coasting over Trapezoidal Cycle')
            ax.set_xlabel('Elapsed Time (s)')
            ax.set_ylabel('Distance Traveled (m)')
            ax.legend()
            fig.tight_layout()
            fig.savefig('coasting-distance-by-time.png', dpi=300)
            plt.close()
            print(f'Distance Traveled for Coasting Vehilce: {self.sim_drive_coast.distMeters.sum()} m')
            print(f'Distance Traveled for Cycle           : {self.sim_drive.cyc0.cycDistMeters.sum()} m')

    def test_eco_approach_modeling(self):
        "Test a simplified model of eco-approach"
        self.sim_drive_coast.sim_drive()
        self.assertFalse(self.sim_drive_coast.impose_coast.all(), "Assert we are not always in coast")
        self.assertTrue(self.sim_drive_coast.impose_coast.any(), "Assert we are at least sometimes in coast")
        max_trace_miss_coast_m__s = np.absolute(self.trapz.cycMps - self.sim_drive_coast.mpsAch).max()
        self.assertTrue(max_trace_miss_coast_m__s > 1.0, "Assert we deviate from the shadow trace")
        self.assertTrue(self.sim_drive_coast.mphAch.max() > 20.0, "Assert we at least reach 20 mph")
        # TODO: can we increase the precision of matching?
        self.assertAlmostEqual(
            self.trapz.cycDistMeters.sum(),
            self.sim_drive_coast.distMeters.sum(), 0,
            "Assert the end distances are equal\n" +
            f"Got {self.trapz.cycDistMeters.sum()} m and {self.sim_drive_coast.distMeters.sum()} m")

    def test_consistency_of_constant_jerk_trajectory(self):
        "Confirm that acceleration, speed, and distances are as expected for constant jerk trajectory"
        n = 10 # ten time-steps
        v0 = 15.0
        vr = 7.5
        d0 = 0.0
        dr = 120.0
        dt = 1.0
        trajectory = fastsim.cycle.calc_constant_jerk_trajectory(n, d0, v0, dr, vr, dt)
        a0 = trajectory['accel_m__s2']
        k = trajectory['jerk_m__s3']
        v = v0
        d = d0
        a = a0
        vs = [v0]
        for n in range(n):
            a_expected = fastsim.cycle.accel_for_constant_jerk(n, a0, k, dt)
            v_expected = fastsim.cycle.speed_for_constant_jerk(n, v0, a0, k, dt)
            d_expected = fastsim.cycle.dist_for_constant_jerk(n, d0, v0, a0, k, dt)
            if n > 0:
                d += dt * (v + v + a * dt) / 2.0
                v += a * dt
            # acceleration is the constant acceleration for the NEXT time-step
            a = a0 + n * k * dt
            self.assertAlmostEqual(a, a_expected)
            self.assertAlmostEqual(v, v_expected)
            self.assertAlmostEqual(d, d_expected)

        print(f"trajectory: {str(trajectory)}")

    def test_that_final_speed_of_cycle_modification_matches_trajectory_calcs(self):
        ""
        trapz = self.trapz.copy()
        idx = 20
        n = 20
        d0 = self.trapz.cycDistMeters[:idx].sum()
        v0 = self.trapz.cycMps[idx-1]
        dt = self.trapz.secs[idx]
        brake_decel_m__s2 = 2.5
        dts0 = fastsim.cycle.calc_distance_to_next_stop(d0, trapz)
        # speed at which friction braking initiates (m/s)
        brake_start_speed_m__s = 7.5
        # distance to brake (m)
        dtb = 0.5 * brake_start_speed_m__s * brake_start_speed_m__s / brake_decel_m__s2
        dtbi0 = dts0 - dtb
        trajectory = fastsim.cycle.calc_constant_jerk_trajectory(n, d0, v0, d0 + dtbi0, brake_start_speed_m__s, dt)
        print(f"TEST: trajectory: {str(trajectory)}")
        final_speed_m__s = fastsim.cycle.modify_cycle_with_trajectory(
            self.trapz,
            idx,
            n,
            trajectory['jerk_m__s3'],
            trajectory['accel_m__s2'])
        self.assertAlmostEqual(final_speed_m__s, brake_start_speed_m__s)

    def test_that_cycle_distance_reported_is_correct(self):
        ""
        # total distance
        d_expected = 900.0
        d_v1 = self.trapz.cycDistMeters.sum()
        d_v2 = self.trapz.cycDistMeters_v2.sum()
        self.assertAlmostEqual(d_expected, d_v1)
        self.assertAlmostEqual(d_expected, d_v2)
        # distance traveled between 0 s and 10 s
        d_expected = 100.0 # 0.5 * (0s - 10s) * 20m/s = 100m
        d_v1 = self.trapz.cycDistMeters[:11].sum()
        d_v2 = self.trapz.cycDistMeters_v2[:11].sum()
        # TODO: is there a way to get the distance from 0 to 10s using existing cycDistMeters system?
        self.assertNotEqual(d_expected, d_v1)
        self.assertAlmostEqual(d_expected, d_v2)
        # distance traveled between 10 s and 45 s
        d_expected = 700.0 # (45s - 10s) * 20m/s = 700m
        d_v1 = self.trapz.cycDistMeters[11:46].sum()
        d_v2 = self.trapz.cycDistMeters_v2[11:46].sum()
        self.assertAlmostEqual(d_expected, d_v1)
        self.assertAlmostEqual(d_expected, d_v2)
        # distance traveled between 45 s and 55 s
        d_expected = 100.0 # 0.5 * (45s - 55s) * 20m/s = 100m
        d_v1 = self.trapz.cycDistMeters[45:56].sum()
        d_v2 = self.trapz.cycDistMeters_v2[46:56].sum()
        # TODO: is there a way to get the distance from 45 to 55s using existing cycDistMeters system?
        self.assertNotEqual(d_expected, d_v1)
        self.assertAlmostEqual(d_expected, d_v2)
        # TRIANGLE RAMP SPEED CYCLE
        const_spd_cyc = fastsim.cycle.Cycle(
            cyc_dict=fastsim.cycle.resample(
                fastsim.cycle.make_cycle(
                    [0.0, 20.0],
                    [0.0, 20.0]
                ),
                new_dt=1.0
            )
        )
        expected_dist_m = 200.0 # 0.5 * 20m/s x 20s = 200m
        self.assertAlmostEqual(expected_dist_m, const_spd_cyc.cycDistMeters_v2.sum())
        self.assertNotEqual(expected_dist_m, const_spd_cyc.cycDistMeters.sum())

    def test_brake_trajectory(self):
        ""
        trapz = self.trapz.copy()
        brake_accel_m__s2 = -2.0
        idx = 30
        dt = 1.0
        v0 = trapz.cycMps[idx]
        # distance required to stop (m)
        expected_dts_m = 0.5 * v0 * v0 / abs(brake_accel_m__s2)
        tts_s = -v0 / brake_accel_m__s2
        n = int(np.ceil(tts_s / dt))
        fastsim.cycle.modify_cycle_adding_braking_trajectory(trapz, brake_accel_m__s2, idx+1)
        self.assertAlmostEqual(v0, trapz.cycMps[idx])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*dt, trapz.cycMps[idx+1])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*2*dt, trapz.cycMps[idx+2])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*3*dt, trapz.cycMps[idx+3])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*4*dt, trapz.cycMps[idx+4])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*5*dt, trapz.cycMps[idx+5])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*6*dt, trapz.cycMps[idx+6])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*7*dt, trapz.cycMps[idx+7])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*8*dt, trapz.cycMps[idx+8])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*9*dt, trapz.cycMps[idx+9])
        self.assertAlmostEqual(v0 + brake_accel_m__s2*10*dt, trapz.cycMps[idx+10])
        self.assertEqual(10, n)
        self.assertAlmostEqual(20.0, trapz.cycMps[idx+11])
        dts_m = trapz.cycDistMeters_v2[idx+1:idx+n+1].sum()
        self.assertAlmostEqual(expected_dts_m, dts_m)
        # Now try with a brake deceleration that doesn't devide evenly by time-steps
        trapz = self.trapz.copy()
        brake_accel_m__s2 = -1.75
        idx = 30
        dt = 1.0
        v0 = trapz.cycMps[idx]
        # distance required to stop (m)
        expected_dts_m = 0.5 * v0 * v0 / abs(brake_accel_m__s2)
        tts_s = -v0 / brake_accel_m__s2
        n = int(np.ceil(tts_s / dt))
        fastsim.cycle.modify_cycle_adding_braking_trajectory(trapz, brake_accel_m__s2, idx+1)
        self.assertAlmostEqual(v0, trapz.cycMps[idx])
        self.assertEqual(12, n)
        dts_m = trapz.cycDistMeters_v2[idx+1:idx+n+1].sum()
        self.assertAlmostEqual(expected_dts_m, dts_m)
