"""Test suite for cycle instantiation and manipulation."""
import time

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import fastsim as fsim
from fastsim import cycle, params, inspect_utils


def calc_distance_traveled_m(cyc, up_to=None):
    """
    Calculate the distance traveled in meters
    - cyc: a cycle dictionary
    - up_to: None or a positive number indicating a time in seconds. Will calculate the distance up-to that given time
    RETURN: Number, the distance traveled in meters
    """
    if up_to is None:
        return (np.diff(cyc['time_s']) * ((cyc['mps'][1:] + cyc['mps'][:-1])*0.5)).sum()
    dist = 0.0
    ts = cyc['time_s']
    avg_speeds = (cyc['mps'][1:] + cyc['mps'][:-1]) * 0.5
    for (t, dt_s, avg_speed_m__s) in zip(ts[1:], np.diff(ts), avg_speeds):
        if t > up_to:
            return dist
        dist += dt_s * avg_speed_m__s
    return dist


def dicts_are_equal(d1, d2, d1_name=None, d2_name=None):
    """Checks if dictionaries are equal
    - d1: dict
    - d2: dict
    - d1_name: None or string, the name used for dict 1 in messaging
    - d2_name: None or string, the name used for dict 1 in messaging
    RETURN: (boolean, (Array string)),
    Returns (True, []) if the dictionaries are equal; otherwise, returns
    (False, [... list of issues here])
    """
    if d1_name is None:
        d1_name = "d1"
    if d2_name is None:
        d2_name = "d2"
    are_equal = True
    issues = []
    d1_keys = sorted([k for k in d1.keys()])
    d2_keys = sorted([k for k in d2.keys()])
    if len(d1_keys) != len(d2_keys):
        are_equal = False
        issues.append(
            f"Expected the number of keys in {d1_name} to equal that of {d2_name} " +
            f"but got {len(d1_keys)} and {len(d2_keys)} respectively\n" +
            f"{d1_name} keys: {str(d1_keys)}\n" +
            f"{d2_name} keys: {str(d2_keys)}")
    if (are_equal):
        for k in d1_keys:
            if len(d1[k]) != len(d2[k]):
                are_equal = False
                issues.append(
                    f"Expected len({d1_name}[\"{k}\"]) == len({d2_name}[\"{k}\"]) " +
                    f"but got {len(d1[k])} and {len(d2[k])}, respectively")
                break
            if isinstance(d1[k], np.ndarray) and isinstance(d2[k], np.ndarray):
                if not (d1[k] == d2[k]).all():
                    are_equal = False
                    issues.append(f"np.ndarray {d1_name}['{k}'] != {d2_name}['{k}']")
                    break
            elif d1[k] != d2[k]:
                are_equal = False
                issues.append(f"{d1_name}['{k}'] != {d2_name}['{k}']")
                break
    return (are_equal, issues)

class TestCycle(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()
    
    def test_monotonicity(self):
        "checks that time is monotonically increasing"
        print(f"Running {type(self)}.test_monotonicity.")
        self.assertTrue((np.diff(cycle.Cycle.from_file('udds').time_s) > 0).all())

    def test_load_dict(self):
        "checks that conversion from dict works"
        print(f"Running {type(self)}.test_load_dict.")
        cyc =cycle.Cycle.from_file('udds')
        cyc_df = pd.read_csv(Path(cycle.__file__).parent / 'resources/cycles/udds.csv')
        cyc_dict = cyc_df.to_dict(orient='list')
        cyc_dict.update({'name': 'udds'})
        cyc_from_dict = cycle.Cycle.from_dict(cyc_dict)

        self.assertTrue((
            pd.DataFrame(cyc.get_cyc_dict()) ==
            pd.DataFrame(cyc_from_dict.get_cyc_dict())).all().all()
        )
    
    def test_that_udds_has_18_microtrips(self):
        "Check that the number of microtrips equals expected"
        cyc = cycle.Cycle.from_file("udds")
        microtrips = cycle.to_microtrips(cyc.get_cyc_dict())
        expected_number_of_microtrips = 18
        actual_number_of_microtrips = len(microtrips)
        self.assertTrue(
            expected_number_of_microtrips == actual_number_of_microtrips,
            f"Expected {expected_number_of_microtrips} microtrips in UDDS but got {actual_number_of_microtrips}")
    
    def test_roundtrip_of_microtrip_and_concat(self):
        "A cycle split into microtrips and concatenated back together should equal the original"
        cyc = cycle.Cycle.from_file("udds")
        cyc_dict = cyc.get_cyc_dict()
        microtrips = cycle.to_microtrips(cyc_dict)
        # NOTE: specifying the name for concat is required to get the same keys 
        reconstituted_cycle = cycle.concat(microtrips, name=cyc_dict["name"])
        (are_equal, issues) = dicts_are_equal(cyc_dict, reconstituted_cycle, "original_cycle", "reconstituted_cycle")
        self.assertTrue(are_equal, "; ".join(issues))

    def test_roundtrip_of_microtrip_and_concat_using_keep_name_arg(self):
        "A cycle split into microtrips and concatenated back together should equal the original"
        cyc = cycle.Cycle.from_file("udds")
        cyc_dict = cyc.get_cyc_dict()
        microtrips = cycle.to_microtrips(cyc_dict, keep_name=True)
        # NOTE: specifying the name for concat is required to get the same keys 
        reconstituted_cycle = cycle.concat(microtrips, name=cyc_dict["name"])
        (are_equal, issues) = dicts_are_equal(cyc_dict, reconstituted_cycle, "original_cycle", "reconstituted_cycle")
        self.assertTrue(are_equal, "; ".join(issues))

    def test_set_from_dict_for_a_microtrip(self):
        "Test splitting into microtrips and setting is as expected"
        cyc = cycle.Cycle.from_file("udds")
        cyc_dict = cyc.get_cyc_dict()
        microtrips = cycle.to_microtrips(cyc_dict, keep_name=True)
        cyc = cycle.Cycle.from_dict(microtrips[1])
        mt_dict = cyc.get_cyc_dict()
        (are_equal, issues) = dicts_are_equal(microtrips[1], mt_dict, "first_microtrip", "microtrip_via_set_from_dict")
        self.assertTrue(are_equal, "; ".join(issues))
    
    def test_duration_of_concatenated_cycles_is_the_sum_of_the_components(self):
        "Test that two cycles concatenated have the same duration as the sum of the constituents"
        cyc1 =cycle.Cycle.from_file('udds')
        cyc2 =cycle.Cycle.from_file('us06')
        cyc_concat12 = cycle.concat([cyc1.get_cyc_dict(), cyc2.get_cyc_dict()])
        cyc_concat21 = cycle.concat([cyc2.get_cyc_dict(), cyc1.get_cyc_dict()])
        cyc12 = cycle.Cycle.from_dict(cyc_dict=cyc_concat12)
        cyc21 = cycle.Cycle.from_dict(cyc_dict=cyc_concat21)
        self.assertEqual(cyc_concat12["time_s"][-1], cyc_concat21["time_s"][-1])
        self.assertEqual(cyc1.time_s[-1] + cyc2.time_s[-1], cyc_concat21["time_s"][-1])
        self.assertEqual(cyc12.time_s[-1], cyc1.time_s[-1] + cyc2.time_s[-1])
        self.assertEqual(cyc21.time_s[-1], cyc1.time_s[-1] + cyc2.time_s[-1])
        self.assertEqual(len(cyc12.time_s), len(cyc1.time_s) + len(cyc2.time_s) - 1)
        self.assertEqual(len(cyc12.mps), len(cyc1.mps) + len(cyc2.mps) - 1)
        self.assertEqual(len(cyc12.grade), len(cyc1.grade) + len(cyc2.grade) - 1)
        self.assertEqual(len(cyc12.road_type), len(cyc1.road_type) + len(cyc2.road_type) - 1)
    
    def test_cycle_equality(self):
        "Test structural equality of driving cycles"
        udds = cycle.Cycle.from_file("udds")
        us06 = cycle.Cycle.from_file("us06")
        self.assertFalse(cycle.equals(udds.get_cyc_dict(), us06.get_cyc_dict()))
        udds_2 = cycle.Cycle.from_file("udds")
        self.assertTrue(cycle.equals(udds.get_cyc_dict(), udds_2.get_cyc_dict()))
        cyc2dict = udds_2.get_cyc_dict()
        cyc2dict['extra key'] = None
        self.assertFalse(cycle.equals(udds.get_cyc_dict(), cyc2dict))
    
    def test_that_cycle_resampling_works_as_expected(self):
        "Test resampling the values of a cycle"
        for cycle_name in ["udds", "us06", "hwfet", "longHaulDriveCycle"]:
            cyc = cycle.Cycle.from_file(cycle_name)
            cyc_at_dt0_1 = cycle.Cycle.from_dict(cycle.resample(cyc.get_cyc_dict(), new_dt=0.1))
            cyc_at_dt10 = cycle.Cycle.from_dict(cycle.resample(cyc.get_cyc_dict(), new_dt=10))
            msg = f"issue for {cycle_name}, {len(cyc.time_s)} points, duration {cyc.time_s[-1]}"
            expected_num_at_dt0_1 = 10 * len(cyc.time_s) - 9
            self.assertEqual(len(cyc_at_dt0_1.time_s), expected_num_at_dt0_1, msg)
            self.assertEqual(len(cyc_at_dt0_1.mps), expected_num_at_dt0_1, msg)
            self.assertEqual(len(cyc_at_dt0_1.grade), expected_num_at_dt0_1, msg)
            self.assertEqual(len(cyc_at_dt0_1.road_type), expected_num_at_dt0_1, msg)
            expected_num_at_dt10 = len(cyc.time_s) // 10 + (0 if len(cyc.time_s) % 10 == 0 else 1)
            self.assertEqual(len(cyc_at_dt10.time_s), expected_num_at_dt10, msg)
            self.assertEqual(len(cyc_at_dt10.mps), expected_num_at_dt10, msg)
            self.assertEqual(len(cyc_at_dt10.grade), expected_num_at_dt10, msg)
            self.assertEqual(len(cyc_at_dt10.road_type), expected_num_at_dt10, msg)
    
    def test_resampling_and_concatenating_cycles(self):
        "Test that concatenating cycles at different sampling rates works as expected"
        udds = cycle.Cycle.from_file("udds")
        udds_10Hz = cycle.Cycle.from_dict(
            cyc_dict=cycle.resample(udds.get_cyc_dict(), new_dt=0.1))
        us06 = cycle.Cycle.from_file("us06")
        combo_resampled = cycle.resample(
            cycle.concat([udds_10Hz.get_cyc_dict(), us06.get_cyc_dict()]),
            new_dt=1)
        combo = cycle.concat([udds.get_cyc_dict(), us06.get_cyc_dict()])
        self.assertTrue(cycle.equals(combo, combo_resampled))
    
    def test_resampling_with_hold_keys(self):
        "Test that 'hold_keys' works with resampling"
        trapz = cycle.make_cycle(
            [0.0, 10.0, 20.0, 30.0],
            [0.0, 40.0 / params.MPH_PER_MPS, 40.0 / params.MPH_PER_MPS, 0.0])
        trapz['auxInKw'] = [1.0, 1.0, 3.0, 3.0]
        trapz_at_1hz = cycle.resample(trapz, new_dt=1.0, hold_keys={'auxInKw'})
        self.assertTrue(len(trapz_at_1hz['auxInKw']) == len(trapz_at_1hz['time_s']),
            f"Expected length of auxInKw ({len(trapz_at_1hz['auxInKw'])}) " +
            f"to equal length of time_s ({len(trapz_at_1hz['time_s'])})"
        )
        self.assertEqual({1.0, 3.0}, {aux for aux in trapz_at_1hz['auxInKw']})

    def test_that_resampling_preserves_total_distance_traveled_using_rate_keys(self):
        "Distance traveled before and after resampling should be the same when rate_keys are used"
        for cycle_name in ['udds', 'us06', 'hwfet', 'longHaulDriveCycle']:
            the_cyc = cycle.Cycle.from_file(cycle_name).get_cyc_dict()
            # DOWNSAMPLING
            new_dt_s = 10.0
            cyc_at_0_1hz = cycle.resample(the_cyc, new_dt=new_dt_s)
            msg = (
                f"issue for {cycle_name} (downsampling)\n" + 
                f"cycle: {cycle_name}\n" +
                f"duration {the_cyc['time_s'][-1]} s\n" +
                f"duration {cyc_at_0_1hz['time_s'][-1]} s (0.1 Hz)\n"
            )
            dist_m = calc_distance_traveled_m(
                the_cyc, up_to=cyc_at_0_1hz['time_s'][-1])
            dist_at_0_1hz_m = calc_distance_traveled_m(
                cyc_at_0_1hz, up_to=cyc_at_0_1hz['time_s'][-1])
            self.assertTrue(abs(dist_m - dist_at_0_1hz_m) > 1, msg=msg)
            cyc_at_0_1hz_rate_keys = cycle.resample(
                the_cyc, new_dt=new_dt_s, rate_keys={'mps'})
            self.assertAlmostEqual(
                cyc_at_0_1hz['time_s'][-1],
                cyc_at_0_1hz_rate_keys['time_s'][-1], msg=msg)
            dist_at_0_1hz_w_rate_keys_m = calc_distance_traveled_m(
                cyc_at_0_1hz_rate_keys, up_to=cyc_at_0_1hz['time_s'][-1])
            self.assertAlmostEqual(dist_m, dist_at_0_1hz_w_rate_keys_m, 3, msg=msg)
            self.assertTrue((the_cyc['mps'] >= 0.0).all(), msg=msg)
            self.assertTrue((cyc_at_0_1hz['mps'] >= 0.0).all(), msg=msg)
            self.assertTrue((cyc_at_0_1hz_rate_keys['mps'] >= 0.0).all(), msg=msg)
            # UPSAMPLING
            new_dt_s = 0.1
            cyc_at_10hz = cycle.resample(the_cyc, new_dt=new_dt_s)
            msg = (
                f"issue for {cycle_name} (upsampling)\n" + 
                f"cycle: {cycle_name}\n" +
                f"duration {the_cyc['time_s'][-1]} s\n" +
                f"duration {cyc_at_10hz['time_s'][-1]} s (10 Hz)\n"
            )
            dist_m = calc_distance_traveled_m(
                the_cyc, up_to=cyc_at_10hz['time_s'][-1])
            dist_at_10hz_m = calc_distance_traveled_m(
                cyc_at_10hz, up_to=cyc_at_10hz['time_s'][-1])
            # NOTE: upsampling doesn't cause a distance discrepancy
            self.assertAlmostEqual(dist_m, dist_at_10hz_m, msg=msg)
            cyc_at_10hz_rate_keys = cycle.resample(the_cyc, new_dt=new_dt_s, rate_keys={'mps'})
            dist_at_10hz_w_rate_keys_m = calc_distance_traveled_m(
                cyc_at_10hz_rate_keys, up_to=cyc_at_10hz_rate_keys['time_s'][-1])
            # NOTE: specifying rate keys shouldn't change the distance calculation with rate keys
            self.assertAlmostEqual(dist_m, dist_at_10hz_w_rate_keys_m, 3, msg=msg)
            self.assertTrue((cyc_at_10hz['mps'] >= 0.0).all(), msg=msg)
            self.assertTrue((cyc_at_10hz_rate_keys['mps'] >= 0.0).all(), msg=msg)
    
    def test_clip_by_times(self):
        "Test that clipping by times works as expected"
        udds = cycle.Cycle.from_file("udds").get_cyc_dict()
        udds_start = cycle.clip_by_times(udds, t_end=300)
        udds_end = cycle.clip_by_times(udds, t_end=udds["time_s"][-1], t_start=300)
        self.assertTrue(udds_start['time_s'][-1] == 300.0)
        self.assertTrue(udds_start['time_s'][0] == 0.0)
        self.assertTrue(udds_end['time_s'][-1] == udds["time_s"][-1] - 300.0)
        self.assertTrue(udds_end['time_s'][0] == 0.0)
        udds_reconstruct = cycle.concat([udds_start, udds_end], name=udds["name"])
        self.assertTrue(cycle.equals(udds, udds_reconstruct))

    def test_get_accelerations(self):
        "Test getting and processing accelerations"
        tri = {
            'name': "triangular speed trace",
            'time_s': np.array([0.0, 10.0, 20.0]),
            'mps': np.array([0.0, 5.0, 0.0]),
            'cycGrade': np.array([0.0, 0.0, 0.0]),
            'road_type': np.array([0.0, 0.0, 0.0]),
        }
        expected_tri = np.array([0.5, -0.5]) # acceleration (m/s2)
        trapz = {
            'name': "trapezoidal speed trace",
            'time_s': np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            'mps': np.array([0.0, 5.0, 5.0, 0.0, 0.0]),
            'cycGrade': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            'road_type': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        expected_trapz = np.array([0.5, 0.0, -0.5, 0.0]) # acceleration (m/s2)
        for (cyc, expected_accels_m__s2) in [(tri, expected_tri), (trapz, expected_trapz)]:
            actual_accels_m__s2 = cycle.accelerations(cyc)
            self.assertEqual(len(actual_accels_m__s2), len(expected_accels_m__s2))
            for idx, (e, a) in enumerate(zip(expected_accels_m__s2, actual_accels_m__s2)):
                self.assertAlmostEqual(e, a, f"{cyc['name']} {idx}")
            exp_max_accel_m__s2 = np.max(expected_accels_m__s2)
            act_max_accel_m__s2 = cycle.peak_acceleration(cyc)
            self.assertAlmostEqual(exp_max_accel_m__s2, act_max_accel_m__s2)
            exp_min_accel_m__s2 = np.min(expected_accels_m__s2)
            act_min_accel_m__s2 = cycle.peak_deceleration(cyc)
            self.assertAlmostEqual(exp_min_accel_m__s2, act_min_accel_m__s2)

    def test_that_copy_creates_idential_structures(self):
        "Checks that copy methods produce identical cycles"
        udds = cycle.Cycle.from_file('udds')
        another_udds = cycle.copy_cycle(udds)
        self.assertTrue(udds.get_cyc_dict(), another_udds.get_cyc_dict())
    
    def test_make_cycle(self):
        "Check that make_cycle works as expected"
        # TODO: should make_cycle automatically place time on a consistent basis?
        #   That is, if you feed it [1,2,3] for seconds, should you get [0, 1, 2]?
        #   This could be an optional parameter passed in...
        cyc = cycle.make_cycle([0, 1, 2], [0, 1, 0])
        self.assertEqual(cyc['time_s'][0], 0.0)
        self.assertEqual(cyc['time_s'][-1], 2.0)
        expected_keys = {'time_s', 'mps', 'grade', 'road_type'} 
        self.assertEqual(expected_keys, {k for k in cyc.keys()})
        for k in expected_keys:
            self.assertEqual(len(cyc[k]), 3)

    def test_key_conversion(self):
        "check that legacy keys can still be generated"
        old_keys = list(cycle.NEW_TO_OLD.values())
        cyc = cycle.Cycle.from_file('udds')
        old_cyc = cycle.LegacyCycle(cyc)
        self.assertEqual(old_keys, inspect_utils.get_attrs(old_cyc))
    
    def test_get_grade_by_distance(self):
        "check that we can lookup grade by distance"
        expected_distances_m = [  0.0 ,  50.0 , 1050.0 ,  2050.0, 2100.0 ]
        cyc = cycle.Cycle.from_dict(
            cycle.make_cycle(
                ts=[  0.0 ,  10.0 , 110.0 , 210.0 , 220.0 ],
                vs=[  0.0 ,  10.0 ,  10.0 ,  10.0 ,   0.0 ],
                gs=[  0.01,   0.01,   0.02,   0.02,   0.02],
            ))
        ds = cycle.trapz_step_distances(cyc).cumsum()
        self.assertEqual(len(expected_distances_m), len(ds))
        for idx in range(len(expected_distances_m)):
            self.assertAlmostEqual(expected_distances_m[idx], ds[idx])
        cyc0 = cycle.Cycle.from_dict(
            cycle.resample(
                cycle.make_cycle(
                    ts=[  0.0 ,  10.0 , 110.0 , 210.0 , 220.0 ],
                    vs=[  0.0 ,  10.0 ,  10.0 ,  10.0 ,   0.0 ],
                    gs=[  0.01,   0.01,   0.01,   0.02,   0.02],
                ),
                new_dt=1.0,
                hold_keys_next={'grade'},
            )
        )
        # TODO: rewrite test to be based on sample points; i = 1, dist_m (traveled from start) = 0.5, step_dist_m = 0.5, etc.
        test_conditions = [
            {'step': 0,   'dist_m': 0.0, 'expected_dist_start_m': 0.0,    'expected_dist_step_m': 0.0,  'expected_average_grade': 0.010},
            {'step': 10,  'dist_m': 50.0, 'expected_dist_start_m': 45.0,   'expected_dist_step_m': 10.0, 'expected_average_grade': 0.010},
            {'step': 109, 'dist_m': 1040.0, 'expected_dist_start_m': 1035.0, 'expected_dist_step_m': 10.0, 'expected_average_grade': 0.010},
            {'step': 110, 'dist_m': 1050.0, 'expected_dist_start_m': 1045.0, 'expected_dist_step_m': 10.0, 'expected_average_grade': 0.010},
            {'step': 111, 'dist_m': 1060.0, 'expected_dist_start_m': 1055.0, 'expected_dist_step_m': 10.0, 'expected_average_grade': 0.020},
            {'step': 220, 'dist_m': 2100.0, 'expected_dist_start_m': 2100.0, 'expected_dist_step_m': 0.0,  'expected_average_grade': 0.020},
        ]
        cyc_rust = cyc0.to_rust()
        for cond in test_conditions:
            msg = f"Python: Failed for {cond}"
            dist_start_m = cycle.trapz_step_start_distance(cyc0, cond['step'])
            dist_step_m = cycle.trapz_distance_for_step(cyc0, cond['step'])
            self.assertAlmostEqual(cond['dist_m'], dist_start_m + dist_step_m, msg=msg)
            avg_grade = cyc0.average_grade_over_range(dist_start_m, dist_step_m)
            self.assertAlmostEqual(
                cond['expected_average_grade'], avg_grade, places=5,
                msg=f"{msg}; {cond['expected_average_grade']} != {avg_grade}"
            )
            # RUST CHECK
            msg = f"RUST: Failed for {cond}"
            dist_start_m = cycle.trapz_step_start_distance(cyc_rust, cond['step'])
            dist_step_m = cycle.trapz_distance_for_step(cyc_rust, cond['step'])
            self.assertAlmostEqual(cond['dist_m'], dist_start_m + dist_step_m, msg=msg)
            avg_grade = cyc_rust.average_grade_over_range(dist_start_m, dist_step_m)
            self.assertAlmostEqual(
                cond['expected_average_grade'], avg_grade, places=5,
                msg=f"{msg}; {cond['expected_average_grade']} != {avg_grade}"
            )
        gr = cyc0.average_grade_over_range(1040.0, 20.0)
        expected_gr = 0.015
        self.assertAlmostEqual(expected_gr, gr, places=5)

        gr = cyc_rust.average_grade_over_range(1040.0, 20.0)
        expected_gr = 0.015
        self.assertAlmostEqual(expected_gr, gr, places=5)
    
    def test_dt_s_vs_dt_s_at_i(self):
        """
        Test that dt_s_at_i is a true replacement for dt_s[i]
        """
        cyc = cycle.Cycle.from_file('udds')
        ru_cyc = cyc.to_rust()
        dt_s = cyc.dt_s
        for i in range(len(cyc.time_s)):
            self.assertAlmostEqual(dt_s[i], cyc.dt_s_at_i(i))
            self.assertAlmostEqual(dt_s[i], ru_cyc.dt_s_at_i(i))
    
    def test_trapz_step_start_distance(self):
        """
        Test the implementation of trapz_step_start_distance
        """
        verbose = False
        cyc = cycle.Cycle.from_file('udds')
        num_samples = len(cyc.time_s)
        if verbose:
            start_t = time.time()
        ds_test = [cycle.trapz_step_start_distance(cyc, i) for i in range(num_samples)]
        if verbose:
            end_t = time.time()
            print(f"cycle.trapz_step_start_distance(...) took {end_t - start_t:6.3f} s")
        if verbose:
            start_t = time.time()
        ds_good = [cycle.trapz_step_distances(cyc)[:i].sum() for i in range(num_samples)]
        if verbose:
            end_t = time.time()
            print(f"cycle.trapz_step_distances(cyc)[:i].sum() took {end_t - start_t:6.3f} s")
        self.assertEqual(len(ds_test), len(ds_good))
        for (d_test, d_good) in zip(ds_test, ds_good):
            self.assertAlmostEqual(d_test, d_good)
    
    def test_that_cycle_cache_interp_grade_substitutes_for_average_grade_over_range(self):
        """
        Ensure that CycleCache.interp_grade actually predicts the same values as
        Cycle.average_grade_over_range(d, 0.0, cache=None|CycleCache) with and without
        using CycleCache
        """
        cycles = [
            cycle.Cycle.from_dict({
                'time_s': [0.0, 1.0, 2.0, 3.0, 4.0],
                'mps': [0.0, 1.0, 1.0, 1.0, 0.0],
                'grade': [1.0, 1.0, 1.0, -1.0, -1.0],
                'road_type': [0, 0, 0, 0, 0],
                'name': "triangle hill",
            }),
            cycle.Cycle.from_file('udds'),
            cycle.Cycle.from_file('TSDC_tripno_42648_cycle'),
        ]
        for cyc in cycles:
            trapz_distances = cycle.trapz_step_distances(cyc).cumsum()
            self.assertEqual(len(trapz_distances), len(cyc.time_s))
            cache = cyc.build_cache()
            max_idx = len(cyc.grade) - 1
            def make_msg(idx, dd=0.0):
                name = cyc.name
                i0 = max(idx-1,0)
                i1 = min(idx+1,max_idx)
                d = trapz_distances[idx]
                d0 = trapz_distances[i0]
                d1 = trapz_distances[i1]
                v = cyc.mps[idx]
                v0 = cyc.mps[i0]
                v1 = cyc.mps[i1]
                g =  cyc.grade[idx]
                g0 = cyc.grade[i0]
                g1 = cyc.grade[i1]
                return (
                    f"issue at index {idx} for {name} (dd={dd}) looking up {d + dd}:\n"
                    + f"d[{i0}]={d0}; d[{idx}]={d}; d[{i1}]={d1}\n"
                    + f"g[{i0}]={g0}; g[{idx}]={g}; g[{i1}]={g1}\n"
                    + f"v[{i0}]={v0}; v[{idx}]={v}; v[{i1}]={v1}\n"
                )

            for idx, d in enumerate(trapz_distances):
                g0 = cyc.average_grade_over_range(d, 0.0)
                g1 = cyc.average_grade_over_range(d, 0.0, cache=cache)
                g2 = cache.interp_grade(d)
                msg = make_msg(idx)
                self.assertEqual(g0, g1, msg=msg)
                self.assertEqual(g1, g2, msg=msg)
                if cyc.name == "triangle hill":
                    self.assertEqual(g2, 1.0 if d <= 2.0 else -1.0, msg=msg)
                dd = 0.1
                g3 = cyc.average_grade_over_range(d + dd, 0.0)
                g4 = cyc.average_grade_over_range(d + dd, 0.0, cache=cache)
                g5 = cache.interp_grade(d + dd)
                msg = make_msg(idx, dd)
                self.assertEqual(g3, g4, msg=msg)
                self.assertEqual(g4, g5, msg=msg)
                dd = -0.1
                g6 = cyc.average_grade_over_range(d + dd, 0.0)
                g7 = cyc.average_grade_over_range(d + dd, 0.0, cache=cache)
                g8 = cache.interp_grade(d + dd)
                msg = make_msg(idx, dd)
                self.assertEqual(g6, g7, msg=msg)
                self.assertEqual(g7, g8, msg=msg)
    
    def test_that_trapz_step_start_distance_equals_cache_trapz_distances(self):
        """
        Test that cycle.trapz_step_start_distance(self.cyc0, i) == self._cyc0_cache.trapz_distances_m[i-1]
        """
        cycles = [
            cycle.Cycle.from_dict({
                'time_s': [0.0, 1.0, 2.0, 3.0, 4.0],
                'mps': [0.0, 1.0, 1.0, 1.0, 0.0],
                'grade': [1.0, 1.0, 1.0, -1.0, -1.0],
                'road_type': [0, 0, 0, 0, 0],
                'name': "triangle hill",
            }),
            cycle.Cycle.from_file('udds'),
            cycle.Cycle.from_file('TSDC_tripno_42648_cycle'),
            cycle.Cycle.from_file('us06'),
        ]
        for cyc in cycles:
            cache = cyc.build_cache()
            for i in range(len(cyc.time_s)):
                d0 = cycle.trapz_step_start_distance(cyc, i)
                d1 = cache.trapz_distances_m[max(i-1, 0)]
                self.assertAlmostEqual(d0, d1)
    
    def test_average_grade_over_range_with_and_without_cache(self):
        """
        Ensure that CycleCache usage only speeds things up; doesn't change values...
        """
        dist_deltas_m = [1.0, 10.0, 100.0]
        cycles = [
            cycle.Cycle.from_dict({
                'time_s': [0.0, 1.0, 2.0, 3.0, 4.0],
                'mps': [0.0, 1.0, 1.0, 1.0, 0.0],
                'grade': [1.0, 1.0, 1.0, -1.0, -1.0],
                'road_type': [0, 0, 0, 0, 0],
                'name': "triangle hill",
            }),
            cycle.Cycle.from_file('TSDC_tripno_42648_cycle'),
        ]
        for cyc in cycles:
            cache = cyc.build_cache()
            for dd in dist_deltas_m:
                for i in range(len(cyc.time_s)):
                    d = cycle.trapz_step_start_distance(cyc, i)
                    g0 = cyc.average_grade_over_range(d, dd)
                    g1 = cyc.average_grade_over_range(d, dd, cache=cache)
                    msg = f"issue for {cyc.name} for i={i} and dd={dd} and d={d}"
                    self.assertAlmostEqual(g0, g1, msg=msg)


if __name__ == '__main__':
    unittest.main()
