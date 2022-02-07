"""Test suite for cycle instantiation and manipulation."""

from tkinter import N
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import cycle, params


def calc_distance_traveled_m(cyc, up_to=None):
    """
    Calculate the distance traveled in meters
    - cyc: a cycle dictionary
    - up_to: None or a positive number indicating a time in seconds. Will calculate the distance up-to that given time
    RETURN: Number, the distance traveled in meters
    """
    if up_to is None:
        return (np.diff(cyc['cycSecs']) * ((cyc['cycMps'][1:] + cyc['cycMps'][:-1])*0.5)).sum()
    dist = 0.0
    ts = cyc['cycSecs']
    avg_speeds = (cyc['cycMps'][1:] + cyc['cycMps'][:-1]) * 0.5
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
    def test_monotonicity(self):
        "checks that time is monotonically increasing"
        print(f"Running {type(self)}.test_monotonicity.")
        self.assertTrue((np.diff(cycle.Cycle('udds').cycSecs) > 0).all())

    def test_load_dict(self):
        "checks that conversion from dict works"
        print(f"Running {type(self)}.test_load_dict.")
        cyc = cycle.Cycle('udds')
        cyc_df = pd.read_csv(Path(cycle.__file__).parent / 'resources/cycles/udds.csv')
        cyc_dict = cyc_df.to_dict(orient='list')
        cyc_dict.update({'name': 'udds'})
        cyc_from_dict = cycle.Cycle(cyc_dict=cyc_dict)

        self.assertTrue((
            pd.DataFrame(cyc.get_cyc_dict()) ==
            pd.DataFrame(cyc_from_dict.get_cyc_dict())).all().all()
        )
    
    def test_that_udds_has_18_microtrips(self):
        "Check that the number of microtrips equals expected"
        cyc = cycle.Cycle("udds")
        microtrips = cycle.to_microtrips(cyc.get_cyc_dict())
        expected_number_of_microtrips = 18
        actual_number_of_microtrips = len(microtrips)
        self.assertTrue(
            expected_number_of_microtrips == actual_number_of_microtrips,
            f"Expected {expected_number_of_microtrips} microtrips in UDDS but got {actual_number_of_microtrips}")
    
    def test_roundtrip_of_microtrip_and_concat(self):
        "A cycle split into microtrips and concatenated back together should equal the original"
        cyc = cycle.Cycle("udds")
        cyc_dict = cyc.get_cyc_dict()
        microtrips = cycle.to_microtrips(cyc_dict)
        # NOTE: specifying the name for concat is required to get the same keys 
        reconstituted_cycle = cycle.concat(microtrips, name=cyc_dict["name"])
        (are_equal, issues) = dicts_are_equal(cyc_dict, reconstituted_cycle, "original_cycle", "reconstituted_cycle")
        self.assertTrue(are_equal, "; ".join(issues))

    def test_roundtrip_of_microtrip_and_concat_using_keep_name_arg(self):
        "A cycle split into microtrips and concatenated back together should equal the original"
        cyc = cycle.Cycle("udds")
        cyc_dict = cyc.get_cyc_dict()
        microtrips = cycle.to_microtrips(cyc_dict, keep_name=True)
        # NOTE: specifying the name for concat is required to get the same keys 
        reconstituted_cycle = cycle.concat(microtrips, name=cyc_dict["name"])
        (are_equal, issues) = dicts_are_equal(cyc_dict, reconstituted_cycle, "original_cycle", "reconstituted_cycle")
        self.assertTrue(are_equal, "; ".join(issues))

    def test_set_from_dict_for_a_microtrip(self):
        "Test splitting into microtrips and setting is as expected"
        cyc = cycle.Cycle("udds")
        cyc_dict = cyc.get_cyc_dict()
        microtrips = cycle.to_microtrips(cyc_dict, keep_name=True)
        cyc.set_from_dict(microtrips[1])
        mt_dict = cyc.get_cyc_dict()
        (are_equal, issues) = dicts_are_equal(microtrips[1], mt_dict, "first_microtrip", "microtrip_via_set_from_dict")
        self.assertTrue(are_equal, "; ".join(issues))
    
    def test_duration_of_concatenated_cycles_is_the_sum_of_the_components(self):
        "Test that two cycles concatenated have the same duration as the sum of the constituents"
        cyc1 = cycle.Cycle('udds')
        cyc2 = cycle.Cycle('us06')
        cyc_concat12 = cycle.concat([cyc1.get_cyc_dict(), cyc2.get_cyc_dict()])
        cyc_concat21 = cycle.concat([cyc2.get_cyc_dict(), cyc1.get_cyc_dict()])
        cyc12 = cycle.Cycle(cyc_dict=cyc_concat12)
        cyc21 = cycle.Cycle(cyc_dict=cyc_concat21)
        self.assertEqual(cyc_concat12["cycSecs"][-1], cyc_concat21["cycSecs"][-1])
        self.assertEqual(cyc1.cycSecs[-1] + cyc2.cycSecs[-1], cyc_concat21["cycSecs"][-1])
        self.assertEqual(cyc12.cycSecs[-1], cyc1.cycSecs[-1] + cyc2.cycSecs[-1])
        self.assertEqual(cyc21.cycSecs[-1], cyc1.cycSecs[-1] + cyc2.cycSecs[-1])
        self.assertEqual(len(cyc12.cycSecs), len(cyc1.cycSecs) + len(cyc2.cycSecs) - 1)
        self.assertEqual(len(cyc12.cycMps), len(cyc1.cycMps) + len(cyc2.cycMps) - 1)
        self.assertEqual(len(cyc12.cycGrade), len(cyc1.cycGrade) + len(cyc2.cycGrade) - 1)
        self.assertEqual(len(cyc12.cycRoadType), len(cyc1.cycRoadType) + len(cyc2.cycRoadType) - 1)
    
    def test_cycle_equality(self):
        "Test structural equality of driving cycles"
        udds = cycle.Cycle("udds")
        us06 = cycle.Cycle("us06")
        self.assertFalse(cycle.equals(udds.get_cyc_dict(), us06.get_cyc_dict(), verbose=False))
        udds_2 = cycle.Cycle("udds")
        self.assertTrue(cycle.equals(udds.get_cyc_dict(), udds_2.get_cyc_dict(), verbose=False))
        cyc2dict = udds_2.get_cyc_dict()
        cyc2dict['extra key'] = None
        self.assertFalse(cycle.equals(udds.get_cyc_dict(), cyc2dict, verbose=False))
    
    def test_that_cycle_resampling_works_as_expected(self):
        "Test resampling the values of a cycle"
        for cycle_name in ["udds", "us06", "hwfet", "longHaulDriveCycle"]:
            cyc = cycle.Cycle(cycle_name)
            cyc_at_dt0_1 = cycle.Cycle(cyc_dict=cycle.resample(cyc.get_cyc_dict(), new_dt=0.1))
            cyc_at_dt10 = cycle.Cycle(cyc_dict=cycle.resample(cyc.get_cyc_dict(), new_dt=10))
            msg = f"issue for {cycle_name}, {len(cyc.cycSecs)} points, duration {cyc.cycSecs[-1]}"
            expected_num_at_dt0_1 = 10 * len(cyc.cycSecs) - 9
            self.assertEqual(len(cyc_at_dt0_1.cycSecs), expected_num_at_dt0_1, msg)
            self.assertEqual(len(cyc_at_dt0_1.cycMps), expected_num_at_dt0_1, msg)
            self.assertEqual(len(cyc_at_dt0_1.cycGrade), expected_num_at_dt0_1, msg)
            self.assertEqual(len(cyc_at_dt0_1.cycRoadType), expected_num_at_dt0_1, msg)
            expected_num_at_dt10 = len(cyc.cycSecs) // 10 + (0 if len(cyc.cycSecs) % 10 == 0 else 1)
            self.assertEqual(len(cyc_at_dt10.cycSecs), expected_num_at_dt10, msg)
            self.assertEqual(len(cyc_at_dt10.cycMps), expected_num_at_dt10, msg)
            self.assertEqual(len(cyc_at_dt10.cycGrade), expected_num_at_dt10, msg)
            self.assertEqual(len(cyc_at_dt10.cycRoadType), expected_num_at_dt10, msg)
    
    def test_resampling_and_concatenating_cycles(self):
        "Test that concatenating cycles at different sampling rates works as expected"
        udds = cycle.Cycle("udds")
        udds_10Hz = cycle.Cycle(
            cyc_dict=cycle.resample(udds.get_cyc_dict(), new_dt=0.1))
        us06 = cycle.Cycle("us06")
        combo_resampled = cycle.resample(
            cycle.concat([udds_10Hz.get_cyc_dict(), us06.get_cyc_dict()]),
            new_dt=1)
        combo = cycle.concat([udds.get_cyc_dict(), us06.get_cyc_dict()])
        self.assertTrue(cycle.equals(combo, combo_resampled, verbose=False))
    
    def test_resampling_with_hold_keys(self):
        "Test that 'hold_keys' works with resampling"
        trapz = cycle.make_cycle(
            [0.0, 10.0, 20.0, 30.0],
            [0.0, 40.0 / params.mphPerMps, 40.0 / params.mphPerMps, 0.0])
        trapz['auxInKw'] = [1.0, 1.0, 3.0, 3.0]
        trapz_at_1hz = cycle.resample(trapz, new_dt=1.0, hold_keys={'auxInKw'})
        self.assertTrue(len(trapz_at_1hz['auxInKw']) == len(trapz_at_1hz['cycSecs']),
            f"Expected length of auxInKw ({len(trapz_at_1hz['auxInKw'])}) " +
            f"to equal length of cycSecs ({len(trapz_at_1hz['cycSecs'])})"
        )
        self.assertEqual({1.0, 3.0}, {aux for aux in trapz_at_1hz['auxInKw']})

    def test_that_resampling_preservers_total_distance_traveled_using_rate_keys(self):
        """Distance traveled before and after resampling should be the same when rate_keys are used
        TODO:
        - assert that distance is not the same without specifying rate variables
        - assert that with rate_variables specified, you get the same distance
        NOTE:
        - cycMps will be in a derived cycle; only do on cycMps
        """
        udds = cycle.Cycle('udds').get_cyc_dict()
        # Note: UDDS is 1369 seconds, sampling at 10Hz gives us a cycle to 1360 seconds
        # Thus, when checking distance traveled, we need to compare up to 1360 seconds.
        # DOWNSAMPLING
        new_dt_s = 10.0
        udds_at_0_1hz = cycle.resample(udds, new_dt=new_dt_s)
        dist_m = calc_distance_traveled_m(
            udds, up_to=udds_at_0_1hz['cycSecs'][-1])
        dist_at_0_1hz_m = calc_distance_traveled_m(
            udds_at_0_1hz, up_to=udds_at_0_1hz['cycSecs'][-1])
        self.assertTrue(abs(dist_m - dist_at_0_1hz_m) > 1)
        udds_at_0_1hz_rate_keys = cycle.resample(
            udds, new_dt=new_dt_s, rate_keys={'cycMps'})
        self.assertAlmostEqual(
            udds_at_0_1hz['cycSecs'][-1],
            udds_at_0_1hz_rate_keys['cycSecs'][-1])
        dist_at_0_1hz_w_rate_keys_m = calc_distance_traveled_m(
            udds_at_0_1hz_rate_keys, up_to=udds_at_0_1hz['cycSecs'][-1])
        self.assertAlmostEqual(dist_m, dist_at_0_1hz_w_rate_keys_m, 3)
        # UPSAMPLING
        new_dt_s = 0.1
        udds_at_10hz = cycle.resample(udds, new_dt=new_dt_s)
        dist_at_10hz_m = calc_distance_traveled_m(
            udds_at_10hz, up_to=udds_at_10hz['cycSecs'][-1])
        self.assertTrue(abs(dist_m - dist_at_10hz_m) > 1)
        udds_at_10hz_rate_keys = cycle.resample(udds, new_dt=new_dt_s, rate_keys={'cycMps'})
        dist_at_10hz_w_rate_keys_m = calc_distance_traveled_m(
            udds_at_10hz_rate_keys, up_to=udds_at_10hz_rate_keys['cycSecs'][-1])
        #self.assertAlmostEqual(dist_m, dist_at_10hz_w_rate_keys_m, 3)
    
    def test_clip_by_times(self):
        "Test that clipping by times works as expected"
        udds = cycle.Cycle("udds").get_cyc_dict()
        udds_start = cycle.clip_by_times(udds, t_end=300)
        udds_end = cycle.clip_by_times(udds, t_end=udds["cycSecs"][-1], t_start=300)
        self.assertTrue(udds_start['cycSecs'][-1] == 300.0)
        self.assertTrue(udds_start['cycSecs'][0] == 0.0)
        self.assertTrue(udds_end['cycSecs'][-1] == udds["cycSecs"][-1] - 300.0)
        self.assertTrue(udds_end['cycSecs'][0] == 0.0)
        udds_reconstruct = cycle.concat([udds_start, udds_end], name=udds["name"])
        self.assertTrue(cycle.equals(udds, udds_reconstruct, verbose=False))

    def test_get_accelerations(self):
        "Test getting and processing accelerations"
        tri = {
            'name': "triangular speed trace",
            'cycSecs': np.array([0.0, 10.0, 20.0]),
            'cycMps': np.array([0.0, 5.0, 0.0]),
            'cycGrade': np.array([0.0, 0.0, 0.0]),
            'cycRoadType': np.array([0.0, 0.0, 0.0]),
        }
        expected_tri = np.array([0.5, -0.5]) # acceleration (m/s2)
        trapz = {
            'name': "trapezoidal speed trace",
            'cycSecs': np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            'cycMps': np.array([0.0, 5.0, 5.0, 0.0, 0.0]),
            'cycGrade': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            'cycRoadType': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
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
        udds = cycle.Cycle('udds')
        another_udds = udds.copy()
        self.assertTrue(udds.get_cyc_dict(), another_udds.get_cyc_dict())
    
    def test_make_cycle(self):
        "Check that make_cycle works as expected"
        # TODO: should make_cycle automatically place time on a consistent basis?
        #   That is, if you feed it [1,2,3] for seconds, should you get [0, 1, 2]?
        #   This could be an optional parameter passed in...
        cyc = cycle.make_cycle([0, 1, 2], [0, 1, 0])
        self.assertEqual(cyc['cycSecs'][0], 0.0)
        self.assertEqual(cyc['cycSecs'][-1], 2.0)
        expected_keys = {'cycSecs', 'cycMps', 'cycGrade', 'cycRoadType'} 
        self.assertEqual(expected_keys, {k for k in cyc.keys()})
        for k in expected_keys:
            self.assertEqual(len(cyc[k]), 3)
