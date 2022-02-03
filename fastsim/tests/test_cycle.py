"""Test suite for cycle instantiation and manipulation."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import cycle


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


    # TODO: implement this
    # def test_copy(self):
    #     """checks that copy methods produce identical cycles"""

    # TODO: port examples from ../docs/demo.py that use cycle functions to here
