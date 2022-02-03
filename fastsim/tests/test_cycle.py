"""Test suite for cycle instantiation and manipulation."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import cycle

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
        original_keys = sorted([k for k in cyc_dict.keys()])
        reconstituted_keys = sorted([k for k in reconstituted_cycle.keys()])
        self.assertEqual(
            len(original_keys), len(reconstituted_keys),
            f"Expected {len(original_keys)} but got {len(reconstituted_keys)}\n" +
            f"original_keys: {str(original_keys)}\nreconstituted_keys: {str(reconstituted_keys)}")
        for key in original_keys:
            self.assertEqual(
                len(cyc_dict[key]), len(reconstituted_cycle[key]),
                f"Value lengths not equal for key {key}")
            if key == "name":
                self.assertEqual(
                    cyc_dict[key], reconstituted_cycle[key],
                    f"Values not equal for key {key}")
            else:
                self.assertTrue(
                    (cyc_dict[key] == reconstituted_cycle[key]).all(),
                    f"Values not equal for key {key}")

    # TODO: implement this
    # def test_copy(self):
    #     """checks that copy methods produce identical cycles"""

    # TODO: port examples from ../docs/demo.py that use cycle functions to here
