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

    # TODO: implement this
    # def test_copy(self):
    #     """checks that copy methods produce identical cycles"""

    # TODO: port examples from ../docs/demo.py that use cycle functions to here
