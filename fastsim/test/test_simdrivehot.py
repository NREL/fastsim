"""Test suite for simdrivehot instantiation and usage."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import cycle, vehicle, simdrivehot


def get_fc_temp_delta(use_jit=True):
    """Returns FC/engine temperature delta over UDDS cycle."""
    cyc = cycle.Cycle('udds')
    veh = vehicle.Vehicle(1)

    if not use_jit:
        sim_drive = simdrivehot.SimDriveHot(
            cyc, veh,
            teAmbDegC=np.ones(len(cyc.cycSecs)) * 22,
            teFcInitDegC=22)
    else:
        sim_drive = simdrivehot.SimDriveHotJit(
            cyc, veh,
            teAmbDegC=np.ones(len(cyc.cycSecs)) * 22,
            teFcInitDegC=22)

    sim_drive.sim_drive()
    delta = sim_drive.teFcDegC[-1] - sim_drive.teFcDegC[0]
    return delta


class TestSimDriveHot(unittest.TestCase):
    def test_fc_temperature(self):
        """Test to verify expected delta between starting temperature and final temperature."""
        # value for running above vehicle and cycle combination as of git commit
        # 27b99823dfd1
        reference_value = 65.68819480855775

        self.assertEqual(
            reference_value, 
            get_fc_temp_delta()
        )

if __name__ == '__main__':
    unittest.main()
