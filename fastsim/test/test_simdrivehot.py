"""Test suite for simdrivehot instantiation and usage."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import importlib

from fastsim import cycle, vehicle, simdrivehot
importlib.reload(simdrivehot)


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
        cyc = cyc.get_numba_cyc()
        veh = veh.get_numba_veh()
        sim_drive = simdrivehot.SimDriveHotJit(
            cyc, veh,
            teAmbDegC=np.ones(len(cyc.cycSecs)) * 22,
            teFcInitDegC=22)

    sim_drive.sim_drive()
    delta = sim_drive.teFcDegC[-1] - sim_drive.teFcDegC[0]
    reference_delta = 65.675
    return delta - reference_delta         



class TestSimDriveHot(unittest.TestCase):
    def test_fc_temperature(self):
        """Test to verify expected delta between starting temperature and final temperature."""
        # value for running above vehicle and cycle combination as of git commit
        # 27b99823dfd1

        self.assertAlmostEqual( 
            get_fc_temp_delta(), 0.00,
            places=2
        )

class TestAirProperties(unittest.TestCase):
    def test_enthalpy(self):
        air = simdrivehot.AirProperties()
        h0 = 400e3
        T = air.get_T_from_h(h0)
        h = air.get_h(T)
        self.assertAlmostEqual(h, h0)

if __name__ == '__main__':
    print(get_fc_temp_delta())
    unittest.main()
