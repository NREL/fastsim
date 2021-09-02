"""Test suite for simdrive instantiation and usage."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import cycle, vehicle, simdrive


class TestSimDriveClassic(unittest.TestCase):
    """Tests for fastsim.simdrive.SimDriveClassic methods"""
    def test_sim_drive_step(self):
        "Verify that sim_drive_step produces an expected result."
        cyc = cycle.Cycle('udds')
        veh = vehicle.Vehicle(1)
        sim_drive = simdrive.SimDriveClassic(cyc, veh)

        for x in range(100):
            sim_drive.sim_drive_step()
        
        self.assertEqual(sim_drive.fsKwOutAch[100], 29.32533101828119)
    
    def test_sim_drive_walk(self):
        """Verify that sim_drive_walk produces an expected result."""
        cyc = cycle.Cycle('udds')
        veh = vehicle.Vehicle(1)
        sim_drive = simdrive.SimDriveClassic(cyc, veh)
        sim_drive.sim_drive_walk(initSoc=1)

        self.assertAlmostEqual(sim_drive.fsKwOutAch.sum(), 24410.31348426869, places=4)

    def test_split_cycles(self):
        t_clip = 210 # speed is non-zero here
        cyc1 = cycle.Cycle(
            cyc_dict=(cycle.clip_by_times(cycle.Cycle('udds').get_cyc_dict(), t_end=t_clip))
            )
        t_end = cycle.Cycle('udds').cycSecs[-1]
        cyc2 = cycle.Cycle(
            cyc_dict=(cycle.clip_by_times(cycle.Cycle('udds').get_cyc_dict(), 
            t_start=t_clip, t_end=t_end))
            )

        veh = vehicle.Vehicle(1)

        sd1 = simdrive.SimDriveClassic(cyc1, veh)
        sd1.sim_drive()

        sd2 = simdrive.SimDriveClassic(cyc2, veh)
        # this is a bug workaround.  There will be a fix that will make this workaround remain functional but unnecessary
        sd2.mpsAch[0] = cyc2.cycMph[0]
        sd2.mphAch[0] = cyc2.cycMps[0]
        sd2.sim_drive()

        cyc = cycle.Cycle('udds')
        sdtot = simdrive.SimDriveClassic(cyc, veh)
        sdtot.sim_drive()

        self.assertAlmostEqual(
            sd1.fsKwOutAch.sum() + sd2.fsKwOutAch.sum(), 
            sdtot.fsKwOutAch.sum(),
            places=5)