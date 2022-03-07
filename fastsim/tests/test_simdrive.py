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
        
        print(f"Running {type(self)}.test_sim_drive_step.")
        cyc =cycle.Cycle.from_file('udds')
        veh = vehicle.Vehicle(1, verbose=False)
        sim_drive = simdrive.SimDriveClassic(cyc, veh)

        for x in range(100):
            sim_drive.sim_drive_step()
        
        self.assertEqual(sim_drive.fsKwOutAch[100], 29.32533101828119)
    
    def test_sim_drive_walk(self):
        """Verify that sim_drive_walk produces an expected result."""
        
        print(f"Running {type(self)}.test_sim_drive_walk.")
        cyc =cycle.Cycle.from_file('udds')
        veh = vehicle.Vehicle(1, verbose=False)
        sim_drive = simdrive.SimDriveClassic(cyc, veh)
        sim_drive.sim_drive_walk(initSoc=1)

        self.assertAlmostEqual(sim_drive.fsKwOutAch.sum(), 24410.31348426869, places=4)

    def test_split_cycles(self):

        print(f"Running {type(self)}.test_split_cycles.")
        t_clip = 210 # speed is non-zero here
        cyc1 = cycle.Cycle(
            cyc_dict=(cycle.clip_by_times(cycle.Cycle('udds').get_cyc_dict(), t_end=t_clip))
            )
        t_end =cycle.Cycle.from_file('udds').cycSecs[-1]
        cyc2 = cycle.Cycle(
            cyc_dict=(cycle.clip_by_times(cycle.Cycle('udds').get_cyc_dict(), 
            t_start=t_clip, t_end=t_end))
            )

        veh = vehicle.Vehicle(1, verbose=False)

        sd1 = simdrive.SimDriveClassic(cyc1, veh)
        sd1.sim_drive()

        sd2 = simdrive.SimDriveClassic(cyc2, veh)
        # this is a bug workaround.  There will be a fix that will make this workaround remain functional but unnecessary
        sd2.mpsAch[0] = cyc2.cycMph[0]
        sd2.mphAch[0] = cyc2.mps[0]
        sd2.sim_drive()

        cyc =cycle.Cycle.from_file('udds')
        sdtot = simdrive.SimDriveClassic(cyc, veh)
        sdtot.sim_drive()

        self.assertAlmostEqual(
            sd1.fsKwOutAch.sum() + sd2.fsKwOutAch.sum(), 
            sdtot.fsKwOutAch.sum(),
            places=5)

    def test_time_dilation(self):
        veh = vehicle.Vehicle(1)
        cyc = cycle.Cycle(cyc_dict={
            'cycSecs': np.arange(10),
            'mps': np.append(2, np.ones(9) * 6),
        })
        sd = simdrive.SimDriveClassic(cyc, veh)
        sd.sim_params.missed_trace_correction = True
        sd.sim_params.max_time_dilation = 0.05 # maximum upper margin for time dilation
        sd.sim_drive()

        trace_miss_corrected = (
            abs(sd.distMeters.sum() - sd.cyc0.cycDistMeters.sum()) / sd.cyc0.cycDistMeters.sum()) < sd.sim_params.time_dilation_tol

        self.assertTrue(trace_miss_corrected)

    def test_stop_start(self):
        cyc =cycle.Cycle.from_file('udds').get_cyc_dict()
        cyc = cycle.Cycle(cyc_dict=cycle.clip_by_times(cyc, 130))

        veh = vehicle.Vehicle(1)
        veh.stopStart = True
        veh.maxMotorKw = 1
        veh.maxEssKw = 5
        veh.maxEssKwh = 1
        veh.set_init_calcs()

        sd = simdrive.SimDriveClassic(cyc, veh)
        sd.sim_drive()

        self.assertTrue(sd.fcKwInAch[10] == 0)
        self.assertTrue(sd.fcKwInAch[37] == 0)