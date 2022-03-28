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
        veh = vehicle.Vehicle.from_vehdb(1, verbose=False)
        sim_drive = simdrive.SimDrive(cyc, veh)

        for x in range(100):
            sim_drive.sim_drive_step()
        
        self.assertEqual(sim_drive.fs_kw_out_ach[100], 29.32533101828119)
    
    def test_sim_drive_walk(self):
        """Verify that sim_drive_walk produces an expected result."""
        
        print(f"Running {type(self)}.test_sim_drive_walk.")
        cyc =cycle.Cycle.from_file('udds')
        veh = vehicle.Vehicle.from_vehdb(1, verbose=False)
        sim_drive = simdrive.SimDrive(cyc, veh)
        sim_drive.sim_drive_walk(init_soc=1)

        self.assertAlmostEqual(sim_drive.fs_kw_out_ach.sum(), 24410.31348426869, places=4)

    def test_split_cycles(self):

        print(f"Running {type(self)}.test_split_cycles.")
        t_clip = 210 # speed is non-zero here
        cyc1 = cycle.Cycle.from_dict(
            cyc_dict=(cycle.clip_by_times(cycle.Cycle.from_file('udds').get_cyc_dict(), t_end=t_clip))
            )
        t_end =cycle.Cycle.from_file('udds').time_s[-1]
        cyc2 = cycle.Cycle.from_dict(
            cyc_dict=(cycle.clip_by_times(cycle.Cycle.from_file('udds').get_cyc_dict(), 
            t_start=t_clip, t_end=t_end))
            )

        veh = vehicle.Vehicle.from_vehdb(1, verbose=False)

        sd1 = simdrive.SimDrive(cyc1, veh)
        sd1.sim_drive()

        sd2 = simdrive.SimDrive(cyc2, veh)
        # this is a bug workaround.  There will be a fix that will make this workaround remain functional but unnecessary
        sd2.mps_ach[0] = cyc2.mph[0]
        sd2.mph_ach[0] = cyc2.mps[0]
        sd2.sim_drive()

        cyc =cycle.Cycle.from_file('udds')
        sdtot = simdrive.SimDrive(cyc, veh)
        sdtot.sim_drive()

        self.assertAlmostEqual(
            sd1.fs_kw_out_ach.sum() + sd2.fs_kw_out_ach.sum(), 
            sdtot.fs_kw_out_ach.sum(),
            places=5)

    def test_time_dilation(self):
        veh = vehicle.Vehicle.from_vehdb(1)
        cyc = cycle.Cycle.from_dict(cyc_dict={
            'time_s': np.arange(10),
            'mps': np.append(2, np.ones(9) * 6),
        })
        sd = simdrive.SimDrive(cyc, veh)
        sd.sim_params.missed_trace_correction = True
        sd.sim_params.max_time_dilation = 0.05 # maximum upper margin for time dilation
        sd.sim_drive()

        trace_miss_corrected = (
            abs(sd.dist_m.sum() - sd.cyc0.dist_m.sum()) / sd.cyc0.dist_m.sum()) < sd.sim_params.time_dilation_tol

        self.assertTrue(trace_miss_corrected)

    def test_stop_start(self):
        cyc =cycle.Cycle.from_file('udds').get_cyc_dict()
        cyc = cycle.Cycle.from_dict(cyc_dict=cycle.clip_by_times(cyc, 130))

        veh = vehicle.Vehicle.from_vehdb(1)
        veh.stop_start = True
        veh.max_motor_kw = 1
        veh.max_ess_kw = 5
        veh.max_ess_kwh = 1
        veh.__post_init__()

        sd = simdrive.SimDrive(cyc, veh)
        sd.sim_drive()

        self.assertTrue(sd.fc_kw_in_ach[10] == 0)
        self.assertTrue(sd.fc_kw_in_ach[37] == 0)
    
    def test_achieved_speed_never_negative(self):
        for vehid in range(1, 27):
            veh = vehicle.Vehicle.from_vehdb(vehid)
            cyc = cycle.Cycle.from_file('udds')
            sd = simdrive.SimDrive(cyc, veh)
            sd.sim_drive_walk(0.1)
            sd.set_post_scalars()
            self.assertFalse(
                (sd.mps_ach < 0.0).any(),
                msg=f'Achieved speed contains negative values for vehicle {vehid}'
            )
            self.assertFalse(
                (sd.mps_ach > sd.cyc0.mps).any(),
                msg=f'Achieved speed is greater than requested speed for {vehid}'
            )
