"""Test suite for simdrive instantiation and usage."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import fastsim as fsim
from fastsim import cycle, vehicle, simdrive
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable
from fastsim.auxiliaries import set_nested_values

USE_PYTHON = False
USE_RUST = True


if USE_RUST and not RUST_AVAILABLE:
    warn_rust_unavailable(__file__)


class TestSimDriveClassic(unittest.TestCase):
    """Tests for fastsim.simdrive.SimDriveClassic methods"""

    def setUp(self):
        fsim.utils.disable_logging()

    def test_sim_drive_step(self):
        "Verify that sim_drive_step produces an expected result."

        if USE_PYTHON:
            print(f"Running {type(self)}.test_sim_drive_step.")
            cyc = cycle.Cycle.from_file('udds')
            veh = vehicle.Vehicle.from_vehdb(1)
            sim_drive = simdrive.SimDrive(cyc, veh)
            sim_drive.init_for_step()

            for x in range(100):
                sim_drive.sim_drive_step()

            self.assertEqual(sim_drive.fs_kw_out_ach[100], 29.32533101828119)

        if RUST_AVAILABLE and USE_RUST:
            msg = f"Issue Running {type(self)}.test_sim_drive_step. Rust"
            cyc = cycle.Cycle.from_file('udds').to_rust()
            veh = vehicle.Vehicle.from_vehdb(1).to_rust()
            sim_drive = simdrive.RustSimDrive(cyc, veh)
            sim_drive.init_for_step(init_soc=veh.max_soc)

            for x in range(100):
                sim_drive.sim_drive_step()

            self.assertEqual(
                sim_drive.fs_kw_out_ach[100], 29.32533101828119, msg=msg)

    def test_sim_drive_walk(self):
        """Verify that sim_drive_walk produces an expected result."""

        if USE_PYTHON:
            print(f"Running {type(self)}.test_sim_drive_walk. Python")
            cyc = cycle.Cycle.from_file('udds')
            veh = vehicle.Vehicle.from_vehdb(1)
            sim_drive = simdrive.SimDrive(cyc, veh)
            sim_drive.sim_drive_walk(init_soc=1)

            self.assertAlmostEqual(
                sim_drive.fs_kw_out_ach.sum(), 24410.31348426869, places=4)

        if RUST_AVAILABLE and USE_RUST:
            msg = f"Issue Running {type(self)}.test_sim_drive_walk. Rust"
            cyc = cycle.Cycle.from_file('udds')
            veh = vehicle.Vehicle.from_vehdb(1)
            sim_drive = simdrive.SimDrive(cyc, veh)
            sim_drive.sim_drive_walk(init_soc=1)

            self.assertAlmostEqual(
                sim_drive.fs_kw_out_ach.sum(), 24410.31348426869, places=4, msg=msg)

    def test_split_cycles(self):
        if USE_PYTHON:
            print(f"Running {type(self)}.test_split_cycles.")
            t_clip = 210  # speed is non-zero here
            cyc1 = cycle.Cycle.from_dict(
                cyc_dict=(cycle.clip_by_times(cycle.Cycle.from_file(
                    'udds').get_cyc_dict(), t_end=t_clip))
            )
            t_end = cycle.Cycle.from_file('udds').time_s[-1]
            cyc2 = cycle.Cycle.from_dict(
                cyc_dict=(cycle.clip_by_times(cycle.Cycle.from_file('udds').get_cyc_dict(),
                                              t_start=t_clip, t_end=t_end))
            )

            veh = vehicle.Vehicle.from_vehdb(1)

            sd1 = simdrive.SimDrive(cyc1, veh)
            sd1.sim_drive()

            sd2 = simdrive.SimDrive(cyc2, veh)
            # this is a bug workaround.  There will be a fix that will make this workaround remain functional but unnecessary
            sd2.mps_ach[0] = cyc2.mph[0]
            sd2.mph_ach[0] = cyc2.mps[0]
            sd2.sim_drive()

            cyc = cycle.Cycle.from_file('udds')
            sdtot = simdrive.SimDrive(cyc, veh)
            sdtot.sim_drive()

            self.assertAlmostEqual(
                sd1.fs_kw_out_ach.sum() + sd2.fs_kw_out_ach.sum(),
                sdtot.fs_kw_out_ach.sum(),
                places=5)
        if RUST_AVAILABLE and USE_RUST:
            msg = f"Issue Running {type(self)}.test_split_cycles. Rust."
            t_clip = 210  # speed is non-zero here
            cyc1 = cycle.Cycle.from_dict(
                cyc_dict=(cycle.clip_by_times(cycle.Cycle.from_file(
                    'udds').get_cyc_dict(), t_end=t_clip))
            ).to_rust()
            t_end = cycle.Cycle.from_file('udds').time_s[-1]
            cyc2 = cycle.Cycle.from_dict(
                cyc_dict=(cycle.clip_by_times(cycle.Cycle.from_file('udds').get_cyc_dict(),
                                              t_start=t_clip, t_end=t_end))
            ).to_rust()

            veh = vehicle.Vehicle.from_vehdb(1).to_rust()

            sd1 = simdrive.RustSimDrive(cyc1, veh)
            sd1.sim_drive()

            sd2 = simdrive.RustSimDrive(cyc2, veh)
            # there is a limitation in Rust where we can't set an array at
            # index; we have to pull the entire array, mutate it, and set the
            # entire array.
            mps_ach = np.array(cyc2.mps)
            mps_ach[0] = cyc2.mph[0]
            sd2.mph_ach = mps_ach
            sd2.sim_drive()

            cyc = cycle.Cycle.from_file('udds').to_rust()
            sdtot = simdrive.RustSimDrive(cyc, veh)
            sdtot.sim_drive()

            self.assertAlmostEqual(
                np.array(sd1.fs_kw_out_ach).sum() +
                np.array(sd2.fs_kw_out_ach).sum(),
                np.array(sdtot.fs_kw_out_ach).sum(),
                places=5)

    def test_time_dilation(self):
        veh = vehicle.Vehicle.from_file('Line_Haul_Conv.csv')
        # make the vehicle really heavy so it'll trigger substantial time dilation
        veh.veh_kg *= 2
        cyc = cycle.Cycle.from_dict(cyc_dict={
            'time_s': np.arange(10),
            'mps': np.append(2, np.ones(9) * 6),
        })
        if USE_PYTHON:
            sd = simdrive.SimDrive(cyc, veh)
            sd.sim_params.missed_trace_correction = True
            sd.sim_params.max_time_dilation = 0.1  # maximum upper margin for time dilation
            sd.sim_drive()

            trace_miss_corrected = (
                abs(sd.dist_m.sum() - sd.cyc0.dist_m.sum()) / sd.cyc0.dist_m.sum()) < sd.sim_params.time_dilation_tol

            self.assertTrue(trace_miss_corrected,
                            msg="Issue in Python version")

        if RUST_AVAILABLE and USE_RUST:
            veh = veh.to_rust()
            cyc = cyc.to_rust()
            sd = simdrive.RustSimDrive(cyc, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                                              missed_trace_correction=True,
                                              max_time_dilation=0.1
                                              )
            sd.sim_drive()

            dist_err = np.abs(np.array(sd.dist_m).sum(
            ) - np.array(sd.cyc0.dist_m).sum()) / np.array(sd.cyc0.dist_m).sum()
            trace_miss_corrected = dist_err < sd.sim_params.time_dilation_tol

            self.assertTrue(sd.sim_params.missed_trace_correction)
            self.assertTrue(
                trace_miss_corrected,
                msg=(
                    f"Issue in Rust version; dist_err: {dist_err}"
                    + f"\ntrace_miss_iters: {np.max(sd.trace_miss_iters)}"
                )

            )

    def test_stop_start(self):
        if USE_PYTHON:
            cyc = cycle.Cycle.from_file('udds').get_cyc_dict()
            cyc = cycle.Cycle.from_dict(cyc_dict=cycle.clip_by_times(cyc, 130))

            veh = vehicle.Vehicle.from_vehdb(1)
            veh.stop_start = True
            veh.mc_max_kw = 1
            veh.ess_max_kw = 5
            veh.ess_max_kwh = 1
            veh.__post_init__()

            sd = simdrive.SimDrive(cyc, veh)
            sd.sim_drive()

            self.assertTrue(sd.fc_kw_in_ach[10] == 0)
            self.assertTrue(sd.fc_kw_in_ach[37] == 0)
        if RUST_AVAILABLE and USE_RUST:
            msg = "Issue with Rust version"
            cyc = cycle.Cycle.from_file('udds').to_rust().get_cyc_dict()
            cyc = cycle.Cycle.from_dict(
                cyc_dict=cycle.clip_by_times(cyc, 130)).to_rust()

            veh = vehicle.Vehicle.from_vehdb(1)
            veh.stop_start = True
            veh.mc_max_kw = 1
            veh.ess_max_kw = 5
            veh.ess_max_kwh = 1
            veh.__post_init__()
            veh = veh.to_rust()

            sd = simdrive.RustSimDrive(cyc, veh)
            sd.sim_drive()

            self.assertTrue(sd.fc_kw_in_ach[10] == 0, msg=msg)
            self.assertTrue(sd.fc_kw_in_ach[37] == 0, msg=msg)

    def test_achieved_speed_never_negative(self):
        if USE_PYTHON:
            for vehid in range(1, 27):
                veh = vehicle.Vehicle.from_vehdb(vehid)
                cyc = cycle.Cycle.from_file('udds')
                sd = simdrive.SimDrive(cyc, veh)
                sd.sim_drive()
                self.assertFalse(
                    (sd.mps_ach < 0.0).any(),
                    msg=f'Achieved speed contains negative values for vehicle {vehid}'
                )
                self.assertFalse(
                    (sd.mps_ach > sd.cyc0.mps).any(),
                    msg=f'Achieved speed is greater than requested speed for {vehid}'
                )
        if RUST_AVAILABLE and USE_RUST:
            for vehid in range(1, 27):
                veh = vehicle.Vehicle.from_vehdb(vehid).to_rust()
                cyc = cycle.Cycle.from_file('udds').to_rust()
                sd = simdrive.RustSimDrive(cyc, veh)
                sd.sim_drive()
                self.assertFalse(
                    (np.array(sd.mps_ach) < 0.0).any(),
                    msg=f'Achieved speed contains negative values for vehicle {vehid}'
                )
                self.assertFalse(
                    (np.array(sd.mps_ach) > np.array(sd.cyc0.mps)).any(),
                    msg=f'Achieved speed is greater than requested speed for {vehid}'
                )

    def test_that_vehdb_single_files_simulate(self):

        for filepath in vehicle.VEHICLE_DIR.iterdir():
            if "fail" not in filepath.name and filepath.suffix == '.csv' and "overrides" not in filepath.name:
                veh = vehicle.Vehicle.from_file(filepath)

        cyc = cycle.Cycle.from_file('udds')

        if USE_PYTHON:
            sd = simdrive.SimDrive(cyc, veh)
            sd.sim_drive()
            self.assertEqual(sd.i, sd.cyc0.len)

        if RUST_AVAILABLE and USE_RUST:
            sd = simdrive.RustSimDrive(cyc.to_rust(), veh.to_rust())
            sd.sim_drive()
            self.assertEqual(sd.i, sd.cyc0.len)


if __name__ == '__main__':
    unittest.main()
