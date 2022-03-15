"""
Tests using the Rust versions of SimDrive, Cycle, and Vehicle
"""
import unittest

import numpy as np

import fastsim.vehicle_base as fsvb
from fastsim import cycle, vehicle, simdrive
import fastsimrust as fsr


class TestRust(unittest.TestCase):
    def test_run_sim_drive(self):
        cyc = cycle.Cycle.from_file('udds').to_rust()
        veh = vehicle.Vehicle.from_vehdb(5).to_rust()
        #sd = simdrive.SimDrive(cyc, veh).to_rust()
        sd = fsr.RustSimDrive(cyc, veh)
        sd.sim_drive_walk(0.5)