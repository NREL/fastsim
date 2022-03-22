"""
Tests using the Rust versions of SimDrive, Cycle, and Vehicle
"""
import unittest

import numpy as np

import fastsim.vehicle_base as fsvb
from fastsim import cycle, vehicle, simdrive
import fastsimrust as fsr


class TestRust(unittest.TestCase):
    def test_run_sim_drive_conv(self):
        cyc = cycle.Cycle.from_file('udds').to_rust()
        veh = vehicle.Vehicle.from_vehdb(5).to_rust()
        #sd = simdrive.SimDrive(cyc, veh).to_rust()
        sd = fsr.RustSimDrive(cyc, veh)
        sd.sim_drive_walk(0.5)
        self.assertTrue(sd.i > 1)
        self.assertEqual(sd.i, len(cyc.time_s))

    def test_run_sim_drive_conv(self):
        cyc = cycle.Cycle.from_file('udds').to_rust()
        veh = vehicle.Vehicle.from_vehdb(11).to_rust()
        sd = fsr.RustSimDrive(cyc, veh)
        sd.sim_drive_walk(0.5)
        self.assertTrue(sd.i > 1)
        self.assertEqual(sd.i, len(cyc.time_s))
    
    def test_fueling_prediction_for_multiple_vehicle(self):
        for vehid in [1, 9, 14, 17, 24]:
            cyc = cycle.Cycle.from_file('udds')
            veh = vehicle.Vehicle.from_vehdb(vehid)
            sd = simdrive.SimDrive(cyc, veh)
            sd.sim_drive_walk(0.5)
            sd.set_post_scalars()
            py_fuel_kj = sd.fuel_kj
            py_ess_dischg_kj = sd.ess_dischg_kj
            cyc = cycle.Cycle.from_file('udds').to_rust()
            veh = vehicle.Vehicle.from_vehdb(vehid).to_rust()
            sd = fsr.RustSimDrive(cyc, veh)
            sd.sim_drive_walk(0.5)
            sd.set_post_scalars()
            rust_fuel_kj = sd.fuel_kj
            rust_ess_dischg_kj = sd.ess_dischg_kj
            self.assertAlmostEqual(py_fuel_kj, rust_fuel_kj, msg=f'Non-agreement for vehicle {vehid} for fuel')
            self.assertAlmostEqual(py_ess_dischg_kj, rust_ess_dischg_kj, msg=f'Non-agreement for vehicle {vehid} for ess discharge')

    def test_that_engine_input_power_is_close_between_rust_and_python(self):
        pass