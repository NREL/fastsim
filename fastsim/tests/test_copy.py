"""Test various copy utilities"""

import unittest

import numpy as np

import fastsim as fsim
import fastsim.vehicle_base as fsvb
from fastsim import cycle, params, utils, vehicle, simdrive
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable
if RUST_AVAILABLE:
    import fastsimrust as fsr
else:
    warn_rust_unavailable(__file__)


class TestCopy(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()
    
    def test_copy_cycle(self):
        "Test that cycle_copy works as expected"
        cyc = cycle.Cycle.from_file('udds')
        self.assertEqual(cycle.Cycle, type(cyc))
        cyc2 = cycle.copy_cycle(cyc)
        self.assertEqual(cycle.Cycle, type(cyc2))
        self.assertFalse(cyc is cyc2, msg="Ensure we actually copied; that we don't just have the same object")
        cyc_dict = cycle.copy_cycle(cyc, 'dict')
        self.assertEqual(dict, type(cyc_dict))
        if RUST_AVAILABLE:
            rust_cyc = cycle.copy_cycle(cyc, 'rust')
            self.assertEqual(type(rust_cyc), fsr.RustCycle)
            rust_cyc2 = cycle.copy_cycle(rust_cyc)
            self.assertEqual(type(rust_cyc2), fsr.RustCycle)
            rust_cyc3 = cycle.Cycle.from_file('udds').to_rust()
            self.assertEqual(type(rust_cyc3), fsr.RustCycle)
            self.assertTrue(cycle.cyc_equal(cyc, rust_cyc3))
            cyc.name = 'bob'
            self.assertFalse(cycle.cyc_equal(cyc, rust_cyc3))
    
    def test_copy_physical_properties(self):
        "Test that copy_physical_properties works as expected"
        p = params.PhysicalProperties()
        self.assertEqual(params.PhysicalProperties, type(p))
        p2 = params.copy_physical_properties(p)
        self.assertEqual(params.PhysicalProperties, type(p2))
        self.assertFalse(p is p2, msg="Ensure we actually copied; that we don't just have the same object")
        p_dict = params.copy_physical_properties(p, 'dict')
        self.assertEqual(dict, type(p_dict))
        if RUST_AVAILABLE:
            rust_p = params.copy_physical_properties(p, 'rust')
            self.assertEqual(type(rust_p), fsr.RustPhysicalProperties)
            rust_p2 = params.copy_physical_properties(rust_p)
            self.assertEqual(type(rust_p2), fsr.RustPhysicalProperties)
            rust_p3 = params.PhysicalProperties().to_rust()
            self.assertEqual(type(rust_p3), fsr.RustPhysicalProperties)
            self.assertTrue(params.physical_properties_equal(p, rust_p))
            p.a_grav_mps2 = 10.0
            self.assertFalse(params.physical_properties_equal(p, rust_p))
    
    def test_copy_vehicle(self):
        "Test that vehicle_copy works as expected"
        veh = vehicle.Vehicle.from_vehdb(5)
        self.assertEqual(vehicle.Vehicle, type(veh))
        veh2 = vehicle.copy_vehicle(veh)
        self.assertEqual(vehicle.Vehicle, type(veh2))
        self.assertFalse(veh is veh2, msg="Ensure we actually copied; that we don't just have the same object")
        veh_dict = vehicle.copy_vehicle(veh, 'dict')
        self.assertEqual(dict, type(veh_dict))
        if RUST_AVAILABLE:
            rust_veh = vehicle.copy_vehicle(veh, 'rust')
            self.assertEqual(type(rust_veh), fsr.RustVehicle)
            rust_veh2 = vehicle.copy_vehicle(rust_veh)
            self.assertEqual(type(rust_veh2), fsr.RustVehicle)
            rust_veh3 = vehicle.Vehicle.from_vehdb(5).to_rust()
            self.assertEqual(type(rust_veh3), fsr.RustVehicle)
            self.assertTrue(
                vehicle.veh_equal(veh, rust_veh3),
                msg=f"Error list: {str(vehicle.veh_equal(veh, rust_veh3, full_out=True))}")

    def test_copy_sim_params(self):
        "Test that copy_sim_params works as expected"
        sdp = simdrive.SimDriveParams()
        self.assertEqual(simdrive.SimDriveParams, type(sdp))
        sdp2 = simdrive.copy_sim_params(sdp)
        self.assertEqual(simdrive.SimDriveParams, type(sdp2))
        self.assertFalse(sdp is sdp2, msg="Ensure we actually copied; that we don't just have the same object")
        self.assertTrue(simdrive.sim_params_equal(sdp, sdp2), msg="We want different objects but equal values")
        sdp_dict = simdrive.copy_sim_params(sdp, 'dict')
        self.assertEqual(dict, type(sdp_dict))
        self.assertEqual(len(sdp_dict), len(simdrive.ref_sim_drive_params.__dict__))
        if RUST_AVAILABLE:
            rust_sdp = simdrive.copy_sim_params(sdp, 'rust')
            self.assertEqual(fsr.RustSimDriveParams, type(rust_sdp))
            self.assertTrue(
                simdrive.sim_params_equal(sdp, rust_sdp),
                msg="Assert that values equal")
            rust_sdp2 = simdrive.copy_sim_params(rust_sdp)
            self.assertEqual(type(rust_sdp2), fsr.RustSimDriveParams)
            self.assertFalse(rust_sdp is rust_sdp2)
            self.assertTrue(simdrive.sim_params_equal(rust_sdp, rust_sdp2))
            rust_sdp3 = simdrive.SimDriveParams().to_rust()
            self.assertEqual(type(rust_sdp3), fsr.RustSimDriveParams)
            self.assertTrue(simdrive.sim_params_equal(simdrive.SimDriveParams(), rust_sdp3))
    
    def test_copy_sim_drive(self):
        "Test that copy_sim_drive works as expected"
        cyc = cycle.Cycle.from_file('udds')
        veh = vehicle.Vehicle.from_vehdb(5)
        sd = simdrive.SimDrive(cyc, veh)
        self.assertEqual(simdrive.SimDrive, type(sd))
        sd2 = simdrive.copy_sim_drive(sd)
        self.assertEqual(simdrive.SimDrive, type(sd2))
        self.assertFalse(sd is sd2, msg="Ensure we actually copied; that we don't just have the same object")
        if RUST_AVAILABLE:
            rust_sd = simdrive.copy_sim_drive(sd, 'rust')
            self.assertEqual(type(rust_sd), fsr.RustSimDrive)
            rust_sd2 = simdrive.copy_sim_drive(rust_sd)
            self.assertEqual(type(rust_sd2), fsr.RustSimDrive)
            self.assertTrue(simdrive.sim_drive_equal(sd, rust_sd))
            original_i = sd.i
            sd.i = original_i + 1
            self.assertFalse(simdrive.sim_drive_equal(sd, rust_sd))
            sd.i = original_i
            rust_sd3 = simdrive.SimDrive(cyc, veh).to_rust()
            self.assertEqual(type(rust_sd3), fsr.RustSimDrive)
            self.assertTrue(simdrive.sim_drive_equal(rust_sd3, sd))
            self.assertTrue(simdrive.sim_drive_equal(rust_sd3, rust_sd))

if __name__ == '__main__':
    unittest.main()
