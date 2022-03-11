"""Test various copy utilities"""

import unittest

import numpy as np

import fastsim.vehicle_base as fsvb
from fastsim import cycle, params, utils, vehicle, simdrive
import fastsimrust as fsr


class TestCopy(unittest.TestCase):
    def test_copy_cycle(self):
        "Test that cycle_copy works as expected"
        cyc = cycle.Cycle.from_file('udds')
        self.assertEqual(cycle.Cycle, type(cyc))
        cyc2 = cycle.copy_cycle(cyc)
        self.assertEqual(cycle.Cycle, type(cyc2))
        self.assertFalse(cyc is cyc2, msg="Ensure we actually copied; that we don't just have the same object")
        cyc_dict = cycle.copy_cycle(cyc, 'dict')
        self.assertEqual(dict, type(cyc_dict))
        rust_cyc = cycle.copy_cycle(cyc, 'rust')
        self.assertEqual(type(rust_cyc), fsr.RustCycle)
        rust_cyc2 = cycle.copy_cycle(rust_cyc)
        self.assertEqual(type(rust_cyc2), fsr.RustCycle)
        rust_cyc3 = cycle.Cycle.from_file('udds').to_rust()
        self.assertEqual(type(rust_cyc3), fsr.RustCycle)
        for key in cycle.NEW_TO_OLD.keys():
            if type(cyc.__getattribute__(key)) is np.ndarray:
                self.assertEqual(
                    len(cyc.__getattribute__(key)),
                    len(rust_cyc3.__getattribute__(key)),
                    msg=f"Value lengths are not equal for {key}\n"
                )
                self.assertTrue(
                    (cyc.__getattribute__(key) == np.array(rust_cyc3.__getattribute__(key))).all(),
                    msg=(
                        f"Values are not equal for {key}\n"
                        + f"Python Cycle: ({type(cyc.__getattribute__(key))}) {cyc.__getattribute__(key)}\n"
                        + f"Rust Cycle  : ({type(rust_cyc3.__getattribute__(key))}) {rust_cyc3.__getattribute__(key)}"
                ))
            else:
                self.assertEqual(
                    cyc.__getattribute__(key),
                    rust_cyc3.__getattribute__(key),
                    msg=(
                        f"Values are not equal for {key}\n"
                        + f"Python Cycle: ({type(cyc.__getattribute__(key))}) {cyc.__getattribute__(key)}\n"
                        + f"Rust Cycle  : ({type(rust_cyc3.__getattribute__(key))}) {rust_cyc3.__getattribute__(key)}"
                ))
    
    def test_copy_physical_properties(self):
        "Test that copy_physical_properties works as expected"
        p = params.PhysicalProperties()
        self.assertEqual(params.PhysicalProperties, type(p))
        p2 = params.copy_physical_properties(p)
        self.assertEqual(params.PhysicalProperties, type(p2))
        self.assertFalse(p is p2, msg="Ensure we actually copied; that we don't just have the same object")
        p_dict = params.copy_physical_properties(p, 'dict')
        self.assertEqual(dict, type(p_dict))
        rust_p = params.copy_physical_properties(p, 'rust')
        self.assertEqual(type(rust_p), fsr.RustPhysicalProperties)
        rust_p2 = params.copy_physical_properties(rust_p)
        self.assertEqual(type(rust_p2), fsr.RustPhysicalProperties)
        rust_p3 = params.PhysicalProperties().to_rust()
        self.assertEqual(type(rust_p3), fsr.RustPhysicalProperties)
        for key in params.ref_physical_properties.__dict__.keys():
            self.assertEqual(
                p.__getattribute__(key),
                rust_p3.__getattribute__(key),
                msg=(
                    f"Values are not equal for {key}\n"
                    + f"Python Physical Properties: ({type(p.__getattribute__(key))}) {p.__getattribute__(key)}\n"
                    + f"Rust Physical Properties  : ({type(rust_p3.__getattribute__(key))}) {rust_p3.__getattribute__(key)}"
            ))
    
    def test_copy_vehicle(self):
        "Test that vehicle_copy works as expected"
        veh = vehicle.Vehicle.from_vehdb(5)
        self.assertEqual(vehicle.Vehicle, type(veh))
        veh2 = vehicle.copy_vehicle(veh)
        self.assertEqual(vehicle.Vehicle, type(veh2))
        self.assertFalse(veh is veh2, msg="Ensure we actually copied; that we don't just have the same object")
        veh_dict = vehicle.copy_vehicle(veh, 'dict')
        self.assertEqual(dict, type(veh_dict))
        rust_veh = vehicle.copy_vehicle(veh, 'rust')
        self.assertEqual(type(rust_veh), fsr.RustVehicle)
        rust_veh2 = vehicle.copy_vehicle(rust_veh)
        self.assertEqual(type(rust_veh2), fsr.RustVehicle)
        rust_veh3 = vehicle.Vehicle.from_vehdb(5).to_rust()
        self.assertEqual(type(rust_veh3), fsr.RustVehicle)
        for key in fsvb.keys_and_types.keys():
            if type(veh.__getattribute__(key)) is np.ndarray:
                self.assertEqual(
                    len(veh.__getattribute__(key)),
                    len(rust_veh3.__getattribute__(key)),
                    msg=f"Value lengths are not equal for {key}\n"
                )
                self.assertTrue(
                    (veh.__getattribute__(key) == np.array(rust_veh3.__getattribute__(key))).all(),
                    msg=(
                        f"Values are not equal for {key}\n"
                        + f"Python Vehicle: ({type(veh.__getattribute__(key))}) {veh.__getattribute__(key)}\n"
                        + f"Rust Vehicle  : ({type(rust_veh3.__getattribute__(key))}) {rust_veh3.__getattribute__(key)}"
                ))
            elif type(veh.__getattribute__(key)) is params.PhysicalProperties:
                # TODO: implement comparison
                pass
            else:
                if type(veh.__getattribute__(key)) is np.float64 and np.isnan(veh.__getattribute__(key)):
                    self.assertTrue(np.isnan(rust_veh3.__getattribute__(key)))
                else:
                    self.assertEqual(
                        veh.__getattribute__(key),
                        rust_veh3.__getattribute__(key),
                        msg=(
                            f"Values are not equal for {key}\n"
                            + f"Python Vehicle: ({type(veh.__getattribute__(key))}) {veh.__getattribute__(key)}\n"
                            + f"Rust Vehicle  : ({type(rust_veh3.__getattribute__(key))}) {rust_veh3.__getattribute__(key)}"
                    ))

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
        rust_sdp = simdrive.copy_sim_params(sdp, 'rust')
        self.assertEqual(fsr.RustSimDriveParams, type(rust_sdp))
        self.assertTrue(simdrive.sim_params_equal(sdp, rust_sdp), msg="Assert that values equal")