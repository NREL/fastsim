"""Test various copy utilities"""

import unittest

import numpy as np

from fastsim import cycle, params, utils, vehicle
import fastsimrust as fsr


class TestCopy(unittest.TestCase):
    def test_cycle_copy(self):
        "Test that cycle_copy works as expected"
        cyc = cycle.Cycle.from_file('udds')
        self.assertEqual(cycle.Cycle, type(cyc))
        cyc2 = cycle.copy_cycle(cyc)
        self.assertEqual(cycle.Cycle, type(cyc2))
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
    
    def test_physical_properties_copy(self):
        "Test that copy_physical_properties works as expected"
        p = params.PhysicalProperties()
        self.assertEqual(params.PhysicalProperties, type(p))
        p2 = params.copy_physical_properties(p)
        self.assertEqual(params.PhysicalProperties, type(p2))
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
    
    def test_vehicle_copy(self):
        "Test that vehicle_copy works as expected"
        veh = vehicle.Vehicle.from_vehdb(5)
        self.assertEqual(vehicle.Vehicle, type(veh))
        veh2 = vehicle.copy_vehicle(veh)
        self.assertEqual(vehicle.Vehicle, type(veh2))
        rust_veh = vehicle.copy_vehicle(veh, 'rust')
        self.assertEqual(type(rust_veh), fsr.RustVehicle)
        rust_veh2 = vehicle.copy_vehicle(rust_veh)
        self.assertEqual(type(rust_veh2), fsr.RustVehicle)