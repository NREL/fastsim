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
    
    def test_vehicle_copy(self):
        "Test that vehicle_copy works as expected"
        veh = vehicle.Vehicle.from_vehdb(5)
        self.assertEqual(vehicle.Vehicle, type(veh))
        veh2 = vehicle.copy_vehicle(veh)
        self.assertEqual(vehicle.Vehicle, type(veh2))
        rust_veh = vehicle.copy_vehicle(veh, 'rust')
        self.assertEqual(type(rust_veh), fsr.RustVehicle)