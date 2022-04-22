"""Test suite for cycle instantiation and manipulation."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from fastsim import parameters, vehicle
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable


USE_PYTHON = True
USE_RUST = True


class TestVehicle(unittest.TestCase):
    def test_equal(self):
        """Verify that a copied Vehicle and original are equal."""
        if USE_PYTHON:
            veh = vehicle.Vehicle.from_vehdb(1, verbose=False)
            veh_copy = vehicle.copy_vehicle(veh)
            self.assertTrue(vehicle.veh_equal(veh, veh_copy))
        if USE_RUST and not RUST_AVAILABLE:
            warn_rust_unavailable()
        if RUST_AVAILABLE and USE_RUST:
            py_veh = vehicle.Vehicle.from_vehdb(1, verbose=False)
            import fastsimrust as fsr
            data = {**py_veh.__dict__}
            data['fc_perc_out_array'] = np.copy(parameters.fc_perc_out_array)
            data['mc_perc_out_array'] = np.copy(parameters.mc_perc_out_array)
            data['props'] = parameters.copy_physical_properties(py_veh.props, 'rust')
            veh = fsr.RustVehicle(**data)
            veh_copy = vehicle.copy_vehicle(veh, 'rust')
            self.assertTrue(vehicle.veh_equal(veh, veh_copy))
            self.assertTrue(vehicle.veh_equal(py_veh, veh_copy))

    def test_properties(self):
        """Verify that some of the property variables are working as expected."""
        if USE_PYTHON:
            veh = vehicle.Vehicle.from_vehdb(10, verbose=False)
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
            veh.mc_peak_eff = 0.85
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
            veh.mc_peak_eff += 0.05
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
            veh.mc_full_eff_array *= 1.05
            veh.mc_eff_array *= 1.05
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
        if USE_RUST:
            veh = vehicle.Vehicle.from_vehdb(10, verbose=False).to_rust()
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
            veh.mc_peak_eff = 0.85
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
            veh.mc_peak_eff += 0.05
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
            veh.mc_full_eff_array = np.array(veh.mc_full_eff_array) * 1.05
            veh.mc_eff_array = np.array(veh.mc_eff_array) * 1.05
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
            self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
    
    def test_fc_efficiency_override(self):
        """Verify that we can scale FC"""
        TOL = 1e-6
        def add_csv_parameter(param_name:str, value:float, pristine_path:Path, temp_dirname:Path)->Path:
            csv_contents = None
            with open(pristine_path, 'r') as f:
                csv_contents = f.read()
            csv_contents += f',{param_name},{str(value)},\n'
            assert param_name in csv_contents, f"Can't find param_name '{param_name}' in csv_contents!"
            test_path = Path(temp_dirname) / "Test_Vehicle.csv"
            with open(test_path, 'w') as f:
                f.write(csv_contents)
            return test_path
        veh_name = "2012_Ford_Fusion.csv"
        pristine_path = Path(vehicle.VEHICLE_DIR) / veh_name
        test_peak_eff = 0.5
        if USE_PYTHON:
            veh_pristine = vehicle.Vehicle.from_file(veh_name)
            pristine_fc_peak_eff = veh_pristine.fc_peak_eff
            pristine_mc_peak_eff = veh_pristine.mc_peak_eff
            with tempfile.TemporaryDirectory() as temp_dirname:
                test_path = add_csv_parameter("fc_peak_eff_override", test_peak_eff, pristine_path, temp_dirname)
                veh = vehicle.Vehicle.from_file(test_path)
                self.assertAlmostEqual(test_peak_eff, veh.fc_peak_eff)
                self.assertTrue(abs(pristine_fc_peak_eff - veh.fc_peak_eff) > TOL)
                self.assertAlmostEqual(pristine_mc_peak_eff, veh.mc_peak_eff)
                test_path = add_csv_parameter("mc_peak_eff_override", test_peak_eff, pristine_path, temp_dirname)
                veh = vehicle.Vehicle.from_file(test_path)
                self.assertAlmostEqual(test_peak_eff, veh.mc_peak_eff)
                self.assertTrue(abs(pristine_mc_peak_eff - veh.mc_peak_eff) > TOL)
                self.assertAlmostEqual(pristine_fc_peak_eff, veh.fc_peak_eff)
        if USE_RUST:
            veh_pristine = vehicle.Vehicle.from_file(veh_name).to_rust()
            pristine_fc_peak_eff = veh_pristine.fc_peak_eff
            pristine_mc_peak_eff = veh_pristine.mc_peak_eff
            with tempfile.TemporaryDirectory() as temp_dirname:
                test_path = add_csv_parameter("fc_peak_eff_override", test_peak_eff, pristine_path, temp_dirname)
                veh = vehicle.Vehicle.from_file(test_path).to_rust()
                self.assertAlmostEqual(test_peak_eff, veh.fc_peak_eff)
                self.assertTrue(abs(pristine_fc_peak_eff - veh.fc_peak_eff) > TOL)
                self.assertAlmostEqual(pristine_mc_peak_eff, veh.mc_peak_eff)
                test_path = add_csv_parameter("mc_peak_eff_override", test_peak_eff, pristine_path, temp_dirname)
                veh = vehicle.Vehicle.from_file(test_path).to_rust()
                self.assertAlmostEqual(test_peak_eff, veh.mc_peak_eff)
                self.assertTrue(abs(pristine_mc_peak_eff - veh.mc_peak_eff) > TOL)
                self.assertAlmostEqual(pristine_fc_peak_eff, veh.fc_peak_eff)


if __name__ == '__main__':
    from fastsim import vehicle 
    veh = vehicle.Vehicle.from_vehdb(1)