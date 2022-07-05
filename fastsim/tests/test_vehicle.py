"""Test suite for cycle instantiation and manipulation."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import vehicle


class TestVehicle(unittest.TestCase):
    def test_equal(self):
        """Verify that a copied VehicleJit and identically instantiated Vehicle are equal."""
        
        print(f"Running {type(self)}.test_equal.")
        veh = vehicle.Vehicle(1, verbose=False)
        veh_jit = veh.get_numba_veh()
        veh_jit_copy = vehicle.copy_vehicle(veh_jit)
        self.assertTrue(vehicle.veh_equal(veh_jit, veh_jit_copy))

    def test_properties(self):
        """Verify that some of the property variables are working as expected."""

        print(f"Running {type(self)}.test_properties.")
        veh = vehicle.Vehicle(10, verbose=False)
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))
        veh.mcPeakEff = 0.85
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))
        veh.mcPeakEff += 0.05
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))
        veh.mcFullEffArray *= 1.05
        veh.mcEffArray *= 1.05
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))

    def test_set_dependents(self):
        veh = vehicle.Vehicle(1).get_numba_veh()
        veh.set_dependents()

    def test_file_overrides(self):
        veh = vehicle.Vehicle('test_overrides')
        self.assertAlmostEqual(veh.mcPeakEff, 0.2, 3)
        self.assertAlmostEqual(veh.fcPeakEff, 0.9, 3)
        with self.assertRaises(AssertionError):
            vehicle.Vehicle('fail_overrides')
    
    def test_mapping(self):
        skip_keys = ["props", "Selection"]
        vnums = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        filenames = [
            "2010_Mazda_3_i-Stop.csv",
            "2012_Ford_Focus.csv",
            "2012_Ford_Fusion.csv",
            "2017_Toyota_Highlander_3.5_L.csv",
            "Class_4_Box_Truck.csv",
            "2020_EU_VW_Golf_1.5TSI.csv",
            "2020_EU_VW_Golf_2.0TDI.csv",
            "2022_Renault_Zoe_ZE50_R135.csv",
            "2022_TOYOTA_Yaris_Hybrid_Mid.csv",
            "2016_TOYOTA_Prius_Two.csv",
        ]
        db_vehicles = [vehicle.Vehicle(vnum, verbose=False) for vnum in vnums]
        csv_vehicles = [vehicle.Vehicle(filename, verbose=False) for filename in filenames]
        for db_veh, csv_veh in zip(db_vehicles, csv_vehicles):
            compare_attrs = db_veh.__dict__
            for skip_key in skip_keys:
                compare_attrs.pop(skip_key)
            for key, db_value in compare_attrs.items():
                csv_value = csv_veh.__dict__[key]
                if isinstance(db_value, (list, np.ndarray)):
                    self.assertListEqual(list(db_value), list(csv_value), f"Key '{key}' not equal")
                else:
                    if not isinstance(db_value, str):
                        if np.isnan(db_value) and np.isnan(csv_value):
                            continue
                    self.assertEqual(db_value, csv_value, f"Key '{key}' not equal")


if __name__ == '__main__':
    unittest.main()