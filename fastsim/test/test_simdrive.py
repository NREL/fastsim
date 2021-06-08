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
        cyc = cycle.Cycle('udds')
        veh = vehicle.Vehicle(1)
        sim_drive = simdrive.SimDriveClassic(cyc, veh)

        for x in range(100):
            sim_drive.sim_drive_step()
        
        self.assertEqual(sim_drive.fsKwOutAch[100], 29.32533101828119)
    
    def test_sim_drive_walk(self):
        """Verify thta sim_drive_walk produces an expected result."""
        cyc = cycle.Cycle('udds')
        veh = vehicle.Vehicle(1)
        sim_drive = simdrive.SimDriveClassic(cyc, veh)
        sim_drive.sim_drive_walk(initSoc=1)

        self.assertEqual(sim_drive.fsKwOutAch.sum(), 24410.31348426869)
