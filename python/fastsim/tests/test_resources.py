"""Test getting resource lists via Rust API"""
import unittest

import fastsim as fsim
from fastsim import cycle, vehicle

class TestListResources(unittest.TestCase):
    def test_list_resources_for_cycle(self):
        "check if list_resources works and yields results for Cycle"
        c = cycle.Cycle.from_dict({
            "cycSecs": [0.0, 1.0],
            "cycMps": [0.0, 0.0]})
        rc = c.to_rust()
        resources = rc.list_resources()
        self.assertTrue(len(resources) > 0)
