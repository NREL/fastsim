"""
Tests running the docs/cavs_demo.py file.
"""
import os
import unittest
import fastsim as fsim
import fastsim.tests.utils as utils

class TestCavDemo(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()
    
    def test_that_demo_runs_without_error(self):
        orig = utils.start_demo_environment()
        from fastsim.demos.cav_demo import RAN_SUCCESSFULLY
        utils.end_demo_test_environment(orig)
        self.assertTrue(RAN_SUCCESSFULLY)

if __name__ == '__main__':
    unittest.main()
