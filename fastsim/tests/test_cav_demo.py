"""
Tests running the docs/cavs_demo.py file.
"""
import os
import unittest
import fastsim as fsim

class TestCavDemo(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()
    
    def test_that_demo_runs_without_error(self):
        is_interactive_key = 'FASTSIM_DEMO_IS_INTERACTIVE'
        original_value = os.getenv(is_interactive_key)
        os.environ[is_interactive_key] = 'False'
        from fastsim.docs.cav_demo import RAN_SUCCESSFULLY
        if original_value is not None:
            os.environ[is_interactive_key] = original_value
        else:
            del os.environ[is_interactive_key]
        self.assertTrue(RAN_SUCCESSFULLY)

if __name__ == '__main__':
    unittest.main()
