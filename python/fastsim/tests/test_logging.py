import unittest

import fastsim as fsim
import logging

class TestCase(unittest.TestCase):
    def test_python_logging(self):
        fastsim_logger = logging.getLogger("fastsim")
        with self.assertLogs("fastsim", level="DEBUG"):
            fastsim_logger.warning("test log")
    
    def test_rust_logging(self):
        fastsimrust_logger = logging.getLogger("fastsimrust")
        with self.assertLogs("fastsimrust", level="DEBUG"):
            fastsimrust_logger.warning("test log")

    # TODO: add tests for disabling logging

if __name__ == "__main__":
    unittest.main()
