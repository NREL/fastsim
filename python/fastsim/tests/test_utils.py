import tempfile
import unittest
from pathlib import Path
import fastsim as fsim
from fastsim import utils

class TestUtils(unittest.TestCase):
    def setUp(self):
        utils.disable_logging()
    
    def test_camel_to_snake(self):
        self.assertEqual("camel2_camel2_case", utils.camel_to_snake("camel2_camel2_case"))
        self.assertEqual("ess_ae_kw_out", utils.camel_to_snake("essAEKwOut"))
        self.assertEqual("elect_kw_req4_ae", utils.camel_to_snake("electKwReq4AE"))

    def test_copy_demo_files(self):
        v = f"v{fsim.__version__}"
        prepend_str = f"# %% Copied from FASTSim version '{v}'. Guaranteed compatibility with this version only.\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            tf_path = Path(tmpdir)
            utils.copy_demo_files(tf_path)
            with open(next(tf_path.glob("*demo*.py")), 'r') as file:
                lines = file.readlines()
                assert prepend_str in lines[0]
                assert len(lines) > 3

if __name__ == '__main__':
    unittest.main()
