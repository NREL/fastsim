import unittest
from fastsim import utils

class test_utils(unittest.TestCase):
    def setUp(self):
        utils.disable_logging()
    
    def test_camel_to_snake(self):
        self.assertEqual("camel2_camel2_case", utils.camel_to_snake("camel2_camel2_case"))
        self.assertEqual("ess_ae_kw_out", utils.camel_to_snake("essAEKwOut"))
        self.assertEqual("elect_kw_req4_ae", utils.camel_to_snake("electKwReq4AE"))

if __name__ == '__main__':
    unittest.main()
