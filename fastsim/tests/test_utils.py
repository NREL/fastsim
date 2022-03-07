import unittest
from fastsim import utils

class test_utils(unittest.TestCase):
    def test_camel_to_snake(self):
        self.assertEqual("camel2_camel2_case", utils.camel_to_snake("camel2_camel2_case"))
        self.assertEqual("essAEKwOut", utils.camel_to_snake("ess_ae_kw_out"))
        self.assertEqual("electKwReq4AE", utils.camel_to_snake("elect_kw_req4_ae"))