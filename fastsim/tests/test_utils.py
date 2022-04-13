import unittest
from fastsim import utils

class test_utils(unittest.TestCase):
    def test_camel_to_snake(self):
        self.assertEqual("camel2_camel2_case", utils.camel_to_snake("camel2_camel2_case"))
        self.assertEqual("ess_ae_kw_out", utils.camel_to_snake("essAEKwOut"))
        self.assertEqual("elect_kw_req4_ae", utils.camel_to_snake("electKwReq4AE"))

    def test_abc_to_drag_coeffs(self):
        drag_coeff, wheel_rr_coef = utils.abc_to_drag_coeffs(1_500, 4, 40, 0.02, 0.03)
        self.assertEqual(0.2806902916694236, drag_coeff)
        self.assertEqual(0.012170311687114816, wheel_rr_coef)

    def test_drag_coeffs_to_abc(self):
        a_lbf, b_lbf__mph, c_lbf__mph2 = utils.drag_coeffs_to_abc(1_500, 4, 0.2806902916694236, 0.012170311687114816)
        self.assertEqual(40.262170995136486, a_lbf)
        self.assertEqual(-1.900837080788384e-09, b_lbf__mph)
        self.assertEqual(0.03026779028777221, c_lbf__mph2)