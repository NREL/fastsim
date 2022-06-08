import unittest
from fastsim import auxiliaries
from fastsim.vehicle import Vehicle

class test_auxiliaries(unittest.TestCase):
    def test_abc_to_drag_coeffs(self):
        veh = Vehicle.from_vehdb(1).to_rust()
        a = 25.91
        b = 0.1943
        c = 0.01796
        drag_coeff, wheel_rr_coef = auxiliaries.abc_to_drag_coeffs(veh=veh,
                                                                   a_lbf=a, 
                                                                   b_lbf__mph=b, 
                                                                   c_lbf__mph2=c,
                                                                   custom_rho=False,
                                                                   simdrive_optimize=True,
                                                                   show_plots=False,
                                                                   use_rust=True)
        self.assertAlmostEqual(0.29852841290057, drag_coeff)
        self.assertAlmostEqual(0.00805627626436443, wheel_rr_coef)

    def test_drag_coeffs_to_abc(self):
        veh = Vehicle.from_vehdb(1).to_rust()
        a_lbf, b_lbf__mph, c_lbf__mph2 = auxiliaries.drag_coeffs_to_abc(veh=veh,
                                                                        fit_with_curve=False,
                                                                        show_plots=False)
        self.assertAlmostEqual(34.26168611118723, a_lbf)
        self.assertAlmostEqual(0, b_lbf__mph)
        self.assertAlmostEqual(0.020817239083920212, c_lbf__mph2)