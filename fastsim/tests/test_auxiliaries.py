import unittest
from fastsim import auxiliaries
from fastsim.vehicle import Vehicle
from fastsim import utils
from fastsimrust import abc_to_drag_coeffs
import numpy as np

class test_auxiliaries(unittest.TestCase):
    def setUp(self):
        utils.disable_logging()
    
    def test_abc_to_drag_coeffs(self):
        with np.errstate(divide='ignore'):
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
                                                                    use_rust=False)
            self.assertAlmostEqual(0.29396666380768194, drag_coeff, places=5)
            self.assertAlmostEqual(0.00836074507871098, wheel_rr_coef, places=7)

    def test_drag_coeffs_to_abc(self):
        veh = Vehicle.from_vehdb(1).to_rust()
        a_lbf, b_lbf__mph, c_lbf__mph2 = auxiliaries.drag_coeffs_to_abc(veh=veh,
                                                                        fit_with_curve=False,
                                                                        show_plots=False)
        self.assertAlmostEqual(34.26168611118723, a_lbf)
        self.assertAlmostEqual(0, b_lbf__mph)
        self.assertAlmostEqual(0.020817239083920212, c_lbf__mph2)

    def test_abc_to_drag_coeffs_rust_port(self):
        with np.errstate(divide='ignore'):
            veh = Vehicle.from_vehdb(5).to_rust()
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
                                                                    use_rust=False)
            drag_coeff_rust, wheel_rr_coef_rust = abc_to_drag_coeffs(veh=veh,
                                                        a_lbf=a, 
                                                        b_lbf__mph=b, 
                                                        c_lbf__mph2=c,
                                                        custom_rho=False,
                                                        simdrive_optimize=True,
                                                        _show_plots=False)
            self.assertAlmostEqual(drag_coeff_rust, drag_coeff, places=4)
            self.assertAlmostEqual(wheel_rr_coef_rust, wheel_rr_coef, places=6)

if __name__ == '__main__':
    unittest.main()
