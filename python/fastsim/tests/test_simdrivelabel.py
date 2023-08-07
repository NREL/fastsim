import unittest
import numpy as np

from fastsim import fastsimrust as fsr


class TestSimDriveLabel(unittest.TestCase):
    def test_get_label_fe_conv(self):
        veh = fsr.RustVehicle.mock_vehicle()
        label_fe, _ = fsr.get_label_fe(veh, False, False)  # Unpack the tuple
        # Because the full test is already implemented in Rust, we 
        # don't need a comprehensive check here.  
        self.assertEqual(label_fe.lab_udds_mpgge, 32.47503766676829)
        self.assertEqual(label_fe.lab_hwy_mpgge, 42.265348793379445)
        self.assertEqual(label_fe.lab_comb_mpgge, 36.25407690819302)

    def test_get_label_fe_phev(self):
        # Set up the required parameters and objects needed for testing get_label_fe_phev
        veh = fsr.RustVehicle.mock_vehicle()  
        label_fe, _ = fsr.get_label_fe(veh, False, False)
        self.assertEqual(label_fe.adj_udds_mpgge, 25.246151811422468)
       
        
if __name__ == '__main__':
    unittest.main()