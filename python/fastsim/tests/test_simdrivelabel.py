import unittest
from fastsim import fastsimrust as fsr

class TestSimDriveLabel(unittest.TestCase):
    def test_get_label_fe_conv(self):
        veh = fsr.RustVehicle.mock_vehicle()
        # TODO: make this test pass! (delete any TODO comments after success)
        # You'll need to expose `get_label_fe` by making:
        # ```
        # #[pyo3(name = "get_label_fe")]
        # fn get_label_fe_py(...) ...
        # see if you can figure out function body
        # ```
        # and adding the appropriate function wrapper in `fastsim-py/src/lib.rs`
        label_fe, _ = fsr.get_label_fe(veh, False, False)  # Unpack the tuple
        # Because the full test is already implemented in Rust, we 
        # don't need a comprehensive check here.  
        self.assertEqual(label_fe.lab_udds_mpgge, 32.47503766676829)
        self.assertEqual(label_fe.lab_hwy_mpgge, 42.265348793379445)
        self.assertEqual(label_fe.lab_comb_mpgge, 36.25407690819302)

    def test_get_label_fe_phev(self):
        # TODO: fill this out similar to `test_get_label_fe_conv`
        delete_me = "delete this placeholder line"

if __name__ == '__main__':
    unittest.main()
