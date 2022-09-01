"""
Test the eco-cruise feature in FASTSim
"""
import unittest

import numpy as np

import fastsim as fs


class TestEcoCruise(unittest.TestCase):
    def percent_distance_error(self, sd: fs.simdrive.SimDrive) -> float:
        d0 = np.array(sd.cyc0.dist_m).sum()
        d = np.array(sd.cyc.dist_m).sum()
        return 100.0 * (d - d0) / d0
    
    def cycles_differ(self, sd: fs.simdrive.SimDrive) -> bool:
        if (len(sd.cyc0.time_s) != len(sd.cyc.time_s)):
            return True
        for idx in range(len(sd.cyc0.time_s)):
            if sd.cyc0.time_s[idx] != sd.cyc.time_s[idx]:
                return True
            if abs(sd.cyc0.mps[idx] - sd.cyc.mps[idx]) > 0.01:
                return True
            if abs(sd.cyc0.grade[idx] - sd.cyc.grade[idx]) > 0.01:
                return True
        return False
    
    def make_simdrive(self, use_rust=False):
        cyc = fs.cycle.Cycle.from_file('udds')
        veh = fs.vehicle.Vehicle.from_vehdb(1)
        if use_rust:
            cyc = cyc.to_rust()
            veh = veh.to_rust()
            sd = fs.simdrive.RustSimDrive(cyc, veh)
        else:
            sd = fs.simdrive.SimDrive(cyc, veh)
        return sd, cyc, veh
    
    def do_standard_checks(self, sd, msg=None):
        m = msg + ': ' if msg is not None else ''
        self.assertTrue(self.cycles_differ(sd), f"{m}Cycles should differ when running with eco-cruise")
        self.assertTrue(abs(self.percent_distance_error(sd)) < 1.0, f"{m}Error in distance shouldn't be high; less than 1%")
    
    def test_that_eco_cruise_interface_works_for_cycle_average_speed(self):
        for use_rust in [False, True]:
            msg = f"Failed using {'Rust' if use_rust else 'Python'}"
            sd, cyc, _ = self.make_simdrive(use_rust=use_rust)
            expected_idm_target_speed_m_per_s = np.array(cyc.dist_m).sum() / cyc.time_s[-1]
            sd.activate_eco_cruise()
            sd.sim_drive()
            self.do_standard_checks(sd, msg=msg)
            idm_target_speed_m_per_s = np.array(sd.idm_target_speed_m_per_s)[1:].mean()
            self.assertAlmostEqual(expected_idm_target_speed_m_per_s, sd.sim_params.idm_v_desired_m_per_s, msg=msg)
            self.assertAlmostEqual(expected_idm_target_speed_m_per_s, idm_target_speed_m_per_s, msg=msg)

    def test_that_eco_cruise_interface_works_for_microtrip_average_speed(self):
        sd, _, _ = self.make_simdrive()
        sd.activate_eco_cruise(by_microtrip=True)
        sd.sim_drive()
        self.do_standard_checks(sd)

    def test_that_eco_cruise_works_with_blend_factor(self):
        sd, _, _ = self.make_simdrive()
        sd.activate_eco_cruise(by_microtrip=True, blend_factor=1.0)
        sd.sim_drive()
        self.do_standard_checks(sd)

    def test_that_extra_stopped_time_is_added_to_a_cycle_for_eco_cruise(self):
        sd, cyc, _ = self.make_simdrive()
        original_total_time_s = cyc.time_s[-1]
        extend_factor = 0.1
        expected_new_total_time_s = int(round(original_total_time_s * (1 + extend_factor)))
        sd.activate_eco_cruise(by_microtrip=True, extend_fraction=0.1)
        actual_new_total_time_s = sd.cyc0.time_s[-1]
        self.assertEqual(expected_new_total_time_s, actual_new_total_time_s)
