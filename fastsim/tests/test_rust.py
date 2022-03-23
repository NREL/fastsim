"""
Tests using the Rust versions of SimDrive, Cycle, and Vehicle
"""
import unittest

import numpy as np

import fastsim.vehicle_base as fsvb
from fastsim import cycle, vehicle, simdrive
import fastsim.fastsimrust as fsr


class TestRust(unittest.TestCase):
    def test_run_sim_drive_conv(self):
        cyc = cycle.Cycle.from_file('udds').to_rust()
        veh = vehicle.Vehicle.from_vehdb(5).to_rust()
        #sd = simdrive.SimDrive(cyc, veh).to_rust()
        sd = fsr.RustSimDrive(cyc, veh)
        sd.sim_drive_walk(0.5)
        self.assertTrue(sd.i > 1)
        self.assertEqual(sd.i, len(cyc.time_s))

    def test_run_sim_drive_conv(self):
        cyc = cycle.Cycle.from_file('udds').to_rust()
        veh = vehicle.Vehicle.from_vehdb(11).to_rust()
        sd = fsr.RustSimDrive(cyc, veh)
        sd.sim_drive_walk(0.5)
        self.assertTrue(sd.i > 1)
        self.assertEqual(sd.i, len(cyc.time_s))
    
    def test_step_by_step(self):
        use_dict = False
        cyc_dict = {
            'cycSecs': np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            'cycMps':  np.array([0.0, 0.4, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.4, 0.0, 0.0]),
            'cycGrade': np.array([0.0] * 11),
        }
        cyc_name = 'udds'
        vars = [
            "aux_in_kw",
            "max_trac_mps",
            "cur_max_fs_kw_out",
            "cur_max_trans_kw_out",
            "newton_iters",
            "cyc_drag_kw",
            "min_mc_kw_2help_fc",
            "mps_ach",
            "dist_mi",
            "regen_buff_soc",
            "er_ae_kw_out",
            "fc_forced_on",
            "fc_forced_state",
            "ess_desired_kw_4fc_eff",
            "accel_buff_soc",
            "regen_buff_soc",
            "cur_max_ess_kw_out",
            "cur_max_mc_elec_kw_in",
            "cur_max_ess_chg_kw",
            "ess_cur_kwh",
            "soc",
            "ess_accel_regen_dischg_kw",
            "fc_kw_gap_fr_eff",
            "ess_regen_buff_dischg_kw",
            "mc_elec_in_lim_kw",
            "ess_kw_if_fc_req",
            "mc_elec_kw_in_if_fc_req",
            "mc_kw_if_fc_req",
            "mc_mech_kw_out_ach",
            "mc_elec_kw_in_ach",
            "cur_max_fc_kw_out",
            "ess_kw_out_ach",
            "fc_time_on",
            "fc_kw_out_ach",
            "fs_kwh_out_ach",
        ]
        has_any_errors = False
        for vehid in [1, 9, 14, 17, 24]:
            has_errors = False
            printed_vehicle = False
            if use_dict:
                py_cyc = cycle.Cycle.from_dict(cyc_dict)
                self.assertEqual(len(py_cyc.time_s), 11)
            else:
                py_cyc = cycle.Cycle.from_file(cyc_name)
                self.assertEqual(len(py_cyc.time_s), 1370)
            N = len(py_cyc.time_s)
            py_veh = vehicle.Vehicle.from_vehdb(vehid)
            py_sd = simdrive.SimDrive(py_cyc, py_veh)
            if use_dict:
                ru_cyc = cycle.Cycle.from_dict(cyc_dict).to_rust()
            else:
                ru_cyc = cycle.Cycle.from_file(cyc_name).to_rust()
            ru_veh = vehicle.Vehicle.from_vehdb(vehid).to_rust()
            ru_sd = fsr.RustSimDrive(ru_cyc, ru_veh)
            self.assertEqual(py_sd.props.air_density_kg_per_m3, ru_sd.props.air_density_kg_per_m3)
            self.assertEqual(py_sd.sim_params.newton_max_iter, ru_sd.sim_params.newton_max_iter)
            self.assertEqual(py_sd.sim_params.newton_gain, ru_sd.sim_params.newton_gain)
            self.assertEqual(py_sd.sim_params.newton_xtol, ru_sd.sim_params.newton_xtol)
            self.assertAlmostEqual(py_sd.veh.drag_coef, ru_sd.veh.drag_coef)
            self.assertAlmostEqual(py_sd.veh.frontal_area_m2, ru_sd.veh.frontal_area_m2)
            self.assertAlmostEqual(py_sd.veh.mc_max_elec_in_kw, ru_sd.veh.mc_max_elec_in_kw)
            self.assertAlmostEqual(py_sd.veh.max_ess_kwh, ru_sd.veh.max_ess_kwh)
            self.assertAlmostEqual(py_sd.veh.ess_round_trip_eff, ru_sd.veh.ess_round_trip_eff)
            for i in range(1, N):
                py = {}
                ru = {}
                py_sd.sim_drive_step()
                ru_sd.sim_drive_step()
                py_cyc_drag_kw = py_sd.cyc_drag_kw
                ru_cyc_drag_kw = ru_sd.cyc_drag_kw
                ru_cyc_mps = np.array(ru_sd.cyc.mps)
                ru_cyc_dt_s = np.array(ru_sd.cyc.dt_s)
                self.assertAlmostEqual(py_sd.cyc.mps[i], ru_cyc_mps[i])
                self.assertAlmostEqual(py_sd.cyc.dt_s[i], ru_cyc_dt_s[i])
                for v in vars:
                    py[v] = py_sd.__getattribute__(v)
                    ru[v] = ru_sd.__getattribute__(v)
                    if v == "cyc_drag_kw":
                        self.assertEqual(py[v][i], py_cyc_drag_kw[i])
                        self.assertEqual(ru[v][i], ru_cyc_drag_kw[i])
                    if type(py[v][i]) is bool or type(py[v][i]) is np.bool_:
                        if py[v][i] != ru[v][i]:
                            has_errors = True
                            if not printed_vehicle:
                                printed_vehicle = True
                                print(f'DISCREPANCY FOR VEHICLE {vehid}')
                            print(f"BOOL: {v} differs for {i}: py = {py[v][i]}; ru = {ru[v][i]}")
                    else:
                        if abs(py[v][i] - ru[v][i]) > 1e-6:
                            has_errors = True
                            if not printed_vehicle:
                                printed_vehicle = True
                                print(f'DISCREPANCY FOR VEHICLE {vehid}')
                            print(f"REAL: {v} differs for {i}: py = {py[v][i]}; ru = {ru[v][i]}")
                if has_errors:
                    has_any_errors = True
                    break
        self.assertFalse(has_any_errors)
    
    def test_fueling_prediction_for_multiple_vehicle(self):
        for vehid in [1, 9, 14, 17, 24]:
            cyc = cycle.Cycle.from_file('udds')
            veh = vehicle.Vehicle.from_vehdb(vehid)
            sd = simdrive.SimDrive(cyc, veh)
            sd.sim_drive_walk(0.5)
            sd.set_post_scalars()
            py_fuel_kj = sd.fuel_kj
            py_ess_dischg_kj = sd.ess_dischg_kj
            cyc = cycle.Cycle.from_file('udds').to_rust()
            veh = vehicle.Vehicle.from_vehdb(vehid).to_rust()
            sd = fsr.RustSimDrive(cyc, veh)
            sd.sim_drive_walk(0.5)
            sd.set_post_scalars()
            rust_fuel_kj = sd.fuel_kj
            rust_ess_dischg_kj = sd.ess_dischg_kj
            self.assertAlmostEqual(py_fuel_kj, rust_fuel_kj, msg=f'Non-agreement for vehicle {vehid} for fuel')
            self.assertAlmostEqual(py_ess_dischg_kj, rust_ess_dischg_kj, msg=f'Non-agreement for vehicle {vehid} for ess discharge')
