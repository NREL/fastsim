"""
Tests using the Rust versions of SimDrive, Cycle, and Vehicle
"""
import unittest

import numpy as np

import fastsim as fsim
import fastsim.vehicle_base as fsvb
from fastsim import cycle, vehicle, simdrive
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable

if RUST_AVAILABLE:
    import fastsimrust as fsr
else:
    warn_rust_unavailable(__file__)

TEST_VARS = [
    "aux_in_kw",
    "max_trac_mps",
    "cur_max_fs_kw_out",
    "cur_max_trans_kw_out",
    "newton_iters",
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
    "cur_ess_max_kw_out",
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


class TestRust(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()

    def test_run_sim_drive_conv(self):
        if not RUST_AVAILABLE:
            return
        cyc = cycle.Cycle.from_file('udds').to_rust()
        veh = vehicle.Vehicle.from_vehdb(5).to_rust()
        # sd = simdrive.SimDrive(cyc, veh).to_rust()
        sd = fsr.RustSimDrive(cyc, veh)
        sd.sim_drive()
        self.assertTrue(sd.i > 1)
        self.assertEqual(sd.i, len(cyc.time_s))

    def test_run_sim_drive_hev(self):
        if not RUST_AVAILABLE:
            return
        cyc = cycle.Cycle.from_file('udds').to_rust()
        veh = vehicle.Vehicle.from_vehdb(11).to_rust()
        sd = fsr.RustSimDrive(cyc, veh)
        sd.sim_drive()
        self.assertTrue(sd.i > 1)
        self.assertEqual(sd.i, len(cyc.time_s))

    def test_discrepancies(self, veh_type="ALL", use_dict=True, cyc_name="udds"):
        """
        Function for testing for Rust/Python discrepancies, both in the vehicle database
        CSV as well as the individual model files. Uses test_vehicle_for_discrepancies as backend.
        Arguments:
            veh_type: type of vehicle to test for discrepancies
                can be "CONV", "HEV", "PHEV", "BEV", or "ALL"
            use_dict: if True, use small cyc_dict to speed up test
                if false, default to UDDS
            cyc_name: name of cycle from database to use if use_dict == False
        """
        if not RUST_AVAILABLE:
            raise Exception("Rust unavailable.")
        veh_types = ["CONV", "HEV", "PHEV", "BEV", "ALL"]
        if veh_type not in veh_types:
            raise ValueError(f'veh_type "{veh_type}" not in {veh_types}.')

        cyc_dict = {
            "cycSecs": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            "cycMps":  np.array([0.0, 0.4, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.4, 0.0, 0.0]),
            "cycGrade": np.array([0.0] * 11),
        } if use_dict else None

        kwargs = {
            "cyc_dict": cyc_dict,
            "cyc_name": cyc_name,
        }
        if veh_type in ["CONV", "ALL"]:
            self.test_vehicle_for_discrepancies(vnum=4, **kwargs)
            self.test_vehicle_for_discrepancies(vnum=7, **kwargs)
            self.test_vehicle_for_discrepancies(
                veh_filename="2016_EU_VW_Golf_1.4TSI.csv", **kwargs)
        if veh_type in ["HEV", "ALL"]:
            self.test_vehicle_for_discrepancies(vnum=9, **kwargs)
            self.test_vehicle_for_discrepancies(vnum=10, **kwargs)
            self.test_vehicle_for_discrepancies(
                veh_filename="2022_TOYOTA_Yaris_Hybrid_Mid.csv", **kwargs)
        if veh_type in ["PHEV", "ALL"]:
            self.test_vehicle_for_discrepancies(vnum=13, **kwargs)
            self.test_vehicle_for_discrepancies(vnum=16, **kwargs)
        if veh_type in ["BEV", "ALL"]:
            self.test_vehicle_for_discrepancies(vnum=17, **kwargs)
            self.test_vehicle_for_discrepancies(vnum=20, **kwargs)

    def test_vehicle_for_discrepancies(self, vnum=1, veh_filename=None, cyc_dict=None, cyc_name="udds"):
        """
        Test for finding discrepancies between Rust and Python for single vehicle.
        Arguments:
            vnum: vehicle database number, optional, default option without any arguments
            veh_filename: vehicle filename from vehdb folder, optional
            cyc_dict: cycle dictionary for custom cycle, optional
            cyc_name: cycle name from cycle database, optional
        """
        if not RUST_AVAILABLE:
            raise Exception("Rust unavailable.")
        # Load cycle
        if cyc_dict is not None:
            cyc_python = cycle.Cycle.from_dict(cyc_dict)
        else:
            cyc_python = cycle.Cycle.from_file(cyc_name)
        cyc_rust = cyc_python.to_rust()
        # Load selected (or default) vehicle
        if veh_filename is None:
            veh_python = vehicle.Vehicle.from_vehdb(vnum)
        else:
            veh_python = vehicle.Vehicle.from_file(veh_filename)
        veh_name = veh_python.scenario_name
        veh_rust = veh_python.to_rust()
        # Instantiate SimDrive objects
        sim_python = simdrive.SimDrive(cyc_python, veh_python)
        sim_rust = fsr.RustSimDrive(cyc_rust, veh_rust)
        # Check for discrepancies before simulation
        printed_vehicle = False
        places = 6
        tol = 10 ** (-1 * places)
        self.assertEqual(sim_python.props.air_density_kg_per_m3,
                         sim_rust.props.air_density_kg_per_m3)
        self.assertEqual(sim_python.sim_params.newton_max_iter,
                         sim_rust.sim_params.newton_max_iter)
        self.assertEqual(sim_python.sim_params.newton_gain,
                         sim_rust.sim_params.newton_gain)
        self.assertEqual(sim_python.sim_params.newton_xtol,
                         sim_rust.sim_params.newton_xtol)
        self.assertAlmostEqual(sim_python.veh.drag_coef,
                               sim_rust.veh.drag_coef, places=places)
        self.assertAlmostEqual(
            sim_python.veh.frontal_area_m2, sim_rust.veh.frontal_area_m2, places=places)
        self.assertAlmostEqual(
            sim_python.veh.mc_max_elec_in_kw, sim_rust.veh.mc_max_elec_in_kw, places=places)
        self.assertAlmostEqual(sim_python.veh.ess_max_kwh,
                               sim_rust.veh.ess_max_kwh, places=places)
        self.assertAlmostEqual(sim_python.veh.ess_round_trip_eff,
                               sim_rust.veh.ess_round_trip_eff, places=places)
        # Simulate
        sim_python.sim_drive()
        sim_rust.sim_drive()
        # Check for discrepancies after simulation
        py = {}
        ru = {}
        ru_cyc_mps = np.array(sim_rust.cyc.mps)
        ru_cyc_dt_s = np.array(sim_rust.cyc.dt_s)
        self.assertTrue((np.abs(sim_python.cyc.mps - ru_cyc_mps) < tol).all())
        self.assertTrue(
            (np.abs(sim_python.cyc.dt_s - ru_cyc_dt_s) < tol).all())
        ru_sd_mps_ach = np.array(sim_rust.mps_ach)
        self.assertTrue((sim_python.mps_ach >= 0.0).all(),
                        msg=f'PYTHON: Detected negative speed for {veh_name}')
        self.assertTrue((ru_sd_mps_ach >= 0.0).all(),
                        msg=f'RUST  : Detected negative speed for {veh_name}')
        for v in TEST_VARS:
            py[v] = sim_python.__getattribute__(v)
            ru[v] = sim_rust.__getattribute__(v)
        discrepancy = False
        for i in range(1, len(cyc_python.time_s)):
            for v in TEST_VARS:
                if type(py[v][i]) is bool or type(py[v][i]) is np.bool_:
                    if py[v][i] != ru[v][i]:
                        discrepancy = True
                        if not printed_vehicle:
                            printed_vehicle = True
                            print(f'DISCREPANCY FOR VEHICLE {veh_name}')
                        print(
                            f"BOOL: {v} differs for {i}: py = {py[v][i]}; ru = {ru[v][i]}")
                else:
                    if abs(py[v][i] - ru[v][i]) > 1e-6:
                        discrepancy = True
                        if not printed_vehicle:
                            printed_vehicle = True
                            print(f'DISCREPANCY FOR VEHICLE {veh_name}')
                        print(
                            f"REAL: {v} differs for {i}: py = {py[v][i]}; ru = {ru[v][i]}")
        self.assertFalse(discrepancy, "Discrepancy detected")

    def test_step_by_step(self):
        if not RUST_AVAILABLE:
            return
        use_dict = False
        cyc_dict = {
            'cycSecs': np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            'cycMps':  np.array([0.0, 0.4, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.4, 0.0, 0.0]),
            'cycGrade': np.array([0.0] * 11),
        }
        cyc_name = "udds"
        discrepancy = False
        for vehid in [1, 9, 14, 17, 24]:
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
            ru_cyc = py_cyc.to_rust()
            ru_veh = py_veh.to_rust()
            ru_sd = fsr.RustSimDrive(ru_cyc, ru_veh)
            places = 6
            tol = 10 ** (-1 * places)
            self.assertEqual(py_sd.props.air_density_kg_per_m3,
                             ru_sd.props.air_density_kg_per_m3)
            self.assertEqual(py_sd.sim_params.newton_max_iter,
                             ru_sd.sim_params.newton_max_iter)
            self.assertEqual(py_sd.sim_params.newton_gain,
                             ru_sd.sim_params.newton_gain)
            self.assertEqual(py_sd.sim_params.newton_xtol,
                             ru_sd.sim_params.newton_xtol)
            self.assertAlmostEqual(py_sd.veh.drag_coef,
                                   ru_sd.veh.drag_coef, places=places)
            self.assertAlmostEqual(
                py_sd.veh.frontal_area_m2, ru_sd.veh.frontal_area_m2, places=places)
            self.assertAlmostEqual(
                py_sd.veh.mc_max_elec_in_kw, ru_sd.veh.mc_max_elec_in_kw, places=places)
            self.assertAlmostEqual(py_sd.veh.ess_max_kwh,
                                   ru_sd.veh.ess_max_kwh, places=places)
            self.assertAlmostEqual(
                py_sd.veh.ess_round_trip_eff, ru_sd.veh.ess_round_trip_eff, places=places)
            py_sd.sim_drive()
            ru_sd.sim_drive()
            py = {}
            ru = {}
            ru_cyc_mps = np.array(ru_sd.cyc.mps)
            ru_cyc_dt_s = np.array(ru_sd.cyc.dt_s)
            self.assertTrue((np.abs(py_sd.cyc.mps - ru_cyc_mps) < tol).all())
            self.assertTrue((np.abs(py_sd.cyc.dt_s - ru_cyc_dt_s) < tol).all())
            ru_sd_mps_ach = np.array(ru_sd.mps_ach)
            self.assertTrue(
                (py_sd.mps_ach >= 0.0).all(),
                msg=f'PYTHON: Detected negative speed for {vehid}')
            self.assertTrue(
                (ru_sd_mps_ach >= 0.0).all(),
                msg=f'RUST  : Detected negative speed for {vehid}')
            for v in TEST_VARS:
                py[v] = py_sd.__getattribute__(v)
                ru[v] = ru_sd.__getattribute__(v)
            for i in range(1, N):
                for v in TEST_VARS:
                    if type(py[v][i]) is bool or type(py[v][i]) is np.bool_:
                        if py[v][i] != ru[v][i]:
                            discrepancy = True
                            if not printed_vehicle:
                                printed_vehicle = True
                                print(f'DISCREPANCY FOR VEHICLE {vehid}')
                            print(
                                f"BOOL: {v} differs for {i}: py = {py[v][i]}; ru = {ru[v][i]}")
                    else:
                        if abs(py[v][i] - ru[v][i]) > 1e-6:
                            discrepancy = True
                            if not printed_vehicle:
                                printed_vehicle = True
                                print(f'DISCREPANCY FOR VEHICLE {vehid}')
                            print(
                                f"REAL: {v} differs for {i}: py = {py[v][i]}; ru = {ru[v][i]}")
        self.assertFalse(discrepancy, "Discrepancy detected")

    def test_fueling_prediction_for_multiple_vehicle(self):
        """
        This test assures that Rust and Python agree on at least one 
        example of all permutations of veh_pt_type and fc_eff_type.
        """
        if not RUST_AVAILABLE:
            return
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
            self.assertAlmostEqual(
                py_fuel_kj, rust_fuel_kj, msg=f'Non-agreement for vehicle {vehid} for fuel')
            self.assertAlmostEqual(py_ess_dischg_kj, rust_ess_dischg_kj,
                                   msg=f'Non-agreement for vehicle {vehid} for ess discharge')

    def test_achieved_speed_never_negative(self):
        if not RUST_AVAILABLE:
            return
        for vehid in range(1, 27):
            veh = vehicle.Vehicle.from_vehdb(vehid).to_rust()
            cyc = cycle.Cycle.from_file('udds').to_rust()
            sd = fsr.RustSimDrive(cyc, veh)
            sd.sim_drive()
            sd_mps_ach = np.array(sd.mps_ach)
            sd_cyc0_mps = np.array(sd.cyc0.mps)
            self.assertFalse(
                (sd_mps_ach < 0.0).any(),
                msg=f'Achieved speed contains negative values for vehicle {vehid}'
            )
            self.assertFalse(
                (sd_mps_ach > sd_cyc0_mps).any(),
                msg=f'Achieved speed is greater than requested speed for {vehid}'
            )

    def test_grade(self):
        if not RUST_AVAILABLE:
            raise Exception("Rust unavailable.")
        cyc = cycle.Cycle.from_file("udds")
        # Manually enter grade
        amplitude = 0.05
        period = 500  # seconds
        cyc.grade = amplitude * np.sin((2*np.pi/period) * cyc.time_s)
        cyc_dict = cyc.get_cyc_dict()

        self.test_vehicle_for_discrepancies(cyc_dict=cyc_dict)

    def test_serde_json(self):
        cyc = cycle.Cycle.from_file("udds").to_rust()
        veh = vehicle.Vehicle.from_vehdb(10).to_rust()
        sdr = simdrive.RustSimDrive(cyc, veh)
        _ = fsr.RustCycle.from_json(cyc.to_json())
        _ = fsr.RustPhysicalProperties.from_json(sdr.props.to_json())
        # _ = fastsimrust.RustSimDrive.from_json(sdr.to_json()) # this probably fails because vehicle fails
        _ = fsr.RustSimDriveParams.from_json(sdr.sim_params.to_json())
        # _ = fastsimrust.RustVehicle.from_json(veh.to_json()) # TODO: figure out why this fails


if __name__ == '__main__':
    unittest.main()
