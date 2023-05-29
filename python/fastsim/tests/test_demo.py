"""
Tests the code from fastsim/docs/demo.py
"""
import unittest
import pytest
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fastsim as fsim
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable
if RUST_AVAILABLE:
    import fastsimrust as fsr
else:
    warn_rust_unavailable(__file__)


VERBOSE = False
TOL = 1e-6


def load_cycle(use_rust=False, verbose=False):
    t0 = time.time()
    cyc = fsim.cycle.Cycle.from_file("udds")
    if RUST_AVAILABLE and use_rust:
        cyc = cyc.to_rust()
    t1 = time.time()
    if verbose:
        print(f'Time to load cycle: {t1 - t0:.2e} s')
    return cyc


def load_vehicle(use_rust=False, verbose=False):
    t0 = time.time()
    veh = fsim.vehicle.Vehicle.from_vehdb(11)
    if RUST_AVAILABLE and use_rust:
        veh = veh.to_rust()
    print(f'Time to load vehicle: {time.time() - t0:.2e} s')
    return veh


def run_simdrive(cyc=None, veh=None, use_rust=False, verbose=False):
    if cyc is None:
        cyc = load_cycle(use_rust=use_rust, verbose=verbose)
    if veh is None:
        veh = load_vehicle(use_rust=use_rust, verbose=verbose)
    if RUST_AVAILABLE and use_rust:
        sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
    else:
        sim_drive = fsim.simdrive.SimDrive(cyc, veh)
    t0 = time.time()
    with np.errstate(divide='ignore'):
        sim_drive.sim_drive() 
    dt = time.time() - t0
    if verbose:
        if use_rust:
            print(f'Time to simulate in rust: {dt:.2e} s')
        else:
            print(f'Time to simulate: {dt:.2e} s')
    return sim_drive, dt


def run_by_step_with_varying_aux_loads(use_rust=False, verbose=False):
    t0 = time.time()

    veh = fsim.vehicle.Vehicle.from_vehdb(9)
    cyc = fsim.cycle.Cycle.from_file('udds')
    if RUST_AVAILABLE and use_rust:
        cyc = cyc.to_rust()
        veh = veh.to_rust()
        sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
    else:
        sim_drive = fsim.simdrive.SimDrive(cyc, veh)

    sim_drive.init_for_step(init_soc=0.7935)

    while sim_drive.i < len(cyc.time_s):
        # note: we need to copy out and in the entire array to work with the Rust version
        # that is, we can't set just a specific element of an array in rust via python bindings at this time
        aux_in_kw = np.array(sim_drive.aux_in_kw)
        aux_in_kw[sim_drive.i] = sim_drive.i / cyc.time_s[-1] * 10 
        sim_drive.aux_in_kw = aux_in_kw
        # above could be a function of some internal sim_drive state
        sim_drive.sim_drive_step()

    if verbose:
        print(f'Time to simulate: {time.time() - t0:.2e} s')

    return sim_drive


def run_with_aux_overrides_in_simdrive(use_rust=False):
    veh = fsim.vehicle.Vehicle.from_vehdb(9)
    cyc = fsim.cycle.Cycle.from_file('udds')
    if RUST_AVAILABLE and use_rust:
        veh = veh.to_rust()
        cyc = cyc.to_rust()
        sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
    else:
        sim_drive = fsim.simdrive.SimDrive(cyc, veh)
    auxInKwConst = 12
    sim_drive.sim_drive(None, np.ones(len(cyc.time_s))*auxInKwConst)
    return sim_drive


def run_with_aux_override_direct_set(use_rust=False, verbose=False):
    t0 = time.time()
    veh = fsim.vehicle.Vehicle.from_vehdb(9)
    cyc = fsim.cycle.Cycle.from_file('udds')
    if RUST_AVAILABLE and use_rust:
        veh = veh.to_rust()
        cyc = cyc.to_rust()
        sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
    else:
        sim_drive = fsim.simdrive.SimDrive(cyc, veh)
    # by assigning the value directly (this is faster than using positional args)
    sim_drive.init_for_step(
        init_soc=veh.max_soc,
        aux_in_kw_override=np.array(cyc.time_s) / cyc.time_s[-1] * 10
    )
    while sim_drive.i < len(sim_drive.cyc.time_s):
        sim_drive.sim_drive_step()
    if verbose:
        print(f'Time to simulate: {time.time() - t0:.2e} s')
    return sim_drive


def use_simdrive_post(use_rust=False, verbose=False):
    t0 = time.time()
    veh = fsim.vehicle.Vehicle.from_vehdb(19)
    # veh = veh
    if verbose:
        print(f'Time to load vehicle: {time.time() - t0:.2e} s')
    # generate micro-trip 
    t0 = time.time()
    cyc = fsim.cycle.Cycle.from_file("udds")
    microtrips = fsim.cycle.to_microtrips(cyc.get_cyc_dict())
    cyc = fsim.cycle.Cycle.from_dict(microtrips[1])
    if verbose:
        print(f'Time to load cycle: {time.time() - t0:.2e} s')

    t0 = time.time()
    if RUST_AVAILABLE and use_rust:
        veh = veh.to_rust()
        cyc = cyc.to_rust()
        sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
    else:
        sim_drive = fsim.simdrive.SimDrive(cyc, veh)
    sim_drive.sim_drive()
    # sim_drive = fsim.simdrive.SimDriveClassic(cyc, veh)
    # sim_drive.sim_drive()
    if verbose:
        print(f'Time to simulate: {time.time() - t0:.2e} s')

    t0 = time.time()
    sim_drive_post = fsim.simdrive.SimDrivePost(sim_drive)
    sim_drive_post.set_battery_wear()
    diag = sim_drive_post.get_diagnostics()
    if verbose:
        print(f'Time to post process: {time.time() - t0:.2e} s')
    return diag


class TestDemo(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()

    def test_load_cycle(self):
        for use_rust in [False, True]:
            if use_rust and not RUST_AVAILABLE:
                continue
            try:
                c = load_cycle(use_rust=use_rust, verbose=VERBOSE)
                if use_rust:
                    self.assertEqual(type(c), fsr.RustCycle)
                else:
                    self.assertEqual(type(c), fsim.cycle.Cycle)
            except Exception as ex:
                self.assertTrue(
                    False,
                    msg=f"Exception (Rust: {use_rust}): {ex}"
                )

    def test_load_vehicle(self):
        for use_rust in [False, True]:
            if use_rust and not RUST_AVAILABLE:
                continue
            try:
                v = load_vehicle(use_rust=use_rust, verbose=VERBOSE)
                if use_rust:
                    self.assertEqual(type(v), fsr.RustVehicle)
                else:
                    self.assertEqual(type(v), fsim.vehicle.Vehicle)
            except Exception as ex:
                self.assertTrue(
                    False,
                    msg=f"Exception (Rust: {use_rust}): {ex}"
                )
    @pytest.mark.filterwarnings("ignore:.*divide by zero*.")
    def test_run_simdrive(self):
        if not RUST_AVAILABLE:
            return
        py_cyc = load_cycle(use_rust=False)
        py_veh = load_vehicle(use_rust=False)
        ru_cyc = load_cycle(use_rust=True)
        ru_veh = load_vehicle(use_rust=True)
        try:
            py_sd, py_dt = run_simdrive(py_cyc, py_veh, use_rust=False, verbose=VERBOSE)
            self.assertEqual(type(py_sd), fsim.simdrive.SimDrive)
        except Exception as ex:
            self.assertTrue(
                False,
                msg=f"Exception (Rust: False): {ex}"
            )
        try:
            ru_sd, ru_dt = run_simdrive(ru_cyc, ru_veh, use_rust=True, verbose=VERBOSE)
            self.assertEqual(type(ru_sd), fsr.RustSimDrive)
        except Exception as ex:
            self.assertTrue(
                False,
                msg=f"Exception (Rust: False): {ex}"
            )
        speedup = py_dt / ru_dt
        if VERBOSE:
            print(f"Rust provides a {speedup:.5g}x speedup")
        self.assertTrue(
            speedup > 10.0,
            msg=f"Expected a speedup greater than 10, got {speedup}"
        )
        fc_kw_in_ach_max_abs_diff = np.abs(
            np.array(ru_sd.fc_kw_in_ach)
            - np.array(py_sd.fc_kw_in_ach)
        ).max()
        self.assertTrue(fc_kw_in_ach_max_abs_diff < TOL)

    def test_running_by_step_with_modified_aux_loads(self):
        if not RUST_AVAILABLE:
            return
        py_sd = run_by_step_with_varying_aux_loads(
            use_rust=False, verbose=VERBOSE)
        ru_sd = run_by_step_with_varying_aux_loads(
            use_rust=True, verbose=VERBOSE)
        fc_out_max_abs_diff = np.abs(
            py_sd.fc_kw_out_ach
            - np.array(ru_sd.fc_kw_out_ach)
        ).max()
        ess_out_max_abs_diff = np.abs(
            py_sd.ess_kw_out_ach
            - np.array(ru_sd.ess_kw_out_ach)
        ).max()
        self.assertTrue(fc_out_max_abs_diff < TOL)
        self.assertTrue(ess_out_max_abs_diff < TOL)

    def test_running_with_aux_overrides(self):
        if not RUST_AVAILABLE:
            return
        py_sd = run_with_aux_overrides_in_simdrive(use_rust=False)
        ru_sd = run_with_aux_overrides_in_simdrive(use_rust=True)
        fc_out_max_abs_diff = np.abs(
            py_sd.fc_kw_out_ach
            - np.array(ru_sd.fc_kw_out_ach)
        ).max()
        ess_out_max_abs_diff = np.abs(
            py_sd.ess_kw_out_ach
            - np.array(ru_sd.ess_kw_out_ach)
        ).max()
        self.assertTrue(fc_out_max_abs_diff < TOL)
        self.assertTrue(ess_out_max_abs_diff < TOL)

    def test_running_with_aux_overrides_v2(self):
        if not RUST_AVAILABLE:
            return
        py_sd = run_with_aux_override_direct_set(use_rust=False, verbose=VERBOSE)
        ru_sd = run_with_aux_override_direct_set(use_rust=True, verbose=VERBOSE)
        fc_out_max_abs_diff = np.abs(
            py_sd.fc_kw_out_ach
            - np.array(ru_sd.fc_kw_out_ach)
        ).max()
        ess_out_max_abs_diff = np.abs(
            py_sd.ess_kw_out_ach
            - np.array(ru_sd.ess_kw_out_ach)
        ).max()
        self.assertTrue(fc_out_max_abs_diff < TOL)
        self.assertTrue(ess_out_max_abs_diff < TOL)

    def test_using_simdrive_post(self):
        if not RUST_AVAILABLE:
            return
        py_diag = use_simdrive_post(use_rust=False, verbose=VERBOSE)
        ru_diag = use_simdrive_post(use_rust=True, verbose=VERBOSE)
        py_diag_keys = {k for k in py_diag}
        ru_diag_keys = {k for k in ru_diag}
        self.assertEqual(py_diag_keys, ru_diag_keys)

    def test_cycle_to_dict(self):
        if not RUST_AVAILABLE:
            return
        py_cyc = fsim.cycle.Cycle.from_file('udds')
        ru_cyc = py_cyc.to_rust()
        py_dict = py_cyc.get_cyc_dict()
        ru_dict = ru_cyc.get_cyc_dict()
        py_keys = {k for k in py_dict}
        ru_keys = {k for k in ru_dict}
        ru_keys.add("name") # Rust doesn't provide 'name'
        self.assertEqual(py_keys, ru_keys)

if __name__ == '__main__':
    unittest.main()
