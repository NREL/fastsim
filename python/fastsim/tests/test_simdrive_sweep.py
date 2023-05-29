"""Test script that saves results from 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles. 
From command line, pass True (default if left blank) or False argument to use JIT compilation or not, respectively."""

import pandas as pd
import time
import numpy as np
from pathlib import Path
import unittest
from fastsim.auxiliaries import set_nested_values

# local modules
from fastsim import simdrive, vehicle, cycle, utils
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable

if RUST_AVAILABLE:
    import fastsimrust as fsr

RUN_PYTHON = False
RUN_RUST = True

if RUN_RUST and not RUST_AVAILABLE:
    warn_rust_unavailable(__file__)


def main(err_tol=1e-4, verbose=True, use_rust=False):
    """Runs test test for 26 vehicles and 3 cycles.  
    Test compares cumulative positive and negative energy 
    values to a benchmark from earlier.

    Arguments:
    ----------
    err_tol : error tolerance
        default of 1e-4 was selected to prevent minor errors from showing.  
        As of 31 December 2020, a recent python update caused errors that 
        are smaller than this and therefore ok to neglect.
    verbose: if True, prints progress
    use_rust: Boolean, if True, use Rust version of classes, else python version

    Returns:
    --------
    df_err : pandas datafram, fractional errors
    df : pandas dataframe, new values
    df0 : pandas dataframe, original benchmark values
    col_for_max_error: string or None, the column name of the column having max absolute error
    max_abs_err: number or None, the maximum absolute error if it exists
    """
    if not RUST_AVAILABLE and use_rust:
        warn_rust_unavailable(__file__)
        use_rust = False
    t0 = time.time()

    print('Running vehicle sweep.\n')

    def to_rust(obj):
        if use_rust:
            return obj.to_rust()
        return obj

    def make_simdrive(*args, **kwargs):
        if use_rust:
            return simdrive.RustSimDrive(*args, **kwargs)
        return simdrive.SimDrive(*args, **kwargs)

    cyc_names = ['udds', 'hwfet', 'us06']
    cycs = {
        cyc_name: to_rust(cycle.Cycle.from_file(cyc_name)) for cyc_name in cyc_names
    }

    vehnos = np.arange(1, 27)

    veh = to_rust(vehicle.Vehicle.from_vehdb(1))
    energyAuditErrors = []

    dict_diag = {}
    t0a = 0
    iter = 0
    for vehno in vehnos:
        if verbose:
            print('vehno =', vehno)
        if vehno == 2:
            t0a = time.time()
        for cyc_name, cyc in cycs.items():
            if not(vehno == 1):
                veh = to_rust(vehicle.Vehicle.from_vehdb(vehno))
            if RUST_AVAILABLE and use_rust:
                assert type(cyc) == fsr.RustCycle
                assert type(veh) == fsr.RustVehicle
            sim_drive = make_simdrive(cyc, veh)
            if RUST_AVAILABLE and use_rust:
                assert type(sim_drive) == fsr.RustSimDrive
            # US06 is known to cause substantial trace miss.
            # This should probably be addressed at some point, but for now,
            # the tolerances are set high to avoid lots of printed warnings.
            sim_drive.sim_params = set_nested_values(sim_drive.sim_params)
            sim_drive.sim_drive()

            sim_drive_post = simdrive.SimDrivePost(sim_drive)
            # sim_drive_post.set_battery_wear()
            diagno = sim_drive_post.get_diagnostics()
            energyAuditErrors.append(sim_drive.energy_audit_error)

            if iter == 0:
                dict_diag['vnum'] = [vehno]
                dict_diag['Scenario_name'] = [veh.scenario_name]
                dict_diag['cycle'] = [cyc_name]
                for key in diagno.keys():
                    dict_diag[key] = [diagno[key]]
                iter += 1

            else:
                dict_diag['vnum'].append(vehno)
                dict_diag['Scenario_name'].append(veh.scenario_name)
                dict_diag['cycle'].append(cyc_name)
                for key in diagno.keys():
                    dict_diag[key].append(diagno[key])

    df = pd.DataFrame.from_dict(dict_diag)

    t1 = time.time()
    print()
    print('Elapsed time: {:.2f} s'.format(t1 - t0))
    print('Elapsed time since first vehicle: {:.2f} s'.format(t1 - t0a, 2))

    # NOTE: cyc_wheel_* variables are being missed as they are called cyc_whl_* in SimDrive
    df0 = pd.read_csv(Path(simdrive.__file__).parent.resolve() /
                      'resources/master_benchmark_vars.csv')
    df0 = df0.rename(columns=utils.camel_to_snake)

    # make sure new dataframe does not incude newly added or deprecated columns
    new_cols = {col for col in df.columns} - {col for col in df0.columns}

    from math import isclose

    df_err = df.copy().drop(columns=list(new_cols))
    abs_err = []
    col_for_max_error = None
    max_abs_err = None
    for idx in df.index:
        for col in df_err.columns[2:]:
            if not(isclose(df.loc[idx, col], df0.loc[idx, col], rel_tol=err_tol, abs_tol=err_tol)):
                df_err.loc[idx, col] = (
                    df.loc[idx, col] - df0.loc[idx, col]) / df0.loc[idx, col]
                if max_abs_err is None or np.abs(df_err.loc[idx, col]) > max_abs_err:
                    max_abs_err = np.abs(df_err.loc[idx, col])
                    col_for_max_error = col
                abs_err.append(np.abs(df_err.loc[idx, col]))
                print(f"{df_err.loc[idx, col]:.5%} error for {col}")
                print(
                    f"vehicle  : {vehicle.DEFAULT_VEHDF[vehicle.DEFAULT_VEHDF['Selection'] == df.loc[idx, 'vnum']]['Scenario name'].values[0]}")
                print(f"cycle    : {df.loc[idx, 'cycle']}")
                print('New Value: ' + str(round(df.loc[idx, col], 15)))
                print('Old Value: ' + str(round(df0.loc[idx, col], 15)))
                print('Index    : ' + str(idx))
                print()
            else:
                df_err.loc[idx, col] = 0

    abs_err = np.array(abs_err)
    if len(abs_err) > 0:
        print(f'\nmax error = {abs_err.max():.3%}')
    else:
        print(f'No errors exceed the {err_tol:.3g} tolerance threshold.')

    return df_err, df, df0, col_for_max_error, max_abs_err


class TestSimDriveSweep(unittest.TestCase):
    def setUp(self):
        utils.disable_logging()

    def test_sweep(self):
        "Compares results against benchmark."
        print(f"Running {type(self)}.")
        if RUN_PYTHON:
            df_err, _, _, max_err_col, max_abs_err = main(verbose=True)
            self.assertEqual(df_err.iloc[:, 2:].max().max(), 0,
                             msg=f"Failed for Python version; {max_err_col} had max abs error of {max_abs_err}")
        if RUST_AVAILABLE and RUN_RUST:
            df_err, _, _, max_err_col, max_abs_err = main(
                verbose=True, use_rust=True)
            self.assertEqual(df_err.iloc[:, 2:].max().max(), 0,
                             msg=f"Failed for Rust version; {max_err_col} had max abs error of {max_abs_err}")

    def test_post_diagnostics(self):
        if not RUST_AVAILABLE:
            return
        vehid = 9  # FORD C-MAX
        cyc_name = "us06"
        init_soc = None
        # PYTHON
        cyc = cycle.Cycle.from_file(cyc_name)
        veh = vehicle.Vehicle.from_vehdb(vehid)
        py_sd = simdrive.SimDrive(cyc, veh)
        if init_soc is None:
            py_sd.sim_drive()
        else:
            py_sd.sim_drive(init_soc)
        py_sd.set_post_scalars()
        sdp = simdrive.SimDrivePost(py_sd)
        py_diag = sdp.get_diagnostics()
        # RUST
        cyc = cycle.Cycle.from_file(cyc_name).to_rust()
        veh = vehicle.Vehicle.from_vehdb(vehid).to_rust()
        ru_sd = simdrive.RustSimDrive(cyc, veh)
        if init_soc is None:
            ru_sd.sim_drive()
        else:
            ru_sd.sim_drive(init_soc)
        ru_sd.set_post_scalars()
        sdp = simdrive.SimDrivePost(ru_sd)
        ru_diag = sdp.get_diagnostics()
        py_key_set = {k for k in py_diag}
        ru_key_set = {k for k in ru_diag}
        self.assertEqual(py_key_set, ru_key_set,
                         msg=(
                             "Key sets not equal;"
                             + f"\nonly in python: {py_key_set - ru_key_set}"
                             + f"\nonly in Rust  : {ru_key_set - py_key_set}"))
        vars_to_compare = [
            "aux_in_kw",
            "cur_max_avail_elec_kw",
            "cur_max_elec_kw",
            "cur_max_ess_chg_kw",
            "cur_ess_max_kw_out",
            "cur_max_fs_kw_out",
            "cur_max_mc_elec_kw_in",
            "cur_max_mech_mc_kw_in",
            "cur_max_trac_kw",
            "cur_max_trans_kw_out",
            "cyc_fric_brake_kw",
            "cyc_met",
            "cyc_tire_inertia_kw",
            "cyc_trac_kw_req",
            "cyc_trans_kw_out_req",
            "cyc_whl_rad_per_sec",
            "dist_m",
            "er_ae_kw_out",
            "ess_cap_lim_chg_kw",
            "ess_cap_lim_dischg_kw",
            "ess_lim_mc_regen_kw",
            "ess_lim_mc_regen_perc_kw",
            "fc_fs_lim_kw",
            "fc_max_kw_in",
            "fc_max_kw_in",
            "fc_trans_lim_kw",
            "high_acc_fc_on_tag",
            "max_trac_mps",
            "mc_elec_in_lim_kw",
            "mc_transi_lim_kw",
            "min_mc_kw_2help_fc",
            "mps_ach",
            "newton_iters",
            "reached_buff",
            "regen_buff_soc",
            "soc",
            "spare_trac_kw",
            "trans_kw_in_ach",
            "trans_kw_out_ach",
        ]
        N = len(cyc.time_s)
        found_discrepancy = False
        tol = 1e-6
        for i in range(N):
            for var in vars_to_compare:
                ru_val = np.array(ru_sd.__getattribute__(var))
                py_val = np.array(py_sd.__getattribute__(var))
                if type(py_val[i]) is np.bool_ or type(py_val[i]) is bool:
                    if py_val[i] != ru_val[i]:
                        found_discrepancy = True
                        print(
                            f"{var}[{i}]: py = {py_val[i]}; ru = {ru_val[i]}")
                else:
                    abs_diff = np.abs(py_val[i] - ru_val[i])
                    if abs_diff > tol:
                        found_discrepancy = True
                        print(
                            f"{var}[{i}]: difference = {abs_diff}; py = {py_val[i]}; ru = {ru_val[i]}")
            if found_discrepancy:
                break
        self.assertFalse(found_discrepancy)
        for k in py_key_set:
            self.assertAlmostEqual(py_diag[k], ru_diag[k],
                                   msg=f"{k} doesn't equal")


if __name__ == '__main__':
    df_err, df, df0, col_for_max_error, max_abs_err = main()
    unittest.main()
