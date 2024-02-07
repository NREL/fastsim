from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from cycler import cycler

import fastsim as fsim
import fastsim.fastsimrust as fsr


use_nsga2 = True

rho_fuel_kg_per_ml = 0.743e-3
lhv_fuel_btu_per_lbm = 18_344  # from "2012FordFusionV6Overview V5.pdf"
lbm_per_kg = 2.2
btu_per_kj = 0.948
lhv_fuel_kj_per_kg = lhv_fuel_btu_per_lbm * lbm_per_kg / btu_per_kj

def load_data() -> Dict[str, pd.DataFrame]:
    """
    Loads dyno test data from csv files 61811011, 61811012, 61811013, and 61811014
    downloaded from https://www.anl.gov/taps/d3-2018-toyota-camry-xle

    Returns:
        Dict[str, pd.DataFrame]: dictionary of dataframes
    """
    trip_dir = fsim.package_root() / "resources/calibration_demo_assets/dyno_data"
    # full data
    dfs_raw = dict()
    dfs = dict()
    for file in trip_dir.iterdir():
        print("loading: ", file.resolve())
        dfs_raw[file.stem] = pd.read_csv(file)
        # TODO: resample to 1 Hz
        # clip time at zero seconds
        dfs_raw[file.stem] = (
            dfs_raw[file.stem][dfs_raw[file.stem]
                               ['Time[s]'] >= 0.0]
        )

        dfs_raw[file.stem]['Fuel_Power_Calc[kW]'] = dfs_raw[
            file.stem]["Eng_FuelFlow_Direct_DI[ccps]"] * rho_fuel_kg_per_ml * lhv_fuel_kj_per_kg

        dfs_raw[file.stem]['Fuel_Energy_Calc[MJ]'] = (
            dfs_raw[file.stem]['Fuel_Power_Calc[kW]'] *
            dfs_raw[file.stem]['Time[s]'].diff().fillna(0.0)
        ).cumsum() / 1e3

        dfs[file.stem] = fsim.resample(
            dfs_raw[file.stem],
            rate_vars=('Eng_FuelFlow_Direct[cc/s]',)
        )
    assert len(dfs) > 0 
    return dfs


def get_cal_and_val_objs():
    dfs = load_data()

    # Separate calibration and validation cycles
    cal_cyc_patterns = ("12", "13", "14")
    dfs_cal = dict()
    for key in dfs.keys():
        for pattern in cal_cyc_patterns:
            if pattern in key:
                dfs_cal[key] = dfs[key]

    dfs_val_keys = set(dfs.keys()) - set(dfs_cal.keys())
    dfs_val = {key: dfs[key] for key in dfs_val_keys}

    # create cycles and sim_drives
    veh = fsim.vehicle.Vehicle.from_vehdb(3).to_rust()
    assert veh.scenario_name == "2016 TOYOTA Camry 4cyl 2WD"

    cycs = dict()
    cal_sim_drives = dict()
    val_sim_drives = dict()
    for key in dfs.keys():
        cycs[key] = fsim.cycle.Cycle.from_dict(
            {
                "time_s": dfs[key]["Time[s]"],
                "mps": dfs[key]["Dyno_Spd[mph]"] / fsim.params.MPH_PER_MPS
            }
        ).to_rust()
        sd = fsr.RustSimDrive(cycs[key], veh)

        # make tolerances big since runs may print lots of warnings before final design is selected
        fsim.utils.set_attrs_with_path(
            sd,
            {
                "sim_params.trace_miss_speed_mps_tol": 1e9,
                "sim_params.trace_miss_time_tol": 1e9,
                "sim_params.trace_miss_dist_tol": 1e9,
                "sim_params.energy_audit_error_tol": 1e9,
            },
        )

        if key in list(dfs_cal.keys()):
            cal_sim_drives[key] = sd.to_bincode()
        else:
            assert key in list(dfs_val.keys())
            val_sim_drives[key] = sd.to_bincode()

    params_and_bounds = (
        ("veh.fc_peak_eff", (0.2, 0.5), ),
    )
    params = [pb[0] for pb in params_and_bounds]
    params_bounds = [pb[1] for pb in params_and_bounds]
    obj_names = [
        # ("sd.fs_kw_out_ach", "Fuel_Power_Calc[kW]"),
        ("sd.fs_cumu_mj_out_ach", "Fuel_Energy_Calc[MJ]"),
        ("history.fc_te_deg_c", "CylinderHeadTempC"),
    ]

    cal_objectives = fsim.calibration.ModelObjectives(
        models=cal_sim_drives,
        dfs=dfs_cal,
        obj_names=obj_names,
        params=params,
        use_simdrivehot=True,
        verbose=False,
    )

    # to ensure correct key order
    val_sim_drives = {key: val_sim_drives[key] for key in dfs_val.keys()}
    val_objectives = fsim.calibration.ModelObjectives(
        models=val_sim_drives,
        dfs=dfs_val,
        obj_names=obj_names,
        params=params,
        use_simdrivehot=True,
        verbose=False,
    )

    return cal_objectives, val_objectives, params_bounds

if __name__ == "__main__":
    parser = fsim.cal.get_parser(
        # Defaults are set low to allow for fast run time during testing.  For a good
        # optimization, set this much higher.
        def_n_max_gen=3,
        def_pop_size=3,
        # TODO: figure out other terminaton criteria that should be included here.  
    )
    args = parser.parse_args()

    n_processes = args.processes
    n_max_gen = args.n_max_gen
    # should be at least as big as n_processes
    pop_size = args.pop_size
    run_minimize = not(args.skip_minimize)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True) # TODO: make this not save inside fastsim package
    show_plots = args.show
    make_plots = args.make_plots
    # override default of False
    use_simdrivehot = True

    cal_objectives, val_objectives, params_bounds = get_cal_and_val_objs()

    if run_minimize:
        print("Starting calibration.")
        if use_nsga2:
            algorithm = fsim.calibration.NSGA2(
                # size of each population
                pop_size=pop_size,
                sampling=fsim.calibration.LHS(),
            )
        else:
            algorithm = fsim.calibration.NSGA3(
                ref_dirs=fsim.calibration.get_reference_directions(
                    "energy",
                    n_dim=cal_objectives.n_obj,  # must be at least cal_objectives.n_obj
                    n_points=pop_size,  # must be at least pop_size
                ),
                # size of each population
                pop_size=pop_size,
                sampling=fsim.calibration.LHS(),
            )
        termination = fsim.calibration.DMOT(
            # max number of generations, default of 10 is very small
            n_max_gen=n_max_gen,
            # evaluate tolerance over this interval of generations every
            period=5,
        )

        if n_processes == 1:
            print("Running serial evaluation.")
            # series evaluation
            problem = fsim.calibration.CalibrationProblem(
                mod_obj=cal_objectives,
                param_bounds=params_bounds,
            )
            res, res_df = fsim.calibration.run_minimize(
                problem=problem,
                algorithm=algorithm,
                termination=termination,
                save_path=save_path,
            )
        else:
            print(
                f"Running parallel evaluation with n_processes: {n_processes}.")
            assert n_processes > 1
            # parallel evaluation
            import multiprocessing
            with multiprocessing.Pool(n_processes) as pool:
                problem = fsim.calibration.CalibrationProblem(
                    mod_obj=cal_objectives,
                    param_bounds=params_bounds,
                    elementwise_runner=fsim.cal.StarmapParallelization(
                        pool.starmap),
                    # func_eval=fsim.cal.starmap_parallelized_eval,
                )
                res, res_df = fsim.calibration.run_minimize(
                    problem=problem,
                    algorithm=algorithm,
                    termination=termination,
                    save_path=save_path,
                )
    else:
        res_df = pd.read_csv(save_path / "pymoo_res_df.csv")
        # with open(save_path / "pymoo_res.pickle", 'rb') as file:
        #     res = pickle.load(file)

    res_df['euclidean'] = (
        res_df.iloc[:, len(cal_objectives.params):] ** 2).sum(1).pow(1/2)

    best_row = res_df['euclidean'].argmin()
    best_df = res_df.iloc[best_row, :]
    res_df['fuel euclidean'] = (res_df.filter(
        like="fs_cumu") ** 2).sum(1) ** (1 / 2)
    res_df['temp euclidean'] = (res_df.filter(
        like="fc_te") ** 2).sum(1) ** (1 / 2)
    param_vals = res_df.iloc[best_row, :len(cal_objectives.params)].to_numpy()

    _, sdhots = cal_objectives.get_errors(
        cal_objectives.update_params(param_vals),
        plot_save_dir=save_path,
        show=show_plots and make_plots,
        plot=make_plots,
        plotly=make_plots,
        return_mods=True,
    )
    val_objectives.get_errors(
        val_objectives.update_params(param_vals),
        plot_save_dir=save_path,
        show=show_plots,
        plot=make_plots,
        plotly=make_plots,
    )

    # save calibrated vehicle to file
    sdhots[list(sdhots.keys())[0]].vehthrm.to_file(
        str(save_path / "2012_Ford_Fusion_thrml.yaml"))
