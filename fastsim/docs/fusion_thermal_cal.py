from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from cycler import cycler

import fastsim as fsim
import fastsimrust as fsr


def load_data() -> Dict[str, pd.DataFrame]:
    # full data
    dfs_raw = dict()
    # resampled to 1 Hz
    dfs = dict()
    for sub in trip_dir.iterdir():
        if sub.is_dir():
            for file in sub.iterdir():
                if file.suffix == ".csv" and "_cs" in file.stem:
                    print(f"loading: ", file.resolve())
                    dfs_raw[file.stem] = pd.read_csv(file)
                    # clip time at zero seconds
                    dfs_raw[file.stem] = dfs_raw[file.stem][dfs_raw[file.stem]
                                                            ['Time[s]'] >= 0.0]

                    dfs_raw[file.stem]['Fuel_Power_Calc[kW]'] = dfs_raw[
                        file.stem]["Eng_FuelFlow_Direct[cc/s]"] * rho_fuel_kg_per_ml * lhv_fuel_kj_per_kg

                    dfs_raw[file.stem]['Fuel_Energy_Calc[MJ]'] = (
                        dfs_raw[file.stem]['Fuel_Power_Calc[kW]'] *
                        dfs_raw[file.stem]['Time[s]'].diff().fillna(0.0)
                    ).cumsum() / 1e3

                    dfs[file.stem] = fsim.resample(
                        dfs_raw[file.stem],
                        rate_vars=('Eng_FuelFlow_Direct[cc/s]')
                    )
    return dfs


def get_cal_and_val_objs():
    dfs = load_data()

    # Separate calibration and validation cycles
    cal_cyc_patterns = ("49", "56", "73", "60", "69", "77")
    dfs_cal = dict()
    for key in dfs.keys():
        for pattern in cal_cyc_patterns:
            if pattern in key:
                dfs_cal[key] = dfs[key]

    dfs_val_keys = set(dfs.keys()) - set(dfs_cal.keys())
    dfs_val = {key: dfs[key] for key in dfs_val_keys}

    # create cycles and sim_drives
    veh = fsim.vehicle.Vehicle.from_file(
        "2012_Ford_Fusion.csv", to_rust=True).to_rust()
    vehthrm = fsr.VehicleThermal.default()

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
        init_state = fsr.ThermalState(
            amb_te_deg_c=dfs[key]['Cell_Temp[C]'][0],
            fc_te_deg_c_init=dfs[key]['CylinderHeadTempC'][0],
        )
        sdh = fsr.SimDriveHot(
            cycs[key],
            veh,
            vehthrm,
            init_state,
            amb_te_deg_c=dfs[key]["Cell_Temp[C]"]
        )

        # make tolerances big since runs may print lots of warnings before final design is selected
        sdh = fsim.utils.set_attrs_with_path(
            sdh,
            {
                "sd.sim_params.trace_miss_speed_mps_tol": 1e9,
                "sd.sim_params.trace_miss_time_tol": 1e9,
                "sd.sim_params.trace_miss_dist_tol": 1e9,
                "sd.sim_params.energy_audit_error_tol": 1e9,
            },
        )

        if key in list(dfs_cal.keys()):
            cal_sim_drives[key] = sdh.to_bincode()
        else:
            assert key in list(dfs_val.keys())
            val_sim_drives[key] = sdh.to_bincode()

    params_and_bounds = (
        ("vehthrm.fc_c_kj__k", (50, 200), ),
        ("vehthrm.fc_l", (0.25, 2), ),
        ("vehthrm.fc_htc_to_amb_stop", (5, 50), ),
        ("vehthrm.fc_coeff_from_comb", (1e-5, 1e-3), ),
        ("vehthrm.fc_exp_offset", (-10, 30), ),
        ("vehthrm.fc_exp_lag", (15, 100), ),
        ("vehthrm.fc_exp_minimum", (0.15, 0.45), ),
        ("vehthrm.rad_eps", (5, 50), ),
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
        verbose=False
    )

    # to ensure correct key order
    val_sim_drives = {key: val_sim_drives[key] for key in dfs_val.keys()}
    val_objectives = fsim.calibration.ModelObjectives(
        models=val_sim_drives,
        dfs=dfs_val,
        obj_names=obj_names,
        params=params,
        verbose=False
    )

    return cal_objectives, val_objectives, params_bounds


# load test data which can be obtained at
# https://www.anl.gov/taps/d3-2012-ford-fusion-v6
possible_trip_dirs = (
    Path().home() / "Documents/DynoTestData/FordFusionTestData/",
    Path().home() / "scratch/FordFusionTestData/",    
)

for trip_dir in possible_trip_dirs:
    if trip_dir.exists():
        break

rho_fuel_kg_per_ml = 0.743e-3
lhv_fuel_btu_per_lbm = 18_344  # from "2012FordFusionV6Overview V5.pdf"
lbm_per_kg = 2.2
btu_per_kj = 0.948
lhv_fuel_kj_per_kg = lhv_fuel_btu_per_lbm * lbm_per_kg / btu_per_kj


if __name__ == "__main__":
    parser = fsim.cal.get_parser()
    args = parser.parse_args()

    n_processes = args.processes
    n_max_gen = args.n_max_gen
    # should be at least as big as n_processes
    pop_size = args.pop_size
    run_minimize = not(args.skip_minimize)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    show_plots = args.show
    make_plots = args.make_plots

    cal_objectives, val_objectives, params_bounds = get_cal_and_val_objs()

    if run_minimize:
        print("Starting calibration.")

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
        return_mods=True,
    )
    val_objectives.get_errors(
        val_objectives.update_params(param_vals),
        plot_save_dir=save_path,
        show=show_plots,
    )

    # save calibrated vehicle to file
    sdhots[list(sdhots.keys())[0]].vehthrm.to_file(
        str(save_path / "2012_Ford_Fusion_thrml.yaml"))
