from typing import *
from pathlib import Path
import argparse
import pandas as pd
import pickle

import fastsim as fsim
import fastsimrust as fsr

DATA_DIR = Path().home() / "Documents/DynoTestData/FordFusionTestData/"


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
                        dfs_raw[file.stem]['Fuel_Power_Calc[kW]'] * dfs_raw[file.stem]['Time[s]'].diff().fillna(0.0)).cumsum() / 1e3

                    dfs[file.stem] = fsim.resample(
                        dfs_raw[file.stem],
                        rate_vars=('Eng_FuelFlow_Direct[cc/s]')
                    )
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-p', '--processes', type=int,
                        default=4, help="Number of pool processes.")
    parser.add_argument('--n-max-gen', type=int, default=500,
                        help="PyMOO termination criterion: n_max_gen.")
    parser.add_argument('--pop-size', type=int, default=12,
                        help="PyMOO population size in each generation.")
    parser.add_argument('--skip-minimize', action="store_true",
                        help="If provided, load previous results.")
    parser.add_argument('--save-path', type=str, default="pymoo_res",
                        help="File location to save results.")
    args = parser.parse_args()

    n_processes = args.processes
    n_max_gen = args.n_max_gen
    pop_size = args.pop_size
    run_minimize = not(args.skip_minimize)
    save_path = "fusion_pymoo_res"  # args.save_path

    # load test data which can be obtained at
    # https://www.anl.gov/taps/d3-2012-ford-fusion-v6
    possible_trip_dirs = (
        DATA_DIR,
    )

    for trip_dir in possible_trip_dirs:
        if trip_dir.exists():
            break

    rho_fuel_kg_per_ml = 0.743e-3
    lhv_fuel_btu_per_lbm = 18_344
    lbm_per_kg = 2.2
    btu_per_kj = 0.948
    lhv_fuel_kj_per_kg = lhv_fuel_btu_per_lbm * lbm_per_kg / btu_per_kj

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
        )

        # make tolerances big since runs may print lots of warnings before final design is selected
        sdh = fsim.utils.set_attrs_with_path(
            sdh,
            {
                "sd.sim_params.trace_miss_speed_mps_tol": 1e9,
                "sd.sim_params.trace_miss_time_tol": 1e9,
                "sd.sim_params.trace_miss_dist_tol": 1e9,
                "sd.sim_params.energy_audit_error_tol": 1e9,
                "sd.sim_params.verbose": False,
            },
        )

        if key in list(dfs_cal.keys()):
            cal_sim_drives[key] = sdh.to_yaml()
        else:
            assert key in list(dfs_val.keys())
            val_sim_drives[key] = sdh.to_yaml()

    # Simulate
    params = [
        "vehthrm.fc_c_kj__k",
        "vehthrm.fc_l",
        "vehthrm.fc_htc_to_amb_stop",
        "vehthrm.fc_coeff_from_comb",
        "vehthrm.fc_exp_offset",
        "vehthrm.fc_exp_lag",
        "vehthrm.fc_exp_minimum",
        "vehthrm.rad_eps",
    ]

    params_bounds = [
        (50, 200),
        (0.25, 2),
        (5, 50),
        (1e-5, 1e-3),
        (-10, 30),
        (15, 75),
        (0.25, 0.45),
        (5, 50),
    ]

    obj_names = [
        # ("sd.fs_kw_out_ach", "Fuel_Power_Calc[kW]"),
        ("sd.fs_cumu_mj_out_ach", "Fuel_Energy_Calc[MJ]"),
        ("history.fc_te_deg_c", "CylinderHeadTempC"),
    ]

    cal_objectives = fsim.calibration.ModelErrors(
        models=cal_sim_drives,
        dfs=dfs_cal,
        obj_names=obj_names,
        params=params,
        verbose=False
    )

    # to ensure correct key order
    val_sim_drives = {key: val_sim_drives[key] for key in dfs_val.keys()}
    val_objectives = fsim.calibration.ModelErrors(
        models=val_sim_drives,
        dfs=dfs_val,
        obj_names=obj_names,
        params=params,
        verbose=False
    )

    if run_minimize:
        print("Starting calibration.")

        algorithm = fsim.calibration.NSGA2(
            # size of each population
            pop_size=pop_size,
            sampling=fsim.calibration.LHS(),
        )
        termination = fsim.calibration.MODT(
            # max number of generations, default of 10 is very small
            n_max_gen=n_max_gen,
            # evaluate tolerance over this interval of generations every `nth_gen`
            n_last=10,
        )

        if n_processes == 1:
            problem = fsim.calibration.CalibrationProblem(
                err=cal_objectives,
                param_bounds=params_bounds,
            )
            res, res_df = fsim.calibration.run_minimize(
                problem,
                algorithm=algorithm,
                termination=termination,
                save_path=save_path,
            )
        else:
            import multiprocessing
            with multiprocessing.Pool(n_processes) as pool:
                with multiprocessing.Pool(n_processes) as pool:
                    problem = fsim.calibration.CalibrationProblem(
                        err=cal_objectives,
                        param_bounds=params_bounds,
                        runner=pool.starmap,
                        func_eval=fsim.calibration.starmap_parallelized_eval,
                    )
                    res, res_df = fsim.calibration.run_minimize(
                        problem,
                        algorithm,
                        termination=termination,
                    )
    else:
        res_df = pd.read_csv(Path(save_path) / "pymoo_res_df.csv")
        with open(Path(save_path) / "pymoo_res.pickle", 'rb') as file:
            res = pickle.load(file)

    res_df['euclidean'] = (
        res_df.iloc[:, len(params):] ** 2).sum(1).pow(1/2)
    best_row = res_df['euclidean'].argmin()
    best_df = res_df.iloc[best_row, :]
    param_vals = res_df.iloc[0, :len(cal_objectives.params)].to_numpy()

    cal_objectives.get_errors(
        cal_objectives.update_params(param_vals),
        plot_save_dir=Path("plots/fusion/cal/")
    )
    val_objectives.get_errors(
        val_objectives.update_params(param_vals),
        plot_save_dir=Path("plots/fusion/val/")
    )

    # save calibrated vehicle to file
    veh_save_dir = Path("../resources/vehdb/thermal/")
    veh_save_dir.mkdir(exist_ok=True)
    sdh = fsr.SimDriveHot.from_yaml(
        cal_objectives.models[list(cal_objectives.models.keys())[0]])
    sdh.vehthrm.to_file(str(veh_save_dir / "2012_Ford_Fusion_thrml.yaml"))


# # %%

# # params_and_vals = {
# #     'vehthrm.fc_c_kj__k': 125.0,
# #     'vehthrm.fc_l': 1.3,
# #     'vehthrm.fc_htc_to_amb_stop': 100.0,
# #     'vehthrm.fc_coeff_from_comb': 0.00030721481819805005,
# #     'vehthrm.fc_exp_offset': -9.438669088889137,
# #     'vehthrm.fc_exp_lag': 30.0,
# #     'vehthrm.fc_exp_minimum': 0.2500008623533276,
# #     'vehthrm.rad_eps': 20
# #  }

# plot_save_dir = Path("plots")
# plot_save_dir.mkdir(exist_ok=True)

# # problem.err.update_params(params_and_vals.values())
# problem.err.update_params(param_vals)
# problem.err.get_errors(
#     plot=True, plot_save_dir=plot_save_dir, plot_perc_err=False)


# # %%

# # Demonstrate with model showing fuel usage impact

# # get the optimal vehthrm
# vehthrm = problem.err.sim_drives[
#     list(problem.err.sim_drives.keys())[0]
# ].vehthrm

# # manual adjustment of parameters, should be turned off when
# # checking new run
# # vehthrm = fsim.auxiliaries.set_nested_values(
# #     vehthrm, fc_exp_lag=17.0
# # )

# te_amb_deg_c_arr = np.arange(-10, 101)
# mpg_arr = np.zeros(len(te_amb_deg_c_arr))

# for i, te_amb_deg_c in enumerate(te_amb_deg_c_arr):
#     sdh = fsr.SimDriveHot(
#         fsim.cycle.Cycle.from_file("udds").to_rust(),
#         veh,
#         vehthrm,
#         fsr.ThermalState(
#             amb_te_deg_c=min(te_amb_deg_c, 50),
#             fc_te_deg_c_init=te_amb_deg_c
#         )
#     )
#     sdh.sim_drive()
#     mpg_arr[i] = sdh.sd.mpgge

# sd = fsim.simdrive.RustSimDrive(
#     fsim.cycle.Cycle.from_file("udds").to_rust(),
#     veh,
# )
# sd.sim_drive()


# # %%

# # by about 90°C, 'with thermal' should be nearly the same as 'no thermal'

# fig, ax = plt.subplots()
# ax.scatter(te_amb_deg_c_arr, mpg_arr, label='with thermal')
# ax.axhline(y=sd.mpgge, label='no thermal', color='red')
# ax.set_xlabel("Ambient/Cold Start Temperature [°C]")
# ax.set_ylabel("Fuel Economy [mpg]")
# ax.set_title("2012 Ford Fusion V6")
# ax.legend()
# plt.tight_layout()
# plt.savefig("plots/fe v amb temp.svg")
# plt.savefig("plots/fe v amb temp.png")

# # %%
