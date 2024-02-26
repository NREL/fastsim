"""
Script for demonstrating how to calibrate a vehicle model.  See [FASTSim
Calibration/Validation documentation](https://nrel.github.io/fastsim/cal_and_val.html)
for more info on how to use this.  
"""

from typing import Dict, Tuple, List
from pathlib import Path
import pandas as pd
import plotly.express as px
import os

import fastsim as fsim
import fastsim.fastsimrust as fsr

use_nsga2 = True

# density of fuel
rho_fuel_kg_per_ml = 0.743e-3
# lower heating value of fuel
lhv_fuel_btu_per_lbm = 18_344  # from "2012FordFusionV6Overview V5.pdf"
# conversoion factors
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
    # Change this to whatever your data directory is
    trip_dir = fsim.package_root() / "resources/calibration_demo_assets/dyno_data"
    # full data
    dfs_raw = dict()
    dfs = dict()
    for file in trip_dir.iterdir():
        print("loading: ", file.resolve())
        dfs_raw[file.stem] = pd.read_csv(file)
        # clip time at zero seconds
        dfs_raw[file.stem] = dfs_raw[file.stem][dfs_raw[file.stem]["Time[s]"] >= 0.0]

        dfs_raw[file.stem]["Fuel_Power_Calc[kW]"] = (
            dfs_raw[file.stem]["Eng_FuelFlow_Direct_DI[ccps]"]
            * rho_fuel_kg_per_ml
            * lhv_fuel_kj_per_kg
        )

        dfs_raw[file.stem]["Fuel_Energy_Calc[MJ]"] = (
            dfs_raw[file.stem]["Fuel_Power_Calc[kW]"]
            * dfs_raw[file.stem]["Time[s]"].diff().fillna(0.0)
        ).cumsum() / 1e3

        dfs[file.stem] = fsim.resample(
            dfs_raw[file.stem], rate_vars=("Eng_FuelFlow_Direct[cc/s]",)
        )
    assert len(dfs) > 0
    return dfs


def get_cal_and_val_objs(
    dfs:Dict[str, pd.DataFrame]=load_data(),
) -> Tuple[fsim.cal.ModelObjectives, fsim.cal.ModelObjectives, List[Tuple[float, float]]]:
    """
    Returns objects to be used by PyMOO optimizer

    Args:
        dfs (Dict[str, pd.DataFrame]): output of `load_data`

    Returns:
        Tuple[fsim.cal.ModelObjectives, fsim.cal.ModelObjectives, List[Tuple[float, float]]]: _description_
    """
    dfs = load_data()

    # Separate calibration and validation cycles  
    # tuple of regex patterns that match cycles to be used for calibration; any cycles
    # not selected for calibration are reserved for validation.  
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
        # construct cycle from dyno test data
        cycs[key] = fsim.cycle.Cycle.from_dict(
            {
                "time_s": dfs[key]["Time[s]"],
                "mps": dfs[key]["Dyno_Spd[mph]"] / fsim.params.MPH_PER_MPS,
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
            cal_sim_drives[key] = sd.to_json()
        else:
            assert key in list(dfs_val.keys())
            val_sim_drives[key] = sd.to_json()

    params_and_bounds = (
        # `veh.fc_peak_eff` is allowed to vary between 0.2 and 0.5
        ("veh.fc_peak_eff", (0.2, 0.5)), 
    )
    params = [pb[0] for pb in params_and_bounds]
    bounds = [pb[1] for pb in params_and_bounds]
    obj_names = [
        (
            "fs_cumu_mj_out_ach",  # fastsim signal name
            "Fuel_Energy_Calc[MJ]" # matching test data signal to be used as benchmark
        ),
    ]

    cal_objectives = fsim.calibration.ModelObjectives(
        models=cal_sim_drives,
        dfs=dfs_cal,
        obj_names=obj_names,
        params=params,
        verbose=False,
    )

    # to ensure correct key order
    val_sim_drives = {key: val_sim_drives[key] for key in dfs_val.keys()}
    val_objectives = fsim.calibration.ModelObjectives(
        models=val_sim_drives,
        dfs=dfs_val,
        obj_names=obj_names,
        params=params,
        verbose=False,
    )

    return cal_objectives, val_objectives, bounds


if __name__ == "__main__":
    # if True, this file is being run as part of test
    TESTING = os.environ.get("TESTING", "false").lower() == "true"

    parser = fsim.cal.get_parser(
        # Defaults are set low to allow for fast run time during testing.  For a good
        # optimization, set this much higher.
        def_save_path=None,
    )
    args = parser.parse_args()

    n_processes = args.processes if not TESTING else 1
    n_max_gen = args.n_max_gen if not TESTING else 3
    # should be at least as big as n_processes
    pop_size = args.pop_size if not TESTING else 1
    run_minimize = not (args.skip_minimize)
    if args.save_path is not None:
        save_path = Path(args.save_path) 
        save_path.mkdir(exist_ok=True)
    else:
        save_path = None

    show_plots = args.show
    make_plots = args.make_plots

    cal_objectives, val_objectives, params_bounds = get_cal_and_val_objs()

    if run_minimize:
        print("Starting calibration.")
        if use_nsga2:
            algorithm = fsim.calibration.NSGA2(
                # size of each population
                pop_size=pop_size,
                # LatinHyperCube sampling seems to be more effective than the default
                # random sampling
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
                # LatinHyperCube sampling seems to be more effective than the default
                # random sampling
                sampling=fsim.calibration.LHS(),
            )
        termination = fsim.calibration.DMOT(
            # max number of generations, default of 10 is very small
            n_max_gen=n_max_gen,
            # evaluate tolerance over this interval of generations every
            period=5,
            # parameter variation tolerance
            xtol=args.xtol,
            # objective variation tolerance
            ftol=args.ftol
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
            print(f"Running parallel evaluation with n_processes: {n_processes}.")
            assert n_processes > 1
            # parallel evaluation
            import multiprocessing

            with multiprocessing.Pool(n_processes) as pool:
                problem = fsim.calibration.CalibrationProblem(
                    mod_obj=cal_objectives,
                    param_bounds=params_bounds,
                    elementwise_runner=fsim.cal.StarmapParallelization(pool.starmap),
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

    # ***********************************************************************************
    # IMPORTANT NOTE: usually, all the stuff that happens below this line is in another
    # file that can be run interactively to enable exploration of results besides the
    # best euclidean result.  This is not particularly practical for a demo.
    # ***********************************************************************************

    res_df["euclidean"] = (
        (res_df.iloc[:, len(cal_objectives.params) :] ** 2).sum(1).pow(1 / 2)
    )

    best_row = res_df["euclidean"].argmin()
    best_df = res_df.iloc[best_row, :]
    param_vals = res_df.iloc[best_row, : len(cal_objectives.params)].to_numpy()

    _, sds = cal_objectives.get_errors(
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
        show=show_plots and make_plots,
        plot=make_plots,
        plotly=make_plots,
    )

    # save calibrated vehicle based on euclidean-minimized result to file
    if save_path is not None:
        sds[list(sds.keys())[0]].veh.to_file(
            str(save_path / "2016_TOYOTA_Camry_4cyl_2WD_optimized.yaml")
        )

    if make_plots and save_path is not None and len(res_df) > 1:
        res_df.insert(0, 'index', res_df.index)
        fig = px.parallel_coordinates(
            res_df, 
            color="euclidean", 
            color_continuous_scale=px.colors.diverging.Tealrose
        )
        fig.update_layout(
            xaxis_title="Parameters and Objectives",
            yaxis_title="Values",
            coloraxis_colorbar=dict(title="Euclidean Min")
        )
        fig.write_html(save_path / "parallel coord.html", auto_open=True)
    else:
        print("Skipping parallel coordinates plot because only 1 design was found.")
