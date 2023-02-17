from dataclasses import dataclass, InitVar
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pickle
import argparse

# pymoo
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.termination.default import DefaultMultiObjectiveTermination as DMOT
from pymoo.core.problem import Problem, ElementwiseProblem, LoopedElementwiseEvaluation, StarmapParallelization
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.optimize import minimize

# misc
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import numpy as np

# local -- expects rust-port version of fastsim so make sure this is in the env
import fastsim as fsim
import fastsimrust as fsr


def get_error_val(model, test, time_steps):
    """Returns time-averaged error for model and test signal.
    Arguments:
    ----------
    model: array of values for signal from model
    test: array of values for signal from test data
    time_steps: array (or scalar for constant) of values for model time steps [s]
    test: array of values for signal from test

    Output:
    -------
    err: integral of absolute value of difference between model and
    test per time"""

    assert len(model) == len(test) == len(
        time_steps), f"{len(model)}, {len(test)}, {len(time_steps)}"

    return np.trapz(y=abs(model - test), x=time_steps) / (time_steps[-1] - time_steps[0])


@dataclass
class ModelObjectives(object):
    """
    Class for calculating eco-driving objectives
    """

    # dictionary of bincode models to be simulated
    models: Dict[str, str]

    # dictionary of test data to calibrate against
    dfs: Dict[str, pd.DataFrame]

    # list of 1- or 2-element tuples of objectives for error calcluation
    # 1 element:
    # value to be minimized -- e.g. `(("sim_drive", "mpgge"), )`
    # 2 element:
    # - model value, e.g. achieved cycle duration
    # - reference value, e.g. prescribed cycle duration
    # the code for this example may not be fully implemented or may need tweaking
    # - example: `[
    #   (("sim_drive", "cyc", "time_s", -1), ("sim_drive", "cyc0", "time_s", -1)),
    #   ("sim_drive", "mpgge"),
    # ]`
    obj_names: List[Tuple[str, Optional[Tuple[str]]]]

    # list of tuples hierarchical paths to parameters to manipulate
    # example - `[('sim_drive', 'sim_params', 'idm_accel_m_per_s2')]`
    params: List[Tuple[str]]

    # if True, prints timing and misc info
    verbose: bool = False

    # calculated in __post_init__
    n_obj: int = None

    def __post_init__(self):
        assert len(self.dfs) == len(
            self.models), f"{len(self.dfs)} != {len(self.models)}"
        self.n_obj = len(self.models) * len(self.obj_names)

    def get_errors(
        self,
        sim_drives: Dict[str, fsr.SimDriveHot],
        return_mods: Optional[bool] = False,
        plot: Optional[bool] = False,
        plot_save_dir: Optional[str] = None,
        plot_perc_err: Optional[bool] = True,
        show: Optional[bool] = False,
        fontsize: Optional[float] = 12,
    ) -> Union[
        Dict[str, Dict[str, float]],
        # or if return_mods is True
        Dict[str, fsim.simdrive.SimDrive],
    ]:
        # TODO: should return type instead be `Dict[str, Dict[str, float]] | Tuple[Dict[str, Dict[str, float]], Dict[str, fsim.simdrive.SimDrive]]`
        # This would make `from typing import Union` unnecessary
        """
        Calculate model errors w.r.t. test data for each element in dfs/models for each objective.
        Arguments:
        ----------
            - return_mods: if true, also returns dict of solved models
            - plot: if true, plots objectives
        """

        objectives = {}
        solved_mods = {}

        # loop through all the provided trips
        for ((key, df_exp), sim_drive) in zip(self.dfs.items(), sim_drives.values()):
            t0 = time.perf_counter()
            sim_drive = sim_drive.copy()  # TODO: do we need this?
            sim_drive.sim_drive()
            objectives[key] = {}
            if return_mods or plot:
                solved_mods[key] = sim_drive.copy()

            if plot or plot_save_dir:
                Path(plot_save_dir).mkdir(exist_ok=True, parents=True)
                time_hr = np.array(sim_drive.sd.cyc.time_s) / 3_600
                ax_multiplier = 2 if plot_perc_err else 1
                fig, ax = plt.subplots(
                    len(self.obj_names) * ax_multiplier + 1, 1, sharex=True, figsize=(12, 8),
                )
                plt.suptitle(f"trip: {key}", fontsize=fontsize)
                ax[-1].plot(
                    time_hr,
                    sim_drive.sd.mph_ach,
                )
                ax[-1].set_xlabel('Time [hr]', fontsize=fontsize)
                ax[-1].set_ylabel('Speed [mph]', fontsize=fontsize)

            t1 = time.perf_counter()
            if self.verbose:
                print(f"Time to simulate {key}: {t1 - t0:.3g}")

            # loop through the objectives for each trip
            for i_obj, obj in enumerate(self.obj_names):
                assert len(obj) == 1 or len(obj) == 2
                if len(obj) == 2:
                    # If objective attribute path and reference signal passed
                    mod_path = obj[0]  # str, path to objective
                    ref_path = obj[1]  # str, df column name
                elif len(obj) == 1:
                    # Minimizing objective attribute
                    mod_path = obj[0]  # str
                    ref_path = None

                # extract model value
                mod_sig = fsim.utils.get_attr_with_path(sim_drive, mod_path)

                if ref_path:
                    ref_sig = df_exp[ref_path]
                    objectives[key][obj[0]] = get_error_val(
                        mod_sig,
                        ref_sig,
                        sim_drive.sd.cyc.time_s,
                    )
                else:
                    pass
                    # TODO: write else block for objective minimization

                if plot or plot_save_dir:
                    # this code needs to be cleaned up
                    # raw signals
                    ax[i_obj * ax_multiplier].set_title(
                        f"error: {objectives[key][obj[0]]:.3g}", fontsize=fontsize)
                    ax[i_obj * ax_multiplier].plot(time_hr,
                                                   mod_sig, label='mod')
                    ax[i_obj * ax_multiplier].plot(time_hr,
                                                   ref_sig,
                                                   linestyle='--',
                                                   label="exp",
                                                   )
                    ax[i_obj *
                        ax_multiplier].set_ylabel(obj[0], fontsize=fontsize)
                    ax[i_obj * ax_multiplier].legend(fontsize=fontsize)

                    if plot_perc_err:
                        # error
                        if "deg_c" in mod_path:
                            perc_err = (mod_sig - ref_sig) / \
                                (ref_sig + 273.15) * 100
                        else:
                            perc_err = (mod_sig - ref_sig) / ref_sig * 100
                        # clean up inf and nan
                        perc_err[np.where(perc_err == np.inf)[0][:]] = 0.0
                        # trim off the first few bits of junk
                        perc_err[np.where(perc_err > 500)[0][:]] = 0.0
                        ax[i_obj * ax_multiplier + 1].plot(
                            time_hr,
                            perc_err
                        )
                        ax[i_obj * ax_multiplier +
                            1].set_ylabel(obj[0] + "\n%Err", fontsize=fontsize)
                        ax[i_obj * ax_multiplier + 1].set_ylim([-20, 20])

                    if show:
                        plt.show()

                if plot_save_dir:
                    if not Path(plot_save_dir).exists():
                        Path(plot_save_dir).mkdir()
                    plt.tight_layout()
                    plt.savefig(Path(plot_save_dir) / f"{key}.svg")
                    plt.savefig(Path(plot_save_dir) / f"{key}.png")

            t2 = time.perf_counter()
            if self.verbose:
                print(f"Time to postprocess: {t2 - t1:.3g} s")

        if return_mods:
            return objectives, solved_mods
        else:
            return objectives

    def update_params(self, xs: List[Any]):
        """
        Updates model parameters based on `x`, which must match length of self.params
        """
        assert len(xs) == len(self.params), f"({len(xs)} != {len(self.params)}"
        paths = [fullpath.split(".") for fullpath in self.params]
        t0 = time.perf_counter()
        # Load SimDriveHot instances from bincode strings
        sim_drives = {key: fsr.SimDriveHot.from_bincode(
            model_bincode) for key, model_bincode in self.models.items()}
        # Update all model parameters
        for key in sim_drives.keys():
            sim_drives[key] = fsim.utils.set_attrs_with_path(
                sim_drives[key],
                dict(zip(self.params, xs)),
            )
        t1 = time.perf_counter()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")
        return sim_drives


@dataclass
class CalibrationProblem(ElementwiseProblem):
    """
    Problem for calibrating models to match test data
    """

    def __init__(
        self,
        mod_obj: ModelObjectives,
        param_bounds: List[Tuple[float, float]],
        elementwise_runner=LoopedElementwiseEvaluation(),
    ):
        self.mod_obj = mod_obj
        # parameter lower and upper bounds
        self.param_bounds = param_bounds
        assert len(self.param_bounds) == len(
            self.mod_obj.params), f"{len(self.param_bounds)} != {len(self.mod_obj.params)}"
        super().__init__(
            n_var=len(self.mod_obj.params),
            n_obj=self.mod_obj.n_obj,
            xl=[bounds[0]
                for bounds in self.param_bounds],
            xu=[bounds[1]
                for bounds in self.param_bounds],
            elementwise_runner=elementwise_runner,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        sim_drives = self.mod_obj.update_params(x)
        out['F'] = [
            val for inner_dict in self.mod_obj.get_errors(sim_drives).values() for val in inner_dict.values()
        ]


class CustomOutput(Output):
    def __init__(self):
        super().__init__()
        self.t_gen_start = time.perf_counter()
        self.n_nds = Column("n_nds", width=8)
        self.t_s = Column("t [s]", width=10)
        self.euclid_min = Column("euclid min", width=13)
        self.columns += [self.n_nds, self.t_s, self.euclid_min]

    def update(self, algorithm):
        super().update(algorithm)
        self.n_nds.set(len(algorithm.opt))
        self.t_s.set(f"{(time.perf_counter() - self.t_gen_start):.3g}")
        f = algorithm.pop.get('F')
        euclid_min = np.sqrt((np.array(f) ** 2).sum(axis=1)).min()
        self.euclid_min.set(f"{euclid_min:.3g}")


def run_minimize(
    problem: CalibrationProblem,
    algorithm: GeneticAlgorithm,
    termination: DMOT,
    copy_algorithm: bool = False,
    copy_termination: bool = False,
    save_history: bool = False,
    save_path: Optional[str] = Path("pymoo_res/"),
):
    print("`run_minimize` starting at")
    fsim.utils.print_dt()

    t0 = time.perf_counter()
    res = minimize(
        problem,
        algorithm,
        termination,
        copy_algorithm,
        copy_termination,
        seed=1,
        verbose=True,
        save_history=save_history,
        output=CustomOutput(),
    )

    f_columns = [
        f"{key}: {obj[0]}"
        for key in problem.mod_obj.dfs.keys()
        for obj in problem.mod_obj.obj_names
    ]
    f_df = pd.DataFrame(
        data=[f for f in res.F.tolist()],
        columns=f_columns,
    )

    x_df = pd.DataFrame(
        data=[x for x in res.X.tolist()],
        columns=[param for param in problem.mod_obj.params],
    )

    Path(save_path).mkdir(exist_ok=True, parents=True)
    # with open(Path(save_path) / "pymoo_res.pickle", 'wb') as file:
    #     pickle.dump(res, file)

    res_df = pd.concat([x_df, f_df], axis=1)
    res_df['euclidean'] = (
        res_df.iloc[:, len(problem.mod_obj.params):] ** 2).sum(1).pow(1/2)
    res_df.to_csv(Path(save_path) / "pymoo_res_df.csv", index=False)

    t1 = time.perf_counter()
    print(f"Elapsed time to run minimization: {t1-t0:.5g} s")

    return res, res_df


def get_parser() -> argparse.ArgumentParser:
    """
    Generate parser for optimization hyper params and misc. other params
    """
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
    parser.add_argument('--show', action="store_true",
                        help="If provided, shows plots.")
    parser.add_argument("--make-plots", action="store_true",
                        help="Generates plots, if provided.")
    return parser
