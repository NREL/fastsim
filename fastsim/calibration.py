from dataclasses import dataclass, InitVar
from typing import *
from pathlib import Path
import pickle

# pymoo
from pymoo.util.display import Display
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.core.sampling import Sampling
from pymoo.util.termination.default import MultiObjectiveDefaultTermination as MODT
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.optimize import minimize
from pymoo.core.problem import starmap_parallelized_eval, looped_eval

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
class ModelErrors(object):
    """
    Class for calculating eco-driving objectives
    """

    # dictionary of YAML models to be simulated
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

    def __post_init__(self):
        assert len(self.dfs) == len(
            self.models), f"{len(self.dfs)} != {len(self.models)}"

    def get_errors(
        self,
        sim_drives: Dict[str, fsr.SimDriveHot],
        return_mods: Optional[bool] = False,
        plot: Optional[bool] = False,
        plot_save_dir: Optional[str] = None,
        plot_perc_err: Optional[bool] = True,
    ) -> Union[
        Dict[str, Dict[str, float]],
        # or if return_mods is True
        Dict[str, fsim.simdrive.SimDrive],
    ]:
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
                ax[-1].plot(
                    time_hr,
                    sim_drive.sd.mph_ach,
                )
                ax[-1].set_xlabel('Time [hr]')
                ax[-1].set_ylabel('Speed [mph]')

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
                        f"trip: {key}, error: {objectives[key][obj[0]]:.3g}")
                    ax[i_obj * ax_multiplier].plot(time_hr,
                                                   mod_sig, label='mod',)
                    ax[i_obj * ax_multiplier].plot(time_hr,
                                                   ref_sig,
                                                   linestyle='--',
                                                   label="exp",
                                                   )
                    ax[i_obj * ax_multiplier].set_ylabel(obj[0])
                    ax[i_obj * ax_multiplier].legend()

                    if plot_perc_err:
                        # error
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
                            1].set_ylabel(obj[0] + "\n%Err")
                        ax[i_obj * ax_multiplier + 1].set_ylim([-20, 20])

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
        # Load SimDriveHot instances from YAML strings
        sim_drives = {key: fsr.SimDriveHot.from_yaml(
            model_yaml) for key, model_yaml in self.models.items()}
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
        err: ModelErrors,
        param_bounds: List[Tuple[float, float]],
        runner=None,
        func_eval: Callable = looped_eval,
    ):
        self.err = err
        # parameter lower and upper bounds
        self.param_bounds = param_bounds

        assert len(self.param_bounds) == len(
            self.err.params), f"{len(self.param_bounds)} != {len(self.err.params)}"
        super().__init__(
            n_var=len(self.err.params),
            n_obj=len(self.err.models) * len(self.err.obj_names),
            xl=[bounds[0]
                for bounds in self.param_bounds],
            xu=[bounds[1]
                for bounds in self.param_bounds],
            runner=runner,
            func_eval=func_eval,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        sim_drives = self.err.update_params(x)
        out['F'] = [
            val for inner_dict in self.err.get_errors(sim_drives).values() for val in inner_dict.values()
        ]


class CustomDisplay(Display):
    def __init__(self):
        super().__init__()
        self.term = MODT()
        self.t_gen_start = time.perf_counter()

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("n_nds", len(algorithm.opt), width=7)
        self.t_elapsed = time.perf_counter() - self.t_gen_start
        self.output.append(
            't [s]', f"{self.t_elapsed:.3g}", width=7)
        f = algorithm.pop.get('F')
        euclid_min = np.sqrt((np.array(f) ** 2).sum(axis=1)).min()
        self.output.append(
            "euclid min", f"{euclid_min:.3g}", width=8)

        self.term.do_continue(algorithm)


def run_minimize(
    prob: CalibrationProblem,
    algorithm: GeneticAlgorithm,
    termination: MODT,
    copy_algorithm: bool = False,
    copy_termination: bool = False,
    save_history: bool = False,
    save_path: Optional[str] = Path("pymoo_res/"),
):
    t0 = time.perf_counter()
    res = minimize(
        prob,
        algorithm,
        termination,
        copy_algorithm,
        copy_termination,
        seed=1,
        verbose=True,
        save_history=save_history,
        display=CustomDisplay(),
    )

    f_columns = [
        f"{key}: {obj[0]}"
        for key in prob.err.dfs.keys()
        for obj in prob.err.obj_names
    ]
    f_df = pd.DataFrame(
        data=[f for f in res.F.tolist()],
        columns=f_columns,
    )

    x_df = pd.DataFrame(
        data=[x for x in res.X.tolist()],
        columns=[param for param in prob.err.params],
    )

    Path(save_path).mkdir(exist_ok=True, parents=True)
    with open(Path(save_path) / "pymoo_res.pickle", 'wb') as file:
        pickle.dump(res, file)
    res_df = pd.concat([x_df, f_df], axis=1)
    res_df.to_csv(Path(save_path) / "pymoo_res_df.csv", index=False)

    t1 = time.perf_counter()
    print(f"Elapsed time to run minimization: {t1-t0:.5g} s")

    return res, res_df
