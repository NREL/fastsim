from dataclasses import dataclass, InitVar
from typing import Union, Dict, Tuple, List, Any, Union, Optional
from pathlib import Path

# pymoo
from pymoo.util.display import Display
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.core.sampling import Sampling
from pymoo.util.termination.default import MultiObjectiveDefaultTermination as MODT
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
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

    return np.trapz(y=abs(model - test), x=time_steps) / (time_steps[-1] - time_steps[0])


def get_containers_from_path(
    model: fsr.SimDriveHot,
    path: Union[str, list],
) -> list:
    """
    Get all attributes containing another attribute from `model` using `path` to attribute
    """
    if isinstance(path, str):
        path = path.split(".")
    containers = [getattr(model, path[0])]
    for subpath in path[1:-1]:
        container = getattr(containers[-1], subpath)
        containers.append(container)
    return containers


def get_attr_from_path(
    model: fsr.SimDriveHot,
    path: Union[str, list]
) -> Any:
    """
    Get attribute from `model` using `path` to attribute
    """
    if isinstance(path, str):
        path = path.split(".")
    containers = get_containers_from_path(model, path)
    attr = getattr(containers[-1], path[-1])
    return attr


def set_attr_from_path(
    model: fsr.SimDriveHot,
    path: Union[str, list],
    value: float
) -> fsr.SimDriveHot:
    """
    Set attribute `value` on `model` for `path` to attribute
    """
    # TODO: Does this actually work?
    if isinstance(path, str):
        path = path.split(".")
    containers = get_containers_from_path(model, path)
    containers[-1].reset_orphaned()
    setattr(containers[-1], path[-1], value)
    return model

    """containers = [model]
    for step in path[:-1]:
        containers.append(containers[-1].__getattribute__(step))

    # zip it back up
    # innermost container first
    containers[-1].reset_orphaned()
    containers[-1].__setattr__(
        path[-1], value
    )

    prev_container = containers[-1]

    # iterate through remaining containers, inner to outer
    for i, (container, path_elem) in enumerate(zip(containers[-2::-1], path[-2::-1])):
        if i < len(containers) - 2:
            # reset orphaned for everything but the outermost container
            container.reset_orphaned()
        setattr(
            container,
            path_elem,
            prev_container
        )
        prev_container = container

    return model"""


@dataclass
class ModelErrors(object):
    """
    Class for calculating eco-driving objectives
    """

    # dictionary of models to be simulated
    sim_drives: Dict[str, fsim.simdrive.SimDrive]

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

    objectives: List[Tuple[str, Optional[Tuple[str]]]]

    # list of tuples hierarchical paths to parameters to manipulate
    # example - `[('sim_drive', 'sim_params', 'idm_accel_m_per_s2')]`
    params: List[Tuple[str]]

    # if True, prints timing and misc info
    verbose: bool = False

    def __post_init__(self):
        assert(len(self.dfs) == len(self.sim_drives))

    def get_errors(
        self, return_mods: Optional[bool] = False, plot: Optional[bool] = False,
            plot_save_dir: Optional[str] = None, plot_perc_err: Optional[bool] = True,
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
        for ((key, df_exp), sim_drive) in zip(self.dfs.items(), self.sim_drives.values()):
            t0 = time.time()
            sim_drive = sim_drive.copy()
            sim_drive.sim_drive()
            objectives[key] = {}
            if return_mods or plot:
                solved_mods[key] = sim_drive.copy()

            if plot or plot_save_dir:
                time_hr = np.array(sim_drive.sd.cyc.time_s) / 3_600
                ax_multiplier = 2 if plot_perc_err else 1
                fig, ax = plt.subplots(
                    len(self.objectives) * ax_multiplier + 1, 1, sharex=True, figsize=(12, 8),
                )
                ax[-1].plot(
                    time_hr,
                    sim_drive.sd.mph_ach,
                )
                ax[-1].set_xlabel('Time [hr]')
                ax[-1].set_ylabel('Speed [mph]')

            t1 = time.time()
            if self.verbose:
                print(f"Time to simulate {key}: {t1 - t0:.3g}")

            # loop through the objectives for each trip
            for i_obj, obj in enumerate(self.objectives):
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
                mod_sig = get_attr_from_path(sim_drive, mod_path)

                if ref_path:
                    ref_sig = df_exp[ref_path]

                    # print(mod_path, mod_sig)
                    # print(ref_path, ref_sig)

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

            t2 = time.time()
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
        assert(len(xs) == len(self.params))
        paths = [fullpath.split(".") for fullpath in self.params]

        t0 = time.time()
        # first element of each tuple in self.params should be identical
        mod = list(self.sim_drives.values())[0].__getattribute__(
            paths[0][0]
        )

        # the idea of this code is to step through the hierarchy
        # to collect the different levels in a list, reset orphaned, set
        # new values, and then propagate back up the hierarchy.
        # I'm not convinced it works, but the concept seems sound.  Only the implementation
        # details might be sketchy.
        for path, x in zip(paths, xs):
            # containers = [mod]
            containers = [
                self.sim_drives[list(self.sim_drives.keys())[0]].__getattribute__(
                    path[0]
                )
            ]

            for step in path[1:-1]:
                containers.append(containers[-1].__getattribute__(step))
            # zip it back up
            containers[-1].reset_orphaned()
            containers[-1].__setattr__(
                path[-1], x
            )
            for i, container in enumerate(list(reversed(containers))[1:]):
                container.reset_orphaned()  # seems like this should be necessary
                containers[-2-i].__setattr__(
                    path[-2-i],
                    containers[-1-i]
                )

            for key in self.sim_drives.keys():
                self.sim_drives[key].__setattr__(
                    paths[0][0],
                    containers[0]
                )
        t1 = time.time()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")


@dataclass
class CalibrationProblem(ElementwiseProblem):
    """
    Problem for calibrating models to match test data
    """

    # default of None is needed for dataclass inheritance
    # this is actually mandatory!
    err: ModelErrors = None
    # parameter lower and upper bounds
    # default of None is needed for dataclass inheritance
    # this is actually mandatory!
    param_bounds: List[Tuple[float, float]] = None

    def __post_init__(self):
        assert(len(self.param_bounds) == len(self.err.params))
        super().__init__(
            n_var=len(self.err.params),
            n_obj=len(self.err.sim_drives) * len(self.err.objectives),
            xl=[bounds[0]
                for bounds in self.param_bounds],
            xu=[bounds[1]
                for bounds in self.param_bounds],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        self.err.update_params(x)
        out['F'] = [
            val for inner_dict in self.err.get_errors().values() for val in inner_dict.values()
        ]


def run_minimize(
    prob: CalibrationProblem,
    algorithm: GeneticAlgorithm = NSGA2(
        # size of each population
        pop_size=10,
        eliminate_duplicates=True,
        sampling=LHS(),
    ),
    termination: MODT = MODT(
        # max number of generations, default of 10 is very small
        n_max_gen=10,
        # evaluate tolerance over this interval of generations every `nth_gen`
        n_last=10,
    ),
    copy_algorithm: bool = False,
    copy_termination: bool = False,
    save_history: bool = False,
    save_path: Optional[str] = "pymoo_res",
):
    res = minimize(
        prob,
        algorithm,
        termination,
        copy_algorithm,
        copy_termination,
        seed=1,
        verbose=True,
        save_history=save_history,
    )

    f_columns = [
        f"{key}: {obj[0]}"
        for key in prob.err.dfs.keys()
        for obj in prob.err.objectives
    ]
    f_df = pd.DataFrame(
        data=[f for f in res.F.tolist()],
        columns=f_columns,
    )

    x_df = pd.DataFrame(
        data=[x for x in res.X.tolist()],
        columns=[param for param in prob.err.params],
    )

    res_df = pd.concat([x_df, f_df], axis=1)
    res_df.to_csv(str(save_path) + "_df.csv", index=False)

    return res, res_df
