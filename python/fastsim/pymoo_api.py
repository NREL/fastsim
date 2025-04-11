"""
Module containing functions and classes for easy interaction with PyMOO
"""

import numpy as np
import pprint  # noqa: F401
import numpy.typing as npt
from typing import Tuple, Any, List, Callable, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import argparse
import time
from dataclasses import dataclass

# pymoo
try:
    from pymoo.optimize import minimize
    from pymoo.core.result import Result
    from pymoo.termination.default import DefaultMultiObjectiveTermination as DMOT
    from pymoo.core.problem import ElementwiseProblem, LoopedElementwiseEvaluation
    from pymoo.algorithms.base.genetic import GeneticAlgorithm
    from pymoo.util.display.output import Output
    from pymoo.util.display.column import Column

    # Imports for convenient use in scripts
    from pymoo.core.problem import StarmapParallelization  # noqa: F401
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS  # noqa: F401
    from pymoo.algorithms.moo.nsga3 import NSGA3  # noqa: F401
    from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: F401
    from pymoo.util.ref_dirs import get_reference_directions  # noqa: F401

    PYMOO_AVAILABLE = True
except ModuleNotFoundError as err:
    print(
        f"{err}\nTry running `pip install pymoo==0.6.0.1` to use all features in "
        + "`fastsim.calibration`"
    )
    PYMOO_AVAILABLE = False

import fastsim as fsim


def get_error_val(
    model: npt.NDArray[np.float64],
    test: npt.NDArray[np.float64],
    time_steps: npt.NDArray[np.float64],
) -> float:
    """
    Returns time-averaged error for model and test signal.

    # Args:
        - `model`: array of values for signal from model
        - `test`: array of values for signal from test data
        - `time_steps`: array (or scalar for constant) of values for model time steps [s]

    # Returns:
        - `error`: integral of absolute value of difference between model and test per time
    """
    assert len(model) == len(test) == len(time_steps), (
        f"{len(model)}, {len(test)}, {len(time_steps)}"
    )

    return np.trapz(y=abs(model - test), x=time_steps) / (time_steps[-1] - time_steps[0])


@dataclass
class ModelObjectives(object):
    """
    Class for calculating eco-driving objectives

    # Attributes/Fields
    - `models` (Dict[str, Dict]): dictionary of model dicts to be simulated
    - `dfs` (Dict[str, pd.DataFrame]): dictionary of dataframes from test data
      corresponding to `models`
    - `obj_fns` (Tuple[Callable] | Tuple[Tuple[Callable, Callable]]):
      Tuple of functions for extracting objective signal values for either minimizing a
      scalar metric (e.g. fuel economy) or minimizing error relative to test
      data.
        - minimizing error in fuel consumption relative to test data
          ```
          obj_fns = (
              (
                  # model -- note, `lambda` only works for single thread
                  lambda sd_dict: sd_dict['veh']['pt_type']['Conventional']['fc']['history']['energy_fuel_joules'],
                  # test data
                  lambda df: df['fuel_flow_gps'] * ... (conversion factors to get to same unit),
              ), # note that this trailing comma ensures `obj_fns` is interpreted as a tuple
          )
          ```
        - minimizing fuel consumption
          ```
          obj_fns = (
              (
                  # note that a trailing comma ensures `obj_fns` is interpreted as tuple
                  lambda sd_dict: sd_dict['veh']['pt_type']['Conventional']['fc']['state']['energy_fuel_joules'],
              )
          )
          ```
    - `param_fns_and_bounds` Tuple[Tuple[Callable], Tuple[Tuple[float, float]]]:
      tuple containing functions to modify parameters and bounds for optimizer
      Example
      ```
      def new_peak_res_eff (sd_dict, new_peak_eff) -> Dict:
          sd_dict['veh']['pt_type']['HybridElectricVehicle']['res']['peak_eff'] = new_peak_eff
          return sd_dict
      ...
      param_fns_and_bounds = (
          (new_peak_res_eff, (100.0, 200.0)),
      )
      ```
    - `verbose` (bool): print more stuff or not
    """

    models: Dict[str, Dict]
    dfs: Dict[str, pd.DataFrame]
    obj_fns: Tuple[Callable] | Tuple[Tuple[Callable, Callable]]
    constr_fns: Tuple[Callable]
    param_fns_and_bounds: Tuple[Tuple[Callable], Tuple[Tuple[float, float]]]
    param_fns: Optional[Tuple[Callable]] = None
    bounds: Optional[Tuple[Tuple[float, float]]] = None

    # if True, prints timing and misc info
    verbose: bool = False

    # calculated in __post_init__
    n_obj: Optional[int] = None
    n_constr: Optional[int] = None

    def __post_init__(self):
        assert self.n_obj is None, "`n_obj` is not intended to be user provided"
        assert len(self.dfs) == len(self.models), f"{len(self.dfs)} != {len(self.models)}"
        self.param_fns: Tuple[Callable] = tuple([pb[0] for pb in self.param_fns_and_bounds])  # type: ignore[annotation-unchecked]
        self.bounds: Tuple[Tuple[float, float]] = tuple([pb[1] for pb in self.param_fns_and_bounds])  # type: ignore[annotation-unchecked]
        assert len(self.bounds) == len(self.param_fns)
        self.n_obj = len(self.models) * len(self.obj_fns)
        self.n_constr = len(self.models) * len(self.constr_fns)

    def update_params(self, xs: List[Any]):
        """
        Updates model parameters based on `x`, which must match length of self.param_fns
        """
        assert len(xs) == len(self.param_fns), f"({len(xs)} != {len(self.param_fns)}"

        t0 = time.perf_counter()

        # Instantiate SimDrive objects
        sim_drives = {}

        # Update all model parameters
        for key, pydict in self.models.items():
            for param_fn, new_val in zip(self.param_fns, xs):
                pydict = param_fn(pydict, new_val)
            # this assignement may be redundant, but `pydict` is probably **not** mutably modified.
            # If this is correct, then this assignment is necessary
            self.models[key] = pydict

        for key, pydict in self.models.items():
            try:
                sim_drives[key] = fsim.SimDrive.from_pydict(pydict, skip_init=False)
            except Exception as err:
                sim_drives[key] = err
        t1 = time.perf_counter()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")
        return sim_drives

    def get_errors(
        self,
        sim_drives: Dict[str, fsim.SimDrive],
        return_mods: bool = False,
    ) -> Union[
        Dict[str, Dict[str, float]],
        # or if return_mods is True
        Tuple[Dict[str, fsim.SimDrive], Dict[str, Dict[str, float]]],
    ]:
        """
        Calculate model errors w.r.t. test data for each element in dfs/models for each objective.

        # Args:
            - `sim_drives`: dictionary with user-defined keys and SimDrive instances
            - `return_mods`: if true, also returns dict of solved models. Defaults to False.

        # Returns:
            Objectives and optionally solved models
        """

        objectives: Dict = {}
        constraint_violations: Dict = {}
        solved_mods: Dict = {}
        unsolved_mods: Dict = {}

        # loop through all the provided trips
        for (key, df_exp), sd in zip(self.dfs.items(), sim_drives.values()):
            key: str
            df_exp: pd.DataFrame

            # if not isinstance(sd, fsim.SimDrive):
            #     solved_mods[key] = sd
            #     objectives[key] = [1.0e12] * len(self.obj_fns)
            #     continue

            if return_mods:
                unsolved_mods[key] = sd.to_pydict()

            try:
                t0 = time.perf_counter()
                sd.walk_once()  # type: ignore
                t1 = time.perf_counter()
                sd_dict = sd.to_pydict()
                walk_success = True
            except RuntimeError as err:
                t1 = time.perf_counter()
                sd_dict = sd.to_pydict()
                walk_success = True
                print(err)
                if len(sd_dict["veh"]["history"]["time_seconds"]) < np.floor(len(df_exp) / 2):
                    walk_success = False

            if self.verbose:
                print(f"Time to simulate {key}: {t1 - t0:.3g}")

            objectives[key] = []
            constraint_violations[key] = []

            if return_mods:
                solved_mods[key] = sd_dict

            # trim dataframe to match length of cycle completed by vehicle
            df_exp = df_exp[: len(sd_dict["veh"]["history"]["time_seconds"])]

            # loop through the objectives for each trip
            for i_obj, obj_fn in enumerate(self.obj_fns):
                i_obj: int
                obj_fn: Tuple[Callable(sd_dict), Callable(df_exp)]
                if len(obj_fn) == 2:
                    # objective and reference passed
                    mod_sig = obj_fn[0](sd_dict)
                    ref_sig = obj_fn[1](df_exp)
                elif len(obj_fn) == 1:
                    # minimizing scalar objective
                    mod_sig = obj_fn[0](sd_dict)
                    ref_sig = None
                else:
                    raise ValueError("Each element in `self.obj_fns` must have length of 1 or 2")

                if ref_sig is not None:
                    time_s = sd_dict["veh"]["history"]["time_seconds"]
                    # TODO: provision for incomplete simulation in here somewhere

                    if not walk_success:
                        objectives[key].append(1.02e12)
                    else:
                        try:
                            objectives[key].append(
                                get_error_val(
                                    mod_sig,
                                    ref_sig,
                                    time_s,
                                )
                            )
                        except AssertionError:
                            # `get_error_val` checks for length equality with an assertion
                            # If length equality is not satisfied, this design is
                            # invalid because the cycle could not be completed.
                            # NOTE: instead of appending an arbitrarily large
                            # objective value, we could instead either try passing
                            # `np.nan` or trigger a constraint violation.
                            objectives[key].append(1.03e12)
                else:
                    raise Exception("this is here for debugging and should be deleted")
                    objectives[key].append(mod_sig)
            for constr_fn in self.constr_fns:
                constraint_violations[key].append(constr_fn(sd_dict))

            t2 = time.perf_counter()
            if self.verbose:
                print(f"Time to postprocess: {t2 - t1:.3g} s")
        # print("\nobjectives:")
        # pprint.pp(objectives)
        # print("")
        if return_mods:
            return objectives, constraint_violations, solved_mods, unsolved_mods
        else:
            return objectives, constraint_violations


if PYMOO_AVAILABLE:

    class CalibrationProblem(ElementwiseProblem):
        """
        Problem for calibrating models to match test data
        """

        def __init__(
            self,
            mod_obj: ModelObjectives,
            elementwise_runner=LoopedElementwiseEvaluation(),
        ):
            self.mod_obj = mod_obj
            super().__init__(
                n_var=len(self.mod_obj.param_fns),
                n_obj=self.mod_obj.n_obj,
                xl=[bounds[0] for bounds in self.mod_obj.bounds],
                xu=[bounds[1] for bounds in self.mod_obj.bounds],
                elementwise_runner=elementwise_runner,
                n_ieq_constr=self.mod_obj.n_constr,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            sim_drives = self.mod_obj.update_params(x)
            (errs, cvs) = self.mod_obj.get_errors(sim_drives)
            out["F"] = list(errs.values())
            if self.n_ieq_constr > 0:
                out["G"] = list(cvs.values())

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
            f = algorithm.pop.get("F")
            euclid_min = np.sqrt((np.array(f) ** 2).sum(axis=1)).min()
            self.euclid_min.set(f"{euclid_min:.3g}")

    def run_minimize(
        problem: CalibrationProblem,
        algorithm: GeneticAlgorithm,
        termination: DMOT,
        save_path: Union[Path, str],
        copy_algorithm: bool = False,
        copy_termination: bool = False,
        save_history: bool = False,
    ) -> Tuple[Result, pd.DataFrame]:
        """
        Wrapper for pymoo.optimize.minimize that adds various helpful features
        """
        print("`run_minimize` starting at")
        fsim.utils.print_dt()

        t0 = time.perf_counter()
        res = minimize(
            problem,
            algorithm,
            termination,
            copy_algorithm=copy_algorithm,
            copy_termination=copy_termination,
            seed=1,
            verbose=True,
            save_history=save_history,
            output=CustomOutput(),
        )

        f_columns = [
            f"{key.split(' ')[0]}: {obj[0].__name__}"
            for key in problem.mod_obj.dfs.keys()
            for obj in problem.mod_obj.obj_fns
        ]
        f_df = pd.DataFrame(
            data=[f for f in res.F.tolist()],
            columns=f_columns,
        )

        x_df = pd.DataFrame(
            data=[x for x in res.X.tolist()],
            columns=[param.__name__ for param in problem.mod_obj.param_fns],
        )

        Path(save_path).mkdir(exist_ok=True, parents=True)

        res_df = pd.concat([x_df, f_df], axis=1)
        res_df["euclidean"] = (
            (res_df.iloc[:, len(problem.mod_obj.param_fns) :] ** 2).sum(1).pow(1 / 2)
        )
        res_df.to_csv(Path(save_path) / "pymoo_res_df.csv", index=False)

        t1 = time.perf_counter()
        print(f"Elapsed time to run minimization: {t1 - t0:.5g} s")

        return res, res_df


def get_parser(
    def_description: str = "Program for calibrating fastsim models.",
    def_p: int = 4,
    def_n_max_gen: int = 500,
    def_pop_size: int = 12,
) -> argparse.ArgumentParser:
    """
    Generate parser for optimization hyper params and misc. other params

    # Args:
        - `def_p`: default number of processes
        - `def_n_max_gen`: max allowed generations
        - `def_pop_size`: default population size

    # Returns:
        argparse.ArgumentParser: _description_
    """
    parser = argparse.ArgumentParser(description=def_description)
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=def_p,
        help=f"Number of pool processes. Defaults to {def_p}",
    )
    parser.add_argument(
        "--n-max-gen",
        type=int,
        default=def_n_max_gen,
        help=f"PyMOO termination criterion: n_max_gen. Defaults to {def_n_max_gen}",
    )
    parser.add_argument(
        "--xtol",
        type=float,
        default=DMOT().x.termination.tol,
        help=f"PyMOO termination criterion: xtol. Defaluts to {DMOT().x.termination.tol}",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=DMOT().f.termination.tol,
        help=f"PyMOO termination criterion: ftol. Defaults to {DMOT().f.termination.tol}",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=def_pop_size,
        help=f"PyMOO population size in each generation. Defaults to {def_pop_size}",
    )
    parser.add_argument(
        "--skip-minimize", action="store_true", help="If provided, load previous results."
    )
    # parser.add_argument(
    #     '--show',
    #     action="store_true",
    #     help="If provided, shows plots."
    # )
    # parser.add_argument(
    #     "--make-plots",
    #     action="store_true",
    #     help="Generates plots, if provided."
    # )

    return parser
