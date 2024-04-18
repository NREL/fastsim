from __future__ import annotations
from dataclasses import dataclass, InitVar
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pickle
import argparse
import pandas as pd
import numpy as np
import numpy.typing as npt
import json
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
# Logging
import logging
logger = logging.getLogger(__name__)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False

# pymoo
try:
    from pymoo.util.display.output import Output 
    from pymoo.util.display.column import Column
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
    from pymoo.termination.default import DefaultMultiObjectiveTermination as DMOT
    from pymoo.core.problem import Problem, ElementwiseProblem, LoopedElementwiseEvaluation, StarmapParallelization
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.algorithms.base.genetic import GeneticAlgorithm
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except ModuleNotFoundError as err:
    logger.warning(
        f"{err}\nTry running `pip install pymoo==0.6.0.1` to use all features in " + 
        "`fastsim.calibration`"
    )
    PYMOO_AVAILABLE = False

import numpy as np

# local -- expects rust-port version of fastsim so make sure this is in the env
import fastsim as fsim
import fastsim.fastsimrust as fsr


def get_error_val(
    model: npt.NDArray[np.float64], 
    test: npt.NDArray[np.float64], 
    time_steps: npt.NDArray[np.float64]
) -> float:
    """
    Returns time-averaged error for model and test signal.

    Args:
        model (npt.NDArray[np.float64]): array of values for signal from model
        test (npt.NDArray[np.float64]): array of values for signal from test data
        time_steps (npt.NDArray[np.float64]): array (or scalar for constant) of values for model time steps [s]

    Returns:
        float: integral of absolute value of difference between model and test per time
    """
    assert len(model) == len(test) == len(
        time_steps), f"{len(model)}, {len(test)}, {len(time_steps)}"

    return np.trapz(y=abs(model - test), x=time_steps) / (time_steps[-1] - time_steps[0])

@dataclass
class ModelObjectives(object):
    """
    Class for calculating eco-driving objectives
    """

    # dictionary of json models to be simulated
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
    n_obj: Optional[int] = None

    # whether to use simdrive hot
    # TOOD: consider passing the type to be used rather than this boolean in the future
    use_simdrivehot: bool = False

    def __post_init__(self):
        assert len(self.dfs) == len(
            self.models), f"{len(self.dfs)} != {len(self.models)}"
        self.n_obj = len(self.models) * len(self.obj_names)

    def get_errors(
        self,
        sim_drives: Dict[str, fsr.RustSimDrive | fsr.SimDriveHot],
        return_mods: bool = False,
        plot: bool = False,
        plot_save_dir: Optional[str] = None,
        plot_perc_err: bool = False,
        show: bool = False,
        fontsize: float = 12,
        plotly: bool = False,
    ) -> Union[
        Dict[str, Dict[str, float]],
        # or if return_mods is True
        Tuple[Dict[str, fsim.simdrive.SimDrive], Dict[str, Dict[str, float]]]
    ]:
        """
        Calculate model errors w.r.t. test data for each element in dfs/models for each objective.

        Args:
            sim_drives (Dict[str, fsr.RustSimDrive  |  fsr.SimDriveHot]): dictionary with user-defined keys and SimDrive or SimDriveHot instances
            return_mods (bool, optional): if true, also returns dict of solved models. Defaults to False.
            plot (bool, optional): if true, plots objectives using matplotlib.pyplot. Defaults to False.
            plot_save_dir (Optional[str], optional): directory in which to save plots. If None, plots are not saved. Defaults to None.
            plot_perc_err (bool, optional): whether to include % error axes in plots. Defaults to False.
            show (bool, optional): whether to show matplotlib.pyplot plots. Defaults to False.
            fontsize (float, optional): plot font size. Defaults to 12.
            plotly (bool, optional): whether to generate plotly plots, which can be opened manually in a browser window. Defaults to False.

        Returns:
            Objectives and optionally solved models
        """
        # TODO: should return type instead be `Dict[str, Dict[str, float]] | Tuple[Dict[str, Dict[str, float]], Dict[str, fsim.simdrive.SimDrive]]`
        # This would make `from typing import Union` unnecessary

        objectives: Dict = {}
        solved_mods: Dict = {}

        if plotly:
            assert PLOTLY_AVAILABLE, "Package `plotly` not installed." + \
                "Run `pip install plotly`."


        # loop through all the provided trips
        for ((key, df_exp), sim_drive) in zip(self.dfs.items(), sim_drives.values()):
            t0 = time.perf_counter()
            sim_drive = sim_drive.copy()  # TODO: do we need this?
            sim_drive.sim_drive() # type: ignore
            t1 = time.perf_counter()
            if self.verbose:
                print(f"Time to simulate {key}: {t1 - t0:.3g}")
            objectives[key] = {}
            if return_mods or plot:
                solved_mods[key] = sim_drive.copy()

            ax_multiplier = 2 if plot_perc_err else 1
            # extract speed trace for plotting
            if not self.use_simdrivehot:
                time_hr = np.array(sim_drive.cyc.time_s) / 3_600 # type: ignore
                mph_ach = sim_drive.mph_ach # type: ignore
            else:
                time_hr = np.array(sim_drive.sd.cyc.time_s) / 3_600 # type: ignore
                mph_ach = sim_drive.sd.mph_ach # type: ignore
            fig, ax, pltly_fig = self.setup_plots(
                plot or show,
                plot_save_dir,
                sim_drive,
                fontsize,
                key,
                ax_multiplier,
                time_hr,
                mph_ach,
                plotly,
            )                

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
                    if not self.use_simdrivehot:
                        time_s = sim_drive.cyc.time_s
                    else:
                        time_s = sim_drive.sd.cyc.time_s
                    objectives[key][obj[0]] = get_error_val(
                        mod_sig,
                        ref_sig,
                        time_s,
                    )
                else:
                    pass
                    # TODO: write else block for objective minimization
                
                update_plots(
                    ax,
                    pltly_fig,
                    i_obj,
                    ax_multiplier,
                    objectives,
                    key,
                    obj,
                    fontsize,
                    time_hr,
                    mod_sig,
                    ref_sig,
                    plot_perc_err,
                    mod_path,
                    show,
                )    
            
            if plot_save_dir is not None:
                if not Path(plot_save_dir).exists():
                    Path(plot_save_dir).mkdir(exist_ok=True, parents=True)
                if ax is not None:
                    plt.savefig(Path(plot_save_dir) / f"{key}.svg")
                    plt.savefig(Path(plot_save_dir) / f"{key}.png")
                    plt.tight_layout()
                if pltly_fig is not None:
                    pltly_fig.update_layout(showlegend=True)
                    pltly_fig.write_html(str(Path(plot_save_dir) / f"{key}.html"))

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
        # Load instances from json strings
        if not self.use_simdrivehot:
            sim_drives = {key: fsr.RustSimDrive.from_json(
                model_json) for key, model_json in self.models.items()}
        else:
            sim_drives = {key: fsr.SimDriveHot.from_json(
                model_json) for key, model_json in self.models.items()}
        # Update all model parameters
        for key in sim_drives.keys():
            sim_drives[key] = fsim.utils.set_attrs_with_path(
                sim_drives[key],
                dict(zip(self.params, xs)),
            )
            if not self.use_simdrivehot:
                veh = sim_drives[key].veh
                veh.set_derived()
                sim_drives[key].veh = veh
            else:
                veh = sim_drives[key].sd.veh
                veh.set_derived()
                fsim.utils.set_attr_with_path(sim_drives[key], "sd.veh", veh)
        t1 = time.perf_counter()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")
        return sim_drives
    
    def setup_plots(
        self,
        plot: bool, 
        plot_save_dir: Optional[Path],
        sim_drive: Union[fsr.RustSimDrive, fsr.SimDriveHot],
        fontsize: float,
        key: str,
        ax_multiplier: int,
        time_hr: float,
        mph_ach: float,
        plotly: bool,
    ) -> Tuple[Figure, Axes, go.Figure]:
        rows = len(self.obj_names) * ax_multiplier + 1

        if plotly and (plot_save_dir is not None):
            pltly_fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
            )
            pltly_fig.update_layout(title=f'trip: {key}')
            pltly_fig.add_trace(
                go.Scatter(
                    x=time_hr,
                    y=mph_ach,
                    name="mph",
                ),
                row=rows,
                col=1,
            )
            pltly_fig.update_xaxes(title_text="Time [hr]", row=rows, col=1)
            pltly_fig.update_yaxes(title_text="Speed [mph]", row=rows, col=1)
        elif plotly:
            raise Exception("`plot_save_dir` must also be provided for `plotly` to have any effect.")
        else:
            pltly_fig = None

        if plot:
            # make directory if it doesn't exist
            Path(plot_save_dir).mkdir(exist_ok=True, parents=True)
            fig, ax = plt.subplots(
                len(self.obj_names) * ax_multiplier + 1, 1, sharex=True, figsize=(12, 8),
            )
            plt.suptitle(f"trip: {key}", fontsize=fontsize)
            ax[-1].plot(
                time_hr,
                mph_ach,
            )
            ax[-1].set_xlabel('Time [hr]', fontsize=fontsize)
            ax[-1].set_ylabel('Speed [mph]', fontsize=fontsize)
            return fig, ax, pltly_fig
        else:
            return (None, None, None)

def update_plots(
    ax: Optional[Axes],
    pltly_fig: go.Figure,
    i_obj: int,
    ax_multiplier: int,
    objectives: Dict,
    key: str,
    obj: Any,  # need to check type on this
    fontsize: int,
    time_hr: np.ndarray,
    mod_sig: np.ndarray,
    ref_sig: np.ndarray,
    plot_perc_err: bool,
    mod_path: str,
    show: bool,
):
    if ax is not None:
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
    
    if pltly_fig is not None:
        pltly_fig.add_trace(
            go.Scatter(
                x=time_hr,
                y=mod_sig,
                # might want to prepend signal name for this
                name=obj[0] + ' mod',
            ),
            # add 1 for 1-based indexing in plotly
            row=i_obj * ax_multiplier + 1,
            col=1,
        )
        pltly_fig.add_trace(
            go.Scatter(
                x=time_hr,
                y=ref_sig,
                # might want to prepend signal name for this
                name=obj[0] + ' exp',
            ),
            # add 1 for 1-based indexing in plotly
            row=i_obj * ax_multiplier + 1,
            col=1,
        )
        pltly_fig.update_yaxes(title_text=obj[1], row=i_obj * ax_multiplier + 1, col=1)

        if plot_perc_err:
            pltly_fig.add_trace(
                go.Scatter(
                    x=time_hr,
                    y=perc_err,
                    # might want to prepend signal name for this
                    name=obj[0] + ' % err',
                ),
                # add 2 for 1-based indexing and offset for % err plot
                row=i_obj * ax_multiplier + 2,
                col=1,
            )            
            # pltly_fig.update_yaxes(title_text=obj[0] + "%Err", row=i_obj * ax_multiplier + 2, col=1)
            pltly_fig.update_yaxes(title_text="%Err", row=i_obj * ax_multiplier + 2, col=1)

if PYMOO_AVAILABLE:
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


if PYMOO_AVAILABLE:
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


if PYMOO_AVAILABLE:
    def run_minimize(
        problem: CalibrationProblem,
        algorithm: GeneticAlgorithm,
        termination: DMOT,
        copy_algorithm: bool = False,
        copy_termination: bool = False,
        save_history: bool = False,
        save_path: Union[Path, str] = Path("pymoo_res/"),
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

        if save_path is not None:
            Path(save_path).mkdir(exist_ok=True, parents=True)

        res_df = pd.concat([x_df, f_df], axis=1)
        res_df['euclidean'] = (
            res_df.iloc[:, len(problem.mod_obj.params):] ** 2).sum(1).pow(1/2)
        if save_path is not None:
            res_df.to_csv(Path(save_path) / "pymoo_res_df.csv", index=False)

        t1 = time.perf_counter()
        print(f"Elapsed time to run minimization: {t1-t0:.5g} s")

        return res, res_df


def get_parser(
    def_description:str="Program for calibrating fastsim models.",
    def_p:int=4,
    def_n_max_gen:int=500,
    def_pop_size:int=12,
    def_save_path:Optional[str]="pymoo_res"

) -> argparse.ArgumentParser:
    """
    Generate parser for optimization hyper params and misc. other params

    Args:
        def_p (int, optional): default number of processes. Defaults to 4.
        def_n_max_gen (int, optional): max allowed generations. Defaults to 500.
        def_pop_size (int, optional): default population size. Defaults to 12.
        def_save_path (str, optional): default save path.  Defaults to `pymoo_res`.

    Returns:
        argparse.ArgumentParser: _description_
    """
    parser = argparse.ArgumentParser(description=def_description)
    parser.add_argument(
        '-p', 
        '--processes', 
        type=int,
        default=def_p, 
        help=f"Number of pool processes. Defaults to {def_p}"
    )
    parser.add_argument(
        '--n-max-gen', 
        type=int, 
        default=def_n_max_gen,
        help=f"PyMOO termination criterion: n_max_gen. Defaults to {def_n_max_gen}"
    )
    parser.add_argument(
        '--xtol',
        type=float,
        default=DMOT().x.termination.tol,
        help=f"PyMOO termination criterion: xtol. Defaluts to {DMOT().x.termination.tol}"
    )
    parser.add_argument(
        '--ftol',
        type=float,
        default=DMOT().f.termination.tol,
        help=f"PyMOO termination criterion: ftol. Defaults to {DMOT().f.termination.tol}"
    )
    parser.add_argument(
        '--pop-size', 
        type=int, 
        default=def_pop_size,
        help=f"PyMOO population size in each generation. Defaults to {def_pop_size}"
    )
    parser.add_argument(
        '--skip-minimize', 
        action="store_true",
        help="If provided, load previous results."
    )
    parser.add_argument(
        '--save-path', 
        type=str, 
        default=def_save_path,               
        help="File location to save results dataframe with rows of parameter and corresponding" 
            + " objective values and any optional plots." 
            + (" If not provided, results are not saved" if def_save_path is None else "")
    )
    parser.add_argument(
        '--show', 
        action="store_true",
        help="If provided, shows plots."
    )
    parser.add_argument(
        "--make-plots", 
        action="store_true",
        help="Generates plots, if provided."
    )
    parser.add_argument(
        "--use-simdrivehot", 
        action="store_true",
        help="Use fsr.SimDriveHot rather than fsr.RustSimDrive."
    )
    
    return parser
