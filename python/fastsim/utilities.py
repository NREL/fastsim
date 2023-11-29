"""Various optional utilities that may support some applications of FASTSim."""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import fastsim as fsim
from fastsim import parameters as params
import seaborn as sns
import datetime
import logging
from pathlib import Path
import re
from contextlib import contextmanager
import os
from pkg_resources import get_distribution
import pathlib
import shutil

sns.set()

props = params.PhysicalProperties()
R_air = 287  # J/(kg*K)


def get_rho_air(temperature_degC, elevation_m=180):
    """Returns air density [kg/m**3] for given elevation and temperature.
    Source: https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html
    Arguments:
    ----------
    temperature_degC : ambient temperature [°C]
    elevation_m : elevation above sea level [m].
        Default 180 m is for Chicago, IL"""
    #     T = 15.04 - .00649 * h
    #     p = 101.29 * [(T + 273.1)/288.08]^5.256
    T_standard = 15.04 - 0.00649 * elevation_m  # nasa [degC]
    p = 101.29e3 * ((T_standard + 273.1) / 288.08) ** 5.256  # nasa [Pa]
    rho = p / (R_air * (temperature_degC + 273.15))  # [kg/m**3]

    return rho


def l__100km_to_mpg(l__100km):
    """Given fuel economy in L/100km, returns mpg."""

    mpg = 1 / (l__100km / 3.785 / 100 / 1_000 * params.M_PER_MI)

    return mpg


def mpg_to_l__100km(mpg):
    """Given fuel economy in mpg, returns L/100km."""

    l__100km = 1 / (mpg / 3.785 * params.M_PER_MI / 1_000 / 100)

    return l__100km


def rollav(x, y, width=10):
    """
    Returns x-weighted backward-looking rolling average of y.
    Good for resampling data that needs to preserve cumulative information.
    Arguments:
    ----------
    x : x data
    y : y data (`len(y) == len(x)` must be True)
    width: rolling average width
    """

    assert(len(x) == len(y))

    dx = np.concatenate([0, x.diff()])

    yroll = np.zeros(len(x))
    yroll[0] = y[0]

    for i in range(1, len(x)):
        if i < width:
            yroll[i] = (
                dx[:i] * y[:i]).sum() / (x[i] - x[0])
        else:
            yroll[i] = (
                dx[i-width:i] * y[i-width:i]).sum() / (
                    x[i] - x[i-width])
    return yroll


def camel_to_snake(name):
    "Given camelCase, returns snake_case."
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def set_log_level(level: str | int) -> int:
    """
    Sets logging level for both Python and Rust FASTSim.
    The default logging level is WARNING (30).
    https://docs.python.org/3/library/logging.html#logging-levels

    Parameters
    ----------
    level: `str` | `int`
        Logging level to set. `str` level name or `int` logging level
        
        =========== ================
        Level       Numeric value
        =========== ================
        CRITICAL    50
        ERROR       40
        WARNING     30
        INFO        20
        DEBUG       10
        NOTSET      0
    
    Returns
    -------
    `int`
        Previous log level
    """
    # Map string name to logging level
    if isinstance(level, str):
        level = logging._nameToLevel[level]
    # Extract previous log level and set new log level
    fastsim_logger = logging.getLogger("fastsim")
    previous_level = fastsim_logger.level
    fastsim_logger.setLevel(level)
    fastsimrust_logger = logging.getLogger("fastsim_core")
    fastsimrust_logger.setLevel(level)
    return previous_level

def disable_logging() -> int:
    """
    Disable FASTSim logs from being shown by setting log level
    to CRITICAL+1 (51).
    
    Returns
    -------
    `int`
        Previous log level
    """
    return set_log_level(logging.CRITICAL + 1)

def enable_logging(level: Optional[int | str] = None):
    """
    Re-enable FASTSim logging, optionally to a specified log level,
    otherwise to the default WARNING (30) level.

    Parameters
    ----------
    level: `str` | `int`, optional
        Logging level to set. `str` level name or `int` logging level.
        See `utils.set_log_level()` docstring for more details on logging levels.
    """
    if level is None:
        level = logging.WARNING
    set_log_level(level)

@contextmanager
def suppress_logging():
    """
    Disable, then re-enable FASTSim logging using a context manager.
    The log level is returned to its previous value.
    Logging is re-enabled even if the nested code throws an error.
    
    Example:
    ``` python
    with fastsim.utils.suppress_logging():
        ...  # do stuff with logging suppressed
    ```
    """
    previous_level = disable_logging()
    try:
        yield
    finally:
        enable_logging(previous_level)

def set_log_filename(filename: str | Path):
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.root.handlers[0].formatter)
    logging.getLogger("fastsim").addHandler(handler)
    logging.getLogger("fastsim.fastsimrust").addHandler(handler)


def get_containers_with_path(
    struct: Any,
    path: str | list,
) -> list:
    """
    Get all attributes containers from nested struct using `path` to attribute.

    Parameters
    ----------
    struct: Any
        Outermost struct where first name in `path` is an attribute
    path: str | list
        Dot-separated path, e.g. `"sd.veh.drag_coef"` or `["sd", "veh", "drag_coef"]`

    Returns
    -------
    List[Any]
        Ordered list of containers, from outermost to innermost
    """
    if isinstance(path, str):
        path = path.split(".")
    containers = [getattr(struct, path[0])]
    for subpath in path[1:-1]:
        container = getattr(containers[-1], subpath)
        containers.append(container)
    return containers


def get_attr_with_path(
    struct: Any,
    path: str | list,
) -> Any:
    """
    Get attribute from nested struct using `path` to attribute.

    Parameters
    ----------
    struct: Any
        Outermost struct where first name in `path` is an attribute
    path: str | list
        Dot-separated path, e.g. `"sd.veh.drag_coef"` or `["sd", "veh", "drag_coef"]`

    Returns
    -------
    Any
        Requested attribute
    """
    if isinstance(path, str):
        path = path.split(".")

    if len(path) == 1:
        return getattr(struct, path[0])

    containers = get_containers_with_path(struct, path)
    attr = getattr(containers[-1], path[-1])
    return attr


def set_attr_with_path(
    struct: Any,
    path: str | list,
    value: Any,
) -> Any:
    """
    Set attribute on nested struct using `path` to attribute.

    Parameters
    ----------
    struct: Any
        Outermost struct where first name in `path` is an attribute
    path: str | list
        Dot-separated path, e.g. `"sd.veh.drag_coef"` or `["sd", "veh", "drag_coef"]`
    value: Any

    Returns
    -------
    Any
        `struct` with nested value set
    """
    containers = [struct]
    if isinstance(path, str):
        path = path.split(".")
    if len(path) == 1:
        setattr(struct, path[0], value)
        return struct
    containers += get_containers_with_path(struct, path)
    # Set innermost value
    innermost_container = containers[-1]
    innermost_container.reset_orphaned()
    setattr(innermost_container, path[-1], value)
    # Convert containers back into nested structs
    nested_container = containers[-1]
    for container, nested_name in zip(containers[-2::-1], path[-2::-1]):
        if not container is struct:
            # Don't reset orphaned if `container` is outermost container
            container.reset_orphaned()
        setattr(container, nested_name, nested_container)
        nested_container = container
    # Return outermost container
    return container


def set_attrs_with_path(
    struct: Any,
    paths_and_values: Dict[str, Any]
) -> Any:
    """
    Set multiple attributes on nested struct using `path`: `value` pairs.

    Parameters
    ----------
    struct: Any
        Outermost struct where first name in `path` is an attribute
    paths_and_values: Dict[str | list, Any]
        Mapping of dot-separated path (e.g. `sd.veh.drag_coef` or `["sd", "veh", "drag_coef"]`)
        to values (e.g. `0.32`)

    Returns
    -------
    Any
        `struct` with nested values set
    """
    # TODO: make this smarter by reusing common paths
    # e.g. if setting `sd.veh.drag_coef` and `sd.veh.frontal_area_m2`, get sd.veh only once
    for path, value in paths_and_values.items():
        struct = set_attr_with_path(struct, path, value)
    return struct


def print_dt():
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def model_file_to_vehdb(model_filename: str | Path):
    # Imports
    import pandas as pd
    from fastsim.vehicle import VEHICLE_DIR, DEFAULT_VEH_DB, DEFAULT_VEHDF
    # Find last selection in database
    last_selection = DEFAULT_VEHDF.iloc[-1].selection
    # Generate new row from CSV model file
    new_veh = pd.read_csv(
        VEHICLE_DIR / model_filename,
        usecols = ["Param Name", "Param Value"],
        index_col="Param Name",
    )
    new_row = pd.DataFrame({k: [new_veh.loc[k]["Param Value"]] for k in new_veh.index})
    new_row["selection"] = last_selection + 1
    # Append new row to vehicle database
    DEFAULT_VEHDF = pd.concat([DEFAULT_VEHDF, new_row])
    # Save changes
    DEFAULT_VEHDF.to_csv(DEFAULT_VEH_DB, index=False)


def full_vehdb_to_model_files(extension: str = "yaml"):
    import fastsim
    for selection in fastsim.vehicle.DEFAULT_VEHDF.selection:
        try:
            vehdb_entry_to_model_file(selection, extension)
        except FileExistsError:
            # Raised by `open(..., "x")` if file already exists
            pass


def vehdb_entry_to_model_file(selection: int, extension: str = "yaml"):
    import fastsim
    assert extension in ("yaml", "json"), "file extension must be yaml or json"
    veh = fastsim.vehicle.Vehicle.from_vehdb(selection).to_rust()
    filename = veh.scenario_name.replace(" ", "_")+"."+extension
    for disallowed_character in ("(", ")"):
        filename = filename.replace(disallowed_character, "")
    filepath = fastsim.package_root() / "resources/vehdb" / filename
    with open(filepath, "x") as f:
        if extension == "yaml":
            f.write(veh.to_yaml())
        elif extension == "json":
            f.write(veh.to_json())
    

def calculate_tire_radius(tire_code: str, units: str = "m"):
    """
    Calculate tire radius from ISO tire code, with variable units

    Unit options: "m", "cm", "mm", "ft", "in". Default is "m".

    Examples:
    >>> fastsim.utils.calculate_tire_radius("P205/60R16")
    0.3262
    >>> fastsim.utils.calculate_tire_radius("225/70Rx19.5G", units="in")
    15.950787401574804
    """
    # Extract width, aspect ratio, and rim diameter from tire code.
    PATTERN = r"(?i)[P|LT|ST|T]?(([0-9]{2,3}\.)?[0-9]+)/(([0-9]{1,2}\.)?[0-9]+)[B|D|R]?[x|\-| ]?(([0-9]{1,2}\.)?[0-9]+)[A|B|C|D|E|F|G|H|J|L|M|N]*"
    m = re.match(PATTERN, tire_code)
    if not m:
        raise ValueError(f"Invalid tire code {tire_code}, unable to parse")
    width_mm = float(m.group(1))
    aspect_ratio = float(m.group(3))
    rim_diameter_in = float(m.group(5))
    # Calculate tire radius in mm
    sidewall_height_mm = width_mm * aspect_ratio/100
    diameter_mm = rim_diameter_in*25.4 + 2*sidewall_height_mm
    radius_mm = diameter_mm/2
    # Convert units
    UNIT_OPTIONS = ["m", "cm", "mm", "ft", "in"]
    if units == "m":
        radius = radius_mm / 1000
    elif units == "cm":
        radius = radius_mm / 10
    elif units == "mm":
        radius = radius_mm
    elif units == "ft":
        radius = radius_mm / 25.4 / 12
    elif units == "in":
        radius = radius_mm / 25.4
    else:
        raise ValueError(f"Invalid units: {units} not one of {UNIT_OPTIONS}")
    # Return result
    # print(f"Tire radius: {radius} {units}")
    return radius

def show_plots() -> bool:
    """
    Returns true if plots should be displayed
    """
    return os.environ.get("SHOW_PLOTS", "true").lower() == "true"     

def copy_demo_files(path_for_copies: Path=Path("demos")):
    """
    Copies demo files from demos folder into specified local directory

    # Arguments
    - `path_for_copies`: path to copy files into (relative or absolute in)

    # Warning
    Running this function will overwrite existing files with the same name in the specified directory, so 
    make sure any files with changes you'd like to keep are renamed.
    """
    v = f"v{fsim.__version__}"
    current_demo_path = fsim.package_root() / "demos"
    assert path_for_copies.resolve() != Path(current_demo_path), "Can't copy demos inside site-packages"
    demo_files = list(current_demo_path.glob('*demo*.py'))
    test_demo_files = list(current_demo_path.glob('*test*.py'))
    for file in test_demo_files:
        demo_files.remove(file)
    for file in demo_files:
        if os.path.exists(path_for_copies):
            dest_file = path_for_copies / file.name
            shutil.copy(file, path_for_copies)
            with open(dest_file, "r+") as file:
                file_content = file.readlines()
                prepend_str = f"# %% Copied from FASTSim version '{v}'. Guaranteed compatibility with this version only.\n"
                prepend = [prepend_str]
                file_content = prepend + file_content
                file.seek(0)
                file.writelines(file_content)
            print(f"Saved {dest_file.name} to {dest_file}")
        else:
            os.makedirs(path_for_copies)
            dest_file = path_for_copies / file.name
            shutil.copy(file, path_for_copies)
            with open(dest_file, "r+") as file:
                file_content = file.readlines()
                prepend_str = f"# %% Copied from FASTSim version '{v}'. Guaranteed compatibility with this version only.\n"
                prepend = [prepend_str]
                file_content = prepend + file_content
                file.seek(0)
                file.writelines(file_content)
            print(f"Saved {dest_file.name} to {dest_file}")