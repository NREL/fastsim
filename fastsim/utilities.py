"""Various optional utilities that may support some applications of FASTSim."""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import seaborn as sns
import re
import datetime
import logging
from pathlib import Path
from contextlib import contextmanager

sns.set()

from fastsimrust import get_rho_air

# TODO: port to rust!
# def l__100km_to_mpg(l__100km):
#     """Given fuel economy in L/100km, returns mpg."""

#     mpg = 1 / (l__100km / 3.785 / 100 / 1_000 * params.M_PER_MI)

#     return mpg


# TODO: port to rust!
# def mpg_to_l__100km(mpg):
#     """Given fuel economy in mpg, returns L/100km."""

#     l__100km = 1 / (mpg / 3.785 * params.M_PER_MI / 1_000 / 100)

#     return l__100km

# TODO: port to rust!
# def rollav(x, y, width=10):
#     """
#     Returns x-weighted backward-looking rolling average of y.
#     Good for resampling data that needs to preserve cumulative information.
#     Arguments:
#     ----------
#     x : x data
#     y : y data (`len(y) == len(x)` must be True)
#     width: rolling average width
#     """

#     assert(len(x) == len(y))

#     dx = np.concatenate([0, x.diff()])

#     yroll = np.zeros(len(x))
#     yroll[0] = y[0]

#     for i in range(1, len(x)):
#         if i < width:
#             yroll[i] = (
#                 dx[:i] * y[:i]).sum() / (x[i] - x[0])
#         else:
#             yroll[i] = (
#                 dx[i-width:i] * y[i-width:i]).sum() / (
#                     x[i] - x[i-width])
#     return yroll

# TODO: port to rust!
# def camel_to_snake(name):
#     "Given camelCase, returns snake_case."
#     name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
#     return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

# TODO: make sure all functions below here are doing something reasonable w.r.t. rust!
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
<<<<<<< HEAD
    fastsimrust_logger = logging.getLogger("fastsimrust")
    fastsimrust_logger.setLevel(level)
=======
    if RUST_AVAILABLE:
        fastsimrust_logger = logging.getLogger("fastsim_core")
        fastsimrust_logger.setLevel(level)
>>>>>>> 834fcecf2203db09947061845dc8ca9d68b351c1
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
    logging.getLogger("fastsimrust").addHandler(handler)


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
