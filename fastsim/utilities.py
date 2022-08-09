"""Various optional utilities that may support some applications of FASTSim."""

from __future__ import annotations
from typing import *
import numpy as np
from fastsim import parameters as params
import seaborn as sns
import re

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
        assert "." in path, "provide dot-separated path to struct, otherwise use `set_nested_values`"
        path = path.split(".")
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
