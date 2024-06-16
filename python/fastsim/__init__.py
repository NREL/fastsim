from pathlib import Path
from typing import Union, Any
import numpy as np
import re

from .fastsim import *

def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent

from pkg_resources import get_distribution
__version__ = get_distribution("fastsim").version

def set_param_from_path(
    model: Any,
    path: str,
    value: Any,
) -> Any:
    """
    Set parameter `value` on `model` for `path` to parameter

    Example usage:
    todo
    """
    path_list = path.split(".")

    def _get_list(path_elem, container):
        list_match = re.match(r"([\w\d]+)\[(\d+)\]", path_elem)
        if list_match is not None:
            list_name = list_match.group(1)
            index = int(list_match.group(2))
            l = container.__getattribute__(list_name).tolist()
            return l, list_name, index
        else:
            return None, None, None

    containers = [model]
    lists = [None] * len(path_list)
    has_list = [False] * len(path_list)
    for i, path_elem in enumerate(path_list):
        container = containers[-1]

        list_attr, list_name, list_index = _get_list(path_elem, container)
        if list_attr is not None:
            attr = list_attr[list_index]
            # save for when we repack the containers
            lists[i] = (list_attr, list_name, list_index)
        else:
            attr = container.__getattribute__(path_elem)

        if i < len(path_list) - 1:
            containers.append(attr)

    prev_container = value

    # iterate through remaining containers, inner to outer
    for list_tuple, container, path_elem in zip(
        lists[-1::-1], containers[-1::-1], path_list[-1::-1]
    ):
        if list_tuple is not None:
            list_attr, list_name, list_index = list_tuple
            list_attr[list_index] = prev_container

            container.__setattr__(list_name, list_attr)
        else:
            container.__setattr__(f"__{path_elem}", prev_container)

        prev_container = container

    return model

def __array__(self):
    return np.array(self.tolist())


# for param_path_list() method to identify something as a struct so that it
# checks for sub-variables and sub-structs, it must be added to this list:
ACCEPTED_RUST_STRUCTS = ['FuelConverter', 
                         'FuelConverterState', 
                         'FuelConverterStateHistoryVec',
                         'ReversibleEnergyStorage',
                         'ReversibleEnergyStorageState',
                         'ReversibleEnergyStorageStateHistoryVec',
                         'ElectricMachine',
                         'ElectricMachineState',
                         'ElectricMachineStateHistoryVec',
                         'Cycle',
                         'CycleElement',
                         'Vehicle',
                         'SimDrive',
                         'RustSimDrive',
                         'Pyo3VecWrapper',
                         'Pyo3Vec2Wrapper',
                         'Pyo3Vec3Wrapper',
                         'Pyo3VecBoolWrapper']

def param_path_list(self, path = "", param_path_list = []) -> list[str]:
    """Returns list of relative paths to all variables and sub-variables within
    class (relative to the class the method was called on) 
    See example usage in demo_param_paths.py.
    Arguments:
    ----------
    path : Defaults to empty string. This is mainly used within the method in
    order to call the method recursively and does not need to be specified by
    user. Specifies a path to be added in front of all paths returned by the
    method.
    param_path_list : Defaults to empty list.  This is mainly used within the
    method in order to call the method recursively and does not need to be
    specified by user. Specifies a list of paths to be appended to the list
    returned by the method.
    """
    variable_list = [attr for attr in self.__dir__() if not attr.startswith("__") and not callable(getattr(self,attr))]
    for variable in variable_list:
        if not type(getattr(self,variable)).__name__ in ACCEPTED_RUST_STRUCTS:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            param_path_list.append(variable_path)
        elif len([attr for attr in getattr(self,variable).__dir__() if not attr.startswith("__") and not callable(getattr(getattr(self,variable),attr))]) == 0:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            param_path_list.append(variable_path)    
        else:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            param_path_list = getattr(self,variable).param_path_list(path = variable_path, param_path_list = param_path_list)
    return param_path_list

def history_path_list(self) -> list[str]:
    """Returns a list of relative paths to all history variables (all variables
    that contain history as a subpath). 
    See example usage in demo_param_paths.py."""
    return [item for item in self.param_path_list() if "history" in item]
            



setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405

# add param_path_list as an attribute for all Rust structs
setattr(FuelConverter, "param_path_list", param_path_list)
setattr(FuelConverterState, "param_path_list", param_path_list)
setattr(FuelConverterStateHistoryVec, "param_path_list", param_path_list)
setattr(ReversibleEnergyStorage, "param_path_list", param_path_list)
setattr(ReversibleEnergyStorageState, "param_path_list", param_path_list)
setattr(ReversibleEnergyStorageStateHistoryVec, "param_path_list", param_path_list)
setattr(ElectricMachine, "param_path_list", param_path_list)
setattr(ElectricMachineState, "param_path_list", param_path_list)
setattr(ElectricMachineStateHistoryVec, "param_path_list", param_path_list)
setattr(Cycle, "param_path_list", param_path_list)
setattr(CycleElement, "param_path_list", param_path_list)
setattr(Vehicle, "param_path_list", param_path_list)
setattr(SimDrive, "param_path_list", param_path_list)
setattr(RustSimDrive, "param_path_list", param_path_list)
setattr(Pyo3VecWrapper, "param_path_list", param_path_list)
setattr(Pyo3Vec2Wrapper, "param_path_list", param_path_list)
setattr(Pyo3Vec3Wrapper, "param_path_list", param_path_list)
setattr(Pyo3VecBoolWrapper, "param_path_list", param_path_list)

# add history_path_list as an attribute for all Rust structs
setattr(FuelConverter, "history_path_list", history_path_list)
setattr(FuelConverterState, "history_path_list", history_path_list)
setattr(FuelConverterStateHistoryVec, "history_path_list", history_path_list)
setattr(ReversibleEnergyStorage, "history_path_list", history_path_list)
setattr(ReversibleEnergyStorageState, "history_path_list", history_path_list)
setattr(ReversibleEnergyStorageStateHistoryVec, "history_path_list", history_path_list)
setattr(ElectricMachine, "history_path_list", history_path_list)
setattr(ElectricMachineState, "history_path_list", history_path_list)
setattr(ElectricMachineStateHistoryVec, "history_path_list", history_path_list)
setattr(Cycle, "history_path_list", history_path_list)
setattr(CycleElement, "history_path_list", history_path_list)
setattr(Vehicle, "history_path_list", history_path_list)
setattr(SimDrive, "history_path_list", history_path_list)
setattr(RustSimDrive, "history_path_list", history_path_list)
setattr(Pyo3VecWrapper, "history_path_list", history_path_list)
setattr(Pyo3Vec2Wrapper, "history_path_list", history_path_list)
setattr(Pyo3Vec3Wrapper, "history_path_list", history_path_list)
setattr(Pyo3VecBoolWrapper, "history_path_list", history_path_list)