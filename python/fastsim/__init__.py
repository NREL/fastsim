from pathlib import Path
from typing import Union, Any
import numpy as np
import re
import inspect

from .fastsim import *

def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent


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

# def get_class_variables(cls: type) -> set[str]:
#     """Return set of class variables."""
#     # Get class attributes
#     attributes = vars(cls)

#     # Filter out methods, nested classes and dunder (__) attributes
#     return {
#         key for key, value in attributes.items()
#         if not callable(value) and not key.startswith("__")
#     }

# https://www.geeksforgeeks.org/get-variable-name-as-string-in-python/
# https://www.geeksforgeeks.org/how-to-print-a-variables-name-in-python/
def print_variable(variable):
    variable_name = [name for name, value in globals().items() if value is variable][0]
    print(f"Variable name using globals(): {variable_name}")
    return variable_name
        
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
                         'SimParams',
                         'RustSimDrive',
                         'Pyo3VecWrapper',
                         'Pyo3Vec2Wrapper',
                         'Pyo3Vec3Wrapper',
                         'Pyo3VecBoolWrapper']

def param_path_list(self, path = "", param_path_list = []) -> list[str]:
    if path == "":
        # path = str(print_variable(self))
        path = type(self).__name__
    # else:
    #     path = path + "." + name
    variable_list = [attr for attr in self.__dir__() if not attr.startswith("__") and not callable(getattr(self,attr))]
    print(variable_list)
    for variable in variable_list:
        if not type(getattr(self,variable)).__name__ in ACCEPTED_RUST_STRUCTS:
            variable_path = path + "." + variable
            print("variable type not in list: ", variable_path, type(getattr(self,variable)).__name__)
            param_path_list.append(variable_path)
        elif len([attr for attr in getattr(self,variable).__dir__() if not attr.startswith("__") and not callable(getattr(getattr(self,variable),attr))]) == 0:
            variable_path = path + "." + variable
            print("variable length zero: ", variable_path, type(getattr(self,variable)).__name__)
            param_path_list.append(variable_path)    
        else:
            variable_path = path + "." + variable
            print("variable treated as struct: ", variable_path, type(getattr(self,variable)).__name__)
            param_path_list = getattr(self,variable).param_path_list(path = variable_path, param_path_list = param_path_list)
    return param_path_list
            



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
# setattr(VehicleStateHistoryVec, "param_path_list", param_path_list)
setattr(SimDrive, "param_path_list", param_path_list)
# setattr(SimParams, "param_path_list", param_path_list)
setattr(RustSimDrive, "param_path_list", param_path_list)
setattr(Pyo3VecWrapper, "param_path_list", param_path_list)
setattr(Pyo3Vec2Wrapper, "param_path_list", param_path_list)
setattr(Pyo3Vec3Wrapper, "param_path_list", param_path_list)
setattr(Pyo3VecBoolWrapper, "param_path_list", param_path_list)