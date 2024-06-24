from pathlib import Path
from typing import Any, List
import numpy as np
import re
import inspect

from .fastsim import *
import fastsim as fsim

def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent

def resources_root() -> Path:
    """
    Returns the resources root directory.
    """
    path = package_root() / "resources"
    return path


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


# creates a list of all python classes from rust structs that need variable_path_list() and
# history_path_list() added as methods
ACCEPTED_RUST_STRUCTS = [attr for attr in fsim.__dir__() if not\
                         attr.startswith("__") and\
                            isinstance(getattr(fsim,attr), type) and\
                                attr[0].isupper() and\
                                    ("fastsim" in str(inspect.getmodule(getattr(fsim,attr))))]

def variable_path_list(self, path = "", variable_path_list = []) -> List[str]:
    """Returns list of relative paths to all variables and sub-variables within
    class (relative to the class the method was called on) See example usage in
    demo_param_paths.py.  
    Arguments: ----------  
    path : Defaults to empty string. This is mainly used within the method in 
    order to call the method recursively and should not be specified by user. 
    Specifies a path to be added in front of all paths returned by the method.  
    variable_path_list : Defaults to empty list. This is mainly used within the 
    method in order to call the method recursively and should not be specified 
    by user. Specifies a list of paths to be appended to the list returned by 
    the method.  
    """
    variable_list = [attr for attr in self.__dir__() if not attr.startswith("__")\
                     and not callable(getattr(self,attr))]
    for variable in variable_list:
        if not type(getattr(self,variable)).__name__ in ACCEPTED_RUST_STRUCTS:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            variable_path_list.append(variable_path)
        elif len([attr for attr in getattr(self,variable).__dir__() if not attr.startswith("__")\
                  and not callable(getattr(getattr(self,variable),attr))]) == 0:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            variable_path_list.append(variable_path)    
        else:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            variable_path_list = getattr(self,variable).variable_path_list(
                path = variable_path,
                variable_path_list = variable_path_list,
                )
    return variable_path_list

def history_path_list(self) -> List[str]:
    """Returns a list of relative paths to all history variables (all variables
    that contain history as a subpath). 
    See example usage in demo_param_paths.py."""
    return [item for item in self.variable_path_list() if "history" in item]
            



setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405

# adds variable_path_list() and history_path_list() as methods to all classes in
# ACCEPTED_RUST_STRUCTS
for item in ACCEPTED_RUST_STRUCTS:
    setattr(getattr(fsim, item), "variable_path_list", variable_path_list)
    setattr(getattr(fsim, item), "history_path_list", history_path_list)