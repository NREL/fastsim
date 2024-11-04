from pkg_resources import get_distribution
from pathlib import Path
from typing import Any, List, Union, Dict, Optional
from typing_extensions import Self
import logging
import numpy as np
import re
import inspect
import pandas as pd
import polars as pl

from .fastsim import *
from . import utils

DEFAULT_LOGGING_CONFIG = dict(
    format="%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set up logging
logging.basicConfig(**DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent


def resources_root() -> Path:
    """
    Returns the resources root directory.
    """
    path = package_root() / "resources"
    return path


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
ACCEPTED_RUST_STRUCTS = [
    attr for attr in fastsim.__dir__() if not attr.startswith("__") and isinstance(getattr(fastsim, attr), type) and
    attr[0].isupper() and ("fastsim" in str(
        inspect.getmodule(getattr(fastsim, attr))))
]


def variable_path_list(self, element_as_list: bool = False) -> List[str]:
    """
    Returns list of key paths to all variables and sub-variables within
    dict version of `self`. See example usage in `fastsim/demos/
    demo_variable_paths.py`.

    # Arguments:  
    - `element_as_list`: if True, each element is itself a list of the path elements
    """
    return variable_path_list_from_py_objs(self.to_pydict(flatten=False), element_as_list=element_as_list)


def variable_path_list_from_py_objs(
    obj: Union[Dict, List],
    pre_path: Optional[str] = None,
    element_as_list: bool = False,
) -> List[str]:
    """
    Returns list of key paths to all variables and sub-variables within
    dict version of class. See example usage in `fastsim/demos/
    demo_variable_paths.py`.

    # Arguments:  
    - `obj`: fastsim object in dictionary form from `to_pydict()`
    - `pre_path`: This is used to call the method recursively and should not be
        specified by user.  Specifies a path to be added in front of all paths
        returned by the method.
    - `element_as_list`: if True, each element is itself a list of the path elements
    """
    key_paths = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            # check for nested dicts and call recursively
            if isinstance(val, dict):
                key_path = f"['{key}']" if pre_path is None else pre_path + f"['{key}']"
                key_paths.extend(
                    variable_path_list_from_py_objs(val, key_path))
            # check for lists or other iterables that do not contain numeric data
            elif "__iter__" in dir(val) and not (isinstance(val[0], float) or isinstance(val[0], int)):
                key_path = f"['{key}']" if pre_path is None else pre_path + f"['{key}']"
                key_paths.extend(
                    variable_path_list_from_py_objs(val, key_path))
            else:
                key_path = f"['{key}']" if pre_path is None else pre_path + f"['{key}']"
                key_paths.append(key_path)

    elif isinstance(obj, list):
        for key, val in enumerate(obj):
            # check for nested dicts and call recursively
            if isinstance(val, dict):
                key_path = f"[{key}]" if pre_path is None else pre_path + f"[{key}]"
                key_paths.extend(
                    variable_path_list_from_py_objs(val, key_path))
            # check for lists or other iterables that do not contain numeric data
            elif "__iter__" in dir(val) and not (isinstance(val[0], float) or isinstance(val[0], int)):
                key_path = f"[{key}]" if pre_path is None else pre_path + f"[{key}]"
                key_paths.extend(
                    variable_path_list_from_py_objs(val, key_path))
            else:
                key_path = f"[{key}]" if pre_path is None else pre_path + f"[{key}]"
                key_paths.append(key_path)

    if element_as_list:
        re_for_elems = re.compile("\\[('(\\w+)'|(\\w+))\\]")
        for i, kp in enumerate(key_paths):
            kp: str
            groups = re_for_elems.findall(kp)
            selected = [g[1] if len(g[1]) > 0 else g[2] for g in groups]
            key_paths[i] = selected

    return key_paths


def cyc_keys() -> List[str]:
    import json
    cyc = Cycle.from_resource("udds.csv")
    cyc_dict = json.loads(cyc.to_json())
    cyc_keys = [key for key, val in cyc_dict.items() if isinstance(val, list)]

    return cyc_keys


CYC_KEYS = cyc_keys()


def key_as_str(key): 
    return key if isinstance(key, str) else ".".join(key)

def is_cyc_key(key): 
    return any(
        cyc_key for cyc_key in CYC_KEYS if cyc_key == key[-1]) and "cyc" in key

def history_path_list(self, element_as_list: bool = False) -> List[str]:
    """
    Returns a list of relative paths to all history variables (all variables
    that contain history as a subpath). 
    See example usage in `fastsim/demos/demo_variable_paths.py`.

    # Arguments
    - `element_as_list`: if True, each element is itself a list of the path elements
    """
    var_paths = self.variable_path_list(element_as_list=element_as_list)
    history_paths = []
    for key in var_paths:
        if (("history" in key_as_str(key)) or is_cyc_key(key)):
            history_paths.append(key)

    return history_paths


setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405


def to_pydict(self, flatten: bool=True) -> Dict:
    """
    Returns self converted to pure python dictionary with no nested Rust objects
    # Arguments
    - `flatten`: if True, returns dict without any hierarchy
    """
    import json
    pydict = json.loads(self.to_json())
    if not flatten:
        return pydict
    else:
        return next(iter(pd.json_normalize(pydict, sep=".").to_dict(orient='records')))


@classmethod
def from_pydict(cls, pydict: Dict) -> Self:
    """
    Instantiates Self from pure python dictionary 
    """
    import json
    return cls.from_json(json.dumps(pydict))


def to_dataframe(self, pandas: bool = False, allow_partial: bool = False) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Returns time series results from fastsim object as a Polars or Pandas dataframe.

    # Arguments
    - `pandas`: returns pandas dataframe if True; otherwise, returns polars dataframe by default
    - `allow_partial`: returns dataframe of length equal to solved time steps if simulation fails early
    """
    obj_dict = self.to_pydict(flatten=False)
    history_paths = self.history_path_list(element_as_list=True)
    cols = [".".join(hp) for hp in history_paths]
    vals = []
    for hp in history_paths:
        obj: Union[dict | list] = obj_dict
        for elem in hp:
            try:
                obj = obj[elem]
            except:
                try:
                    obj = obj[int(elem)]
                except Error as err:
                    raise err
        vals.append(obj)
    if allow_partial:
        cutoff = min([len(val) for val in vals])
        if not pandas:
            df = pl.DataFrame({col: val[:cutoff]
                              for col, val in zip(cols, vals)})
        else:
            df = pd.DataFrame({col: val[:cutoff]
                              for col, val in zip(cols, vals)})
    else:
        if not pandas:
            df = pl.DataFrame({col: val for col, val in zip(cols, vals)})
        else:
            df = pd.DataFrame({col: val for col, val in zip(cols, vals)})
    return df


# adds variable_path_list() and history_path_list() as methods to all classes in
# ACCEPTED_RUST_STRUCTS
for item in ACCEPTED_RUST_STRUCTS:
    setattr(getattr(fastsim, item), "variable_path_list", variable_path_list)
    setattr(getattr(fastsim, item), "history_path_list", history_path_list)
    setattr(getattr(fastsim, item), "to_pydict", to_pydict)
    setattr(getattr(fastsim, item), "from_pydict", from_pydict)
    setattr(getattr(fastsim, item), "to_dataframe", to_dataframe)
