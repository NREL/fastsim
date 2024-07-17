from pathlib import Path
from typing import Union, Any
import logging
import numpy as np
import re

from .fastsim import *
from . import utils

DEFAULT_LOGGING_CONFIG = dict(
    format = "%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
) 

# Set up logging
logging.basicConfig(**DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)

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

setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405
