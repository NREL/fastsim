"""
Utilities to assist with object introspection.
"""
import inspect
from typing import Callable


def isprop(attr) -> bool:
    "Checks if instance attribute is a property."
    return isinstance(attr, property)

def isfunc(attr) -> bool:
    "Checks if instance attribute is method."
    return isinstance(attr, Callable)

def get_attrs(instance):
    """
    Given an instantiated object, returns attributes that are not:
    -- callable  
    -- special (i.e. start with `__`)  
    -- properties  
    """

    keys = []
    props = [name for (name, _) in inspect.getmembers(type(instance), isprop)]
    methods = [name for (name, _) in inspect.getmembers(type(instance), isfunc)]
    for key in instance.__dir__():
        if not(key.startswith("_")) and key not in (props + methods):
            keys.append(key)
    return keys
