"""
Module for jit version of parameters
"""

import numpy as np
from numba.experimental import jitclass

from .buildspec import build_spec
from .parameters import PhysicalProperties

props_spec = build_spec(PhysicalProperties())

@jitclass(props_spec)
class PhysicalPropertiesJit(PhysicalProperties):
    """Container class for physical constants that could change under certain special 
    circumstances (e.g. high altitude or extreme weather) """
    pass

