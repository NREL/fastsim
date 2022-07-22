"""Package containing modules for running FASTSim.  
For example usage, see """

from pathlib import Path

from . import parameters as params
from . import utilities as utils
from . import simdrive, vehicle, cycle, calibration, tests
from . import calibration as cal
from .resample import resample
from . import auxiliaries

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version

__doc__ += f"{Path(__file__).parent / 'docs/README.md'}"

try:
    from fastsimrust import *
    
    # Enable np.array() on array structs
    import numpy as np
    def _as_numpy_array(self, *args, **kwargs):
        return np.array(list(self), *args, **kwargs)
    for struct in (Pyo3ArrayF64, Pyo3ArrayU32, Pyo3ArrayBool, Pyo3VecF64):
        struct.__array__ = _as_numpy_array
    
    # Enable setting nested structs attributes more easily (UNFINISHED)
    """def _set_nested_attr(self, name, value):
        self.reset_orphaned()  # TODO: doesn't always need to be called, add condition
        object.__setattr__(self, name, value)  # Call original __setattr__
    for struct in (RustCycle, RustVehicle, RustSimDrive, VehicleThermal, SimDriveHot):
        struct.__setattr__ = _set_nested_attr"""
    
    # Enable copy.copy() and copy.deepcopy(), order of assignment matters
    for struct in (RustCycle, RustVehicle, RustSimDrive, VehicleThermal, SimDriveHot):
        struct.__copy__ = struct.copy
        struct.__deepcopy__ = lambda self, _memo: struct.copy(self)

except ImportError:
    print("fastsimrust not installed")
