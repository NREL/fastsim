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
    setattr(Pyo3ArrayF64, "__array__", _as_numpy_array)
    setattr(Pyo3ArrayU32, "__array__", _as_numpy_array)
    setattr(Pyo3ArrayBool, "__array__", _as_numpy_array)
    setattr(Pyo3VecF64, "__array__", _as_numpy_array)

    # Enable copy.copy() and copy.deepcopy()
    setattr(RustCycle, "__copy__", RustCycle.copy)
    setattr(RustCycle, "__deepcopy__", lambda self,
            _memo: RustCycle.copy(self))
    setattr(RustVehicle, "__copy__", RustVehicle.copy)
    setattr(RustVehicle, "__deepcopy__", lambda self,
            _memo: RustVehicle.copy(self))
    setattr(RustSimDrive, "__copy__", RustSimDrive.copy)
    setattr(RustSimDrive, "__deepcopy__", lambda self,
            _memo: RustSimDrive.copy(self))

except ImportError:
    print("fastsimrust not installed")
