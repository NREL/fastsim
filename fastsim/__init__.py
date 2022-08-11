"""Package containing modules for running FASTSim.  
For example usage, see """

from pathlib import Path 

from . import parameters as params
from . import utilities as utils
from . import simdrive, vehicle, cycle, calibration, tests
from . import calibration as cal
from . import auxiliaries

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version

__doc__ += f"{Path(__file__).parent / 'docs/README.md'}"

try:
    import fastsimrust as fsr
except ImportError:
    print("fastsimrust not installed")
else:
    # Enable easier conversion of Pyo3Arrays to numpy arrays
    import numpy as np
    def _as_numpy_array(self, *args, **kwargs):
        return np.array(list(self), *args, **kwargs)
    setattr(fsr.Pyo3ArrayF64, "__array__", _as_numpy_array)
    setattr(fsr.Pyo3ArrayU32, "__array__", _as_numpy_array)
    setattr(fsr.Pyo3ArrayBool, "__array__", _as_numpy_array)
    setattr(fsr.Pyo3VecF64, "__array__", _as_numpy_array)
