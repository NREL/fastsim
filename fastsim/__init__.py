"""Package containing modules for running FASTSim.  
For example usage, see """

from pathlib import Path
import os
import sys
import logging
import traceback

from . import parameters as params
from . import utilities as utils
from . import simdrive, vehicle, cycle, calibration, tests
from . import calibration as cal
from .resample import resample
from . import auxiliaries

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version

__doc__ += f"{Path(__file__).parent / 'docs/README.md'}"


# Set up logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d | %(filename)s#%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# Override exception handler
def _exception_handler(exc_type, exc_value, exc_traceback):
    # Handle exception normally if it's a KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    # Handle exception normally if error originates outside of FASTSim
    fastsim_dir = os.path.realpath(os.path.dirname(__file__)) + os.sep
    exc_filepath = os.path.realpath(traceback.extract_tb(exc_traceback)[-1].filename)
    if not exc_filepath.startswith(fastsim_dir):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    # Log error if it comes from FASTSim
    logger.error("uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = _exception_handler


try:
    import fastsimrust as fsr
except ImportError:
    logger.warn("fastsimrust not installed")
else:
    # Enable np.array() on array structs
    import numpy as np
    def _as_numpy_array(self, *args, **kwargs):
        return np.array(list(self), *args, **kwargs)
    setattr(fsr.Pyo3ArrayF64, "__array__", _as_numpy_array)
    setattr(fsr.Pyo3ArrayU32, "__array__", _as_numpy_array)
    setattr(fsr.Pyo3ArrayBool, "__array__", _as_numpy_array)
    setattr(fsr.Pyo3VecF64, "__array__", _as_numpy_array)
