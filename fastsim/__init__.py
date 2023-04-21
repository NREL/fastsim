"""Package containing modules for running FASTSim.
For example usage, see """

from pathlib import Path
import sys
import logging
import traceback

import fastsimrust as fsr

def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent

# Set up logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Override exception handler
def _exception_handler(exc_type, exc_value, exc_traceback):
    # Handle exception normally if it's a KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    # Handle exception normally if error originates outside of FASTSim
    exc_filepath = Path(traceback.extract_tb(exc_traceback)[-1].filename)
    if not package_root() in exc_filepath.parents:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    # Log error if it comes from FASTSim
    logger.error("uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))
sys.excepthook = _exception_handler

from . import utilities as utils
from .resample import resample
from . import auxiliaries
from . import calibration as cal

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version

__doc__ += "\nhttps://pypi.org/project/fastsim/"
__doc__ += "\nhttps://www.nrel.gov/transportation/fastsim.html"

# Enable np.array() on array structs
import numpy as np

def _as_numpy_array(self, *args, **kwargs):
    return np.array(list(self), *args, **kwargs)
setattr(fsr.Pyo3ArrayF64, "__array__", _as_numpy_array)
setattr(fsr.Pyo3ArrayU32, "__array__", _as_numpy_array)
setattr(fsr.Pyo3ArrayBool, "__array__", _as_numpy_array)
setattr(fsr.Pyo3VecF64, "__array__", _as_numpy_array)
