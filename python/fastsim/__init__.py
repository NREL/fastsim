"""Package containing modules for running FASTSim.
For example usage, see """

from pathlib import Path
import sys
import logging
import traceback

from fastsim import parameters as params
from fastsim import utils
from fastsim import simdrive, vehicle, cycle, calibration, tests
from fastsim import calibration as cal
from fastsim.resample import resample
from fastsim import auxiliaries
from fastsim import fastsimrust
from fastsim import fastsimrust as fsr


def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent


# Set up logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from pkg_resources import get_distribution
__version__ = get_distribution("fastsim").version

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
