"""Package containing modules for running FASTSim.
For example usage, see """

from pathlib import Path
import sys
import logging
import traceback
from typing import Dict
from typing_extensions import Self
import inspect

import fastsim
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

def resources_root() -> Path:
    """Returns the resources root directory."""
    return Path(__file__).parent / "resources"


DEFAULT_LOGGING_CONFIG = dict(
    format = "%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
) 



# Set up logging
logging.basicConfig(**DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)

from importlib.metadata import version
__version__ = version('fastsim')

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

# creates a list of all python classes from rust structs that need to_pydict and
# from_pydict added as methods
ACCEPTED_RUST_STRUCTS = [attr for attr in fastsim.__dir__() if not\
                         attr.startswith("__") and\
                            isinstance(getattr(fastsim, attr), type) and\
                                attr[0].isupper() and\
                                    ("fastsim" in str(inspect.getmodule(getattr(fastsim, attr))))]

def to_pydict(self) -> Dict:
    """
    Returns self converted to pure python dictionary with no nested Rust objects
    """
    import json
    return json.loads(self.to_json())

@classmethod
def from_pydict(cls, pydict: Dict) -> Self:
    """
    Instantiates Self from pure python dictionary 
    """
    import json
    return cls.from_json(json.dumps(pydict))

for item in ACCEPTED_RUST_STRUCTS:
    setattr(getattr(fastsim, item), "to_pydict", to_pydict)
    setattr(getattr(fastsim, item), "from_pydict", from_pydict)

setattr(fsr.RustVehicle, "to_pydict", to_pydict)
setattr(fsr.RustVehicle, "from_pydict", from_pydict)
setattr(fastsim.vehicle.Vehicle, "to_pydict", to_pydict)
setattr(fastsim.vehicle.Vehicle, "from_pydict", from_pydict)
