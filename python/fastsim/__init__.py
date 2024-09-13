"""Package containing modules for running FASTSim.
For example usage, see """

from pathlib import Path
import sys
import logging
import traceback
from typing import Dict

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


DEFAULT_LOGGING_CONFIG = dict(
    format = "%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
) 



# Set up logging
logging.basicConfig(**DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)

from pkg_resources import get_distribution
__version__ = get_distribution("fastsim").version

__doc__ += "\nhttps://pypi.org/project/fastsim/"
__doc__ += "\nhttps://www.nrel.gov/transportation/fastsim.html"
