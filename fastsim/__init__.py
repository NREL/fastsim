"""Package contaning modules for running FASTSim.  
For example usage, see ../README.md"""

from fastsim import cycle
from fastsim import vehicle
from fastsim import simdrive
from fastsim import parameters as params
from fastsim import utilities as utils

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version