"""Package contaning modules for running FASTSim.  
For example usage, see ../README.md"""

# convenient aliases
from fastsim import utilities as utils
from fastsim import parameters as params
from fastsim import simdrive, vehicle, cycle, simdrivehot, tests

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version
