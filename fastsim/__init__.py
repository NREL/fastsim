"""Package contaning modules for running FASTSim.  
For example usage, see ../README.md"""

from fastsim import simdrive, vehicle, cycle
# convenient aliases
from fastsim import utilities as utils
from fastsim import parameters as params

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version
