"""Package contaning modules for running FASTSim.  
For example usage, see """

from pathlib import Path 

from fastsim import simdrive, vehicle, cycle, test
# convenient aliases
from fastsim import utilities as utils
from fastsim import parameters as params

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version

__doc__ += f"{Path(__file__).parent / 'docs/README.md'}"
