"""Package contaning modules for running FASTSim.  
For example usage, see """

from pathlib import Path 

from . import parameters as params
from fastsim import utilities as utils
from . import simdrive, vehicle, cycle, buildspec
from . import parametersjit, vehiclejit, cyclejit, simdrivejit, test, simdrivehot
# TODO: create and add simdrivehotjit

from pkg_resources import get_distribution

__version__ = get_distribution('fastsim').version

__doc__ += f"{Path(__file__).parent / 'docs/README.md'}"
