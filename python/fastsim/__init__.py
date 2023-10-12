"""Package containing modules for running FASTSim.
For example usage, see """

from pathlib import Path
import sys
import logging
import traceback


def package_root() -> Path:
    """Returns the package root directory."""
    return Path(__file__).parent


# Set up logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


from . import fastsimrust
from . import fastsimrust as fsr
from . import parameters as params
from . import utilities as utils
from . import simdrive, vehicle, cycle, calibration, tests
from . import calibration as cal
from .resample import resample
from . import auxiliaries

from pkg_resources import get_distribution

__version__ = get_distribution("fastsim").version

#download demo files
import fastsim
import os
import pathlib
import fnmatch
import requests

p = 'https://github.com/NREL/fastsim/tree/' + fastsim.__version__ + '/python/fastsim/demos'
d = pathlib.Path(__file__).parent
has_demos = False
demos_dir = ''
for dir in os.walk(d):
    if fnmatch.fnmatch(dir[0], '*demos'):
        has_demos = True
        demos_dir = dir[0]
        break
if has_demos:
    #the problem is I can't figure out how to list the contents of the online demos file
    for f in p.get_dir_contents():
        already_downloaded = False
        for file in demos_dir.glob('*demo*.py'):
            #need a way to get the "basename" for the online demos files as well for this to work
            if f == os.path.basename(file.replace('\\', '/')): #necessary to ensure command works on all operating systems
                already_downloaded = True
                print('{} = {} already downloaded'.format(file, f))
                break
        if already_downloaded ==  False:
            #download file
            print('{} != {} needs downloading'.format(file, f)) #placeholder under I figure out how to download the file
else:
    #just download demos folder
    print('demos folder needs downloading') #placeholder until I figure out how to download the file

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
