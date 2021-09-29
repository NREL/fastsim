"""
Global constants representing unit conversions that shourd never change, 
physical properties that should rarely change, and vehicle model parameters 
that can be modified by advanced users.  
Note that modifications to parameters in this module do not propagate to 
numba-jit-compiled objects.  
"""

import os
import numpy as np
import json
from pathlib import Path


THIS_DIR = Path(__file__).parent

# vehicle types
CONV = 1
HEV  = 2
PHEV = 3
BEV  = 4

# vehicle types to string rep
PT_TYPES = {CONV: "Conv", HEV: "HEV", PHEV: "PHEV", BEV: "EV"}


### Unit conversions that should NEVER change
mphPerMps = 2.2369
metersPerMile = 1609.00

class PhysicalProperties(object):
    """Container class for physical constants that could change under certain special 
    circumstances (e.g. high altitude or extreme weather) """

    def __init__(self):
        # Make this altitude and temperature dependent, and allow it to change with time
        self.airDensityKgPerM3 = 1.2  # Sea level air density at approximately 20C
        self.gravityMPerSec2 = 9.81

def PhysicalPropertiesJit():
    "Wrapper for parametersjit: "
    from . import parametersjit

    props = parametersjit.PhysicalPropertiesJit()
    PhysicalPropertiesJit.__doc__ += props.__doc__

    return props

### Vehicle model parameters that should be changed only by advanced users
# Discrete power out percentages for assigning FC efficiencies -- all hardcoded ***
fcPwrOutPerc = np.array(
    [0, 0.005, 0.015, 0.04, 0.06, 0.10, 0.14, 0.20, 0.40, 0.60, 0.80, 1.00], 
    dtype=np.float64)

# fc arrays and parameters
# Efficiencies at different power out percentages by FC type -- all
fcEffMap_si = np.array([0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30])
fcEffMap_atk = np.array([0.10, 0.12, 0.28, 0.35, 0.375, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35])
fcEffMap_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])
fcEffMap_fuel_cell = np.array([0.10, 0.30, 0.36, 0.45, 0.50, 0.56, 0.58, 0.60, 0.58, 0.57, 0.55, 0.54])
fcEffMap_hd_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])


# Relatively continuous power out percentages for assigning FC efficiencies
fcPercOutArray = np.r_[np.arange(0, 3.0, 0.1), np.arange(
    3.0, 7.0, 0.5), np.arange(7.0, 60.0, 1.0), np.arange(60.0, 105.0, 5.0)] / 100  # hardcoded ***

# motor arrays and parameters
mcPwrOutPerc = np.array(
    [0.00, 0.02, 0.04, 0.06, 0.08,	0.10,	0.20,	0.40,	0.60,	0.80,	1.00])
large_baseline_eff = np.array(
    [0.83, 0.85,	0.87,	0.89,	0.90,	0.91,	0.93,	0.94,	0.94,	0.93,	0.92])
small_baseline_eff = np.array(
    [0.12,	0.16,	 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93,	0.93,	0.92])
modern_max = 0.95
mcPercOutArray = np.linspace(0, 1, 101)

ENERGY_AUDIT_ERROR_TOLERANCE = 0.02 # i.e., 2%

chgEff = 0.86 # charger efficiency for PEVs, this should probably not be hard coded long term

# loading long arrays from json file
with open(THIS_DIR / 'resources' / 'longparams.json', 'r') as paramfile:
    param_dict = json.load(paramfile)

# PHEV-specific parameters
rechgFreqMiles = param_dict['rechgFreqMiles']
ufArray = param_dict['ufArray']
