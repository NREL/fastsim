"""Global constants representing unit conversions that shourd never change, 
physical properties that should rarely change, and vehicle model parameters 
that can be modified by advanced users."""

import os
import numpy as np
from numba.experimental import jitclass
import json

from fastsim.buildspec import build_spec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# vehicle types
CONV = 1
HEV  = 2
PHEV = 3
BEV  = 4

# vehicle types to string rep
PT_TYPES = {CONV: "Conv", HEV: "HEV", PHEV: "PHEV", BEV: "EV"}

FC_EFF_TYPES = {1: "SI", 2: "Diesel - ISB280", 3: "Diesel", 4: "Fuel Cell", 5: "Hybrid Diesel", 6: "Diesel - HD",
                7: "Diesel - HDISM Scaled", 8: "Diesel - HDISM Scaled", 9: "CNG"}

### Unit conversions that should NEVER change
mphPerMps = 2.2369
kWhPerGGE = 33.7
metersPerMile = 1609.00

# EPA fuel economy adjustment parameters

# 2008	2017	2016
# City Intercept	0.003259	0.004091	0.003259
# City Slope	1.1805	1.1601	1.1805
# Highway Intercept	0.001376	0.003191	0.001376
# Highway Slope	1.3466	1.2945	1.3466
maxEpaAdj = 0.3 # maximum EPA adjustment factor


class PhysicalProperties(object):
    """Container class for physical constants that could change under certain special 
    circumstances (e.g. high altitude or extreme weather) """

    def __init__(self):
        self.airDensityKgPerM3 = 1.2  # Sea level air density at approximately 20C
        self.gravityMPerSec2 = 9.81

props_spec = build_spec(PhysicalProperties())

@jitclass(props_spec)
class PhysicalPropertiesJit(PhysicalProperties):
    """Container class for physical constants that could change under certain special 
    circumstances (e.g. high altitude or extreme weather) """
    pass

### Vehicle model parameters that should be changed only by advanced users
# Discrete power out percentages for assigning FC efficiencies -- all hardcoded ***
fcPwrOutPerc = np.array(
    [0, 0.005, 0.015, 0.04, 0.06, 0.10, 0.14, 0.20, 0.40, 0.60, 0.80, 1.00])

# fc arrays and parameters
# Efficiencies at different power out percentages by FC type -- all
eff_si = np.array([0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30])
eff_atk = np.array([0.10, 0.12, 0.28, 0.35, 0.375, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35])
eff_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])
eff_fuel_cell = np.array([0.10, 0.30, 0.36, 0.45, 0.50, 0.56, 0.58, 0.60, 0.58, 0.57, 0.55, 0.54])
eff_hd_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])


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


# loading long arrays from json file
with open(os.path.join(THIS_DIR, 'resources', 'longparams.json'), 'r') as paramfile:
    param_dict = json.load(paramfile)

# PHEV-specific parameters
rechgFreqMiles = param_dict['rechgFreqMiles']
ufArray = param_dict['ufArray']
