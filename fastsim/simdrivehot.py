"""Module containing classes and methods for simulating vehicle drive cycle.
For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
import pandas as pd
import re
import sys
from numba import jitclass                 # import the decorator
from numba import float64, int32, bool_, types    # import the types
import warnings
warnings.simplefilter('ignore')

# local modules
from . import globalvars as gl
from .simdrive import SimDriveClassic, SimDriveJit, SimDrivePost

class SimDriveHot(SimDriveClassic):
    """Class containing methods for running FASTSim vehicle 
    fuel economy simulations with thermal behavior. 
    
    This class is not compiled and will run slower for large batch runs."""

    def __init__(self, cyc, veh):
        """Initialize time array variables that are not used in base SimDrive."""
        super().__init__(cyc, veh)
        
        len_cyc = len(self.cyc.cycSecs)

        # variables that require more thought
        # epsAC = 3.0
        # qRadKw = SolIrr*epsAC/1000         

        # temperatures
        self.te_cab_degC = np.zeros(len_cyc, dtype=np.float64)
        self.te_oil_degC = np.zeros(len_cyc, dtype=np.float64)
        self.te_clnt_degC = np.zeros(len_cyc, dtype=np.float64)
        self.te_cat_degC = np.zeros(len_cyc, dtype=np.float64)
        self.te_mot_degC = np.zeros(len_cyc, dtype=np.float64)
        self.te_bat_degC = np.zeros(len_cyc, dtype=np.float64)

        # heat flows
        self.pw_htr_kw = np.zeros(len_cyc, dtype=np.float64)
        self.pw_ac_kw = np.zeros(len_cyc, dtype=np.float64)
        self.pw_ac_kw = np.zeros(len_cyc, dtype=np.float64)
