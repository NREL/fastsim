"""
Module for jit version of cycle.Cycle
"""

import numpy as np
from numba.experimental import jitclass                 # import the decorator
from numba import float64, int32, bool_, types    # import the types

from .buildspec import build_spec
from .cycle import Cycle 

# type specifications for attributes of Cycle class
cyc_spec = build_spec(Cycle('udds'))


@jitclass(cyc_spec)
class CycleJit(Cycle):
    """Just-in-time compiled version of Cycle using numba."""
    
    def __init__(self, len_cyc):
        """This method initialized type numpy arrays as required by 
        numba jitclass."""
        self.cycSecs = np.zeros(len_cyc, dtype=np.float64)
        self.cycMps = np.zeros(len_cyc, dtype=np.float64)
        self.cycGrade = np.zeros(len_cyc, dtype=np.float64)
        self.cycRoadType = np.zeros(len_cyc, dtype=np.float64)
        self.secs = np.zeros(len_cyc, dtype=np.float64)
        self.cycMph = np.zeros(len_cyc, dtype=np.float64)
        self.cycDistMeters = np.zeros(len_cyc, dtype=np.float64)

    def copy(self):
        """Returns copy of CycleJit instance."""
        # must be explicit because jitclass has no __getattribute__ until instantiated
        cyc = CycleJit(len(self.cycSecs))
        cyc.cycSecs = np.copy(self.cycSecs)
        cyc.cycMps = np.copy(self.cycMps)
        cyc.cycGrade = np.copy(self.cycGrade)
        cyc.cycRoadType = np.copy(self.cycRoadType)
        cyc.cycMph = np.copy(self.cycMph)
        cyc.secs = np.copy(self.secs)
        cyc.cycDistMeters = np.copy(self.cycDistMeters)
        cyc.name = self.name # should be copy of name
        return cyc

    def get_numba_cyc(self):
        """Overrides parent class (Cycle) with dummy method
        to avoid numba incompatibilities."""
        print(self.get_numba_cyc.__doc__)

    def set_standard_cycle(self):
        """Overrides parent class (Cycle) with dummy method
        to avoid numba incompatibilities."""
        print(self.set_standard_cycle.__doc__)

    def set_from_file(self):
        """Overrides parent class (Cycle) with dummy method
        to avoid numba incompatibilities."""
        print(self.set_from_file.__doc__)

    def set_from_dict(self):
        """Overrides parent class (Cycle) with dummy method
        to avoid numba incompatibilities."""
        print(self.set_from_dict.__doc__)

    def get_cyc_dict(self):
        """Overrides parent class (Cycle) with dummy method
            to avoid numba incompatibilities."""
        print(self.get_cyc_dict.__doc__)
