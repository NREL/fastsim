"""
Module for jit version of cycle.Cycle
"""

from fastsim import parametersjit
import numpy as np
from numba.experimental import jitclass

from .buildspec import build_spec
from .vehicle import Vehicle
from . import parametersjit

veh_spec = build_spec(Vehicle('template.csv'))


@jitclass(veh_spec)
class VehicleJit(Vehicle):
    """Just-in-time compiled version of Vehicle using numba."""
    
    def __init__(self):
        """This method initialized type numpy arrays as required by
        numba jitclass."""
        self.MaxRoadwayChgKw = np.zeros(6, dtype=np.float64)
        self.fcEffArray = np.zeros(100, dtype=np.float64)
        self.fcKwOutArray = np.zeros(100, dtype=np.float64)
        self.mcKwInArray = np.zeros(101, dtype=np.float64)
        self.mcKwOutArray = np.zeros(101, dtype=np.float64)
        self.mcFullEffArray = np.zeros(101, dtype=np.float64)
        self.mcEffArray = np.zeros(11, dtype=np.float64)

        self.props = parametersjit.PhysicalPropertiesJit()

    def get_numba_veh(self):
        """Overrides parent class (Cycle) with dummy method 
        to avoid numba incompatibilities."""
        print(self.get_numba_veh.__doc__)

    def load_veh(self):
        """Overrides parent class (Cycle) with dummy method 
        to avoid numba incompatibilities."""
        print(self.load_veh.__doc__)

    def set_init_calcs(self):
        """Overrides parent class (Cycle) with dummy method 
        to avoid numba incompatibilities.
        Runs self.set_dependents()"""
        self.set_dependents()


