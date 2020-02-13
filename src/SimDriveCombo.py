"""Module containing classes for simulating 2 Cycle fuel economy and provided label fuel economy"""

import LoadData
import SimDrive 
from numba import jitclass, deferred_type

class SimTwoCycle(object):
    """Class for simulating 2 cycle fuel economy and calculating label fuel economy"""

    def __init__(self):
        """Initializes numba jit udds and hwfet cycles and SimDrive objects"""
        self.udds = LoadData.Cycle('udds').get_numba_cyc()
        self.hwfet = LoadData.Cycle('hwfet').get_numba_cyc()
        self.sim_drv_udds = SimDrive.SimDriveJit(len(self.udds.cycSecs))
        self.sim_drv_hwfet = SimDrive.SimDriveJit(len(self.hwfet.cycSecs))

    def set_2cyc_label_fe(self, veh):
        self.sim_drv_udds.sim_drive(self.udds, veh, -1)
        self.sim_drv_udds.set_post_scalars(self.udds, veh)
        self.sim_drv_hwfet.sim_drive(self.hwfet, veh, -1)
        self.sim_drv_hwfet.set_post_scalars(self.hwfet, veh)

        self.label = 0.55 * self.sim_drv_udds.mpgge_elec + \
            0.45 * self.sim_drv_hwfet.mpgge_elec

spec = []
@jitclass(spec)
class SimTwoCycleJit(SimTwoCycle):
    """Inherits everything needed from SimTwoCycle."""