"""Module containing classes and methods for simulating vehicle drive
cycle. For example usage, see ../README.md"""

### Import necessary python modules
from logging import debug
import numpy as np
import re

from . import params, cycle, vehicle, utils
# these imports are needed for numba to type these correctly
from .vehicle import CONV, HEV, PHEV, BEV 
from .vehicle import SI, ATKINSON, DIESEL, H2FC, HD_DIESEL

class SimDriveParamsClassic(object):
    """Class containing attributes used for configuring sim_drive.
    Usually the defaults are ok, and there will be no need to use this.

    See comments in code for descriptions of various parameters that
    affect simulation behavior. If, for example, you want to suppress
    warning messages, use the following pastable code EXAMPLE:

    >>> cyc = cycle.Cycle.from_file('udds')
    >>> veh = vehicle.Vehicle(1)
    >>> sim_drive = simdrive.SimDriveClassic(cyc, veh)
    >>> sim_drive.sim_params.verbose = False # turn off error messages for large time steps
    >>> sim_drive.sim_drive()"""

    def __init__(self):
        """Default values that affect simulation behavior.  
        Can be modified after instantiation."""
        self.missed_trace_correction = False  # if True, missed trace correction is active, default = False
        # maximum time dilation factor to "catch up" with trace -- e.g. 1.0 means 100% increase in step size
        self.max_time_dilation = 1.0  
        # minimum time dilation margin to let trace "catch up" -- e.g. -0.5 means 50% reduction in step size
        self.min_time_dilation = -0.5  
        self.time_dilation_tol = 5e-4  # convergence criteria for time dilation
        self.max_trace_miss_iters = 5 # number of iterations to achieve time dilation correction
        self.trace_miss_speed_mps_tol = 1.0 # threshold of error in speed [m/s] that triggers warning
        self.trace_miss_time_tol = 1e-3 # threshold for printing warning when time dilation is active
        self.trace_miss_dist_tol = 1e-3 # threshold of fractional eror in distance that triggers warning
        self.sim_count_max = 30  # max allowable number of HEV SOC iterations
        self.verbose = True  # show warning and other messages
        self.newton_gain = 0.9 # newton solver gain
        self.newton_max_iter = 100 # newton solver max iterations
        self.newton_xtol = 1e-9 # newton solver tolerance
        self.energy_audit_error_tol = 0.002 # tolerance for energy audit error warning, i.e. 0.1%
                
        # EPA fuel economy adjustment parameters
        self.maxEpaAdj = 0.3 # maximum EPA adjustment factor

class SimDriveClassic(object):
    """Class containing methods for running FASTSim vehicle 
    fuel economy simulations. This class is not compiled and will 
    run slower for large batch runs.
    Arguments:
    ----------
    cyc: cycle.Cycle instance
    veh: vehicle.Vehicle instance"""

    def __init__(self, cyc: cycle.Cycle, veh: vehicle.Vehicle):
        """Initalizes arrays, given vehicle.Vehicle() and cycle.Cycle() as arguments.
        sim_params is needed only if non-default behavior is desired."""
        self.__init_objects__(cyc, veh)
        self.init_arrays()
        # initialized here for downstream classes that do not run sim_drive
        self.hev_sim_count = 0 

    def __init_objects__(self, cyc: cycle.Cycle, veh: vehicle.Vehicle):        
        self.veh = veh
        self.cyc = cycle.copy_cycle(cyc) # this cycle may be manipulated
        self.cyc0 = cycle.copy_cycle(cyc) # this cycle is not to be manipulated
        self.sim_params = SimDriveParamsClassic()
        self.props = params.PhysicalProperties()

    def init_arrays(self):
        self.i = 1 # initialize step counter for possible use outside sim_drive_walk()

        # Component Limits -- calculated dynamically
        self.curMaxFsKwOut = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcTransLimKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcFsLimKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcMaxKwIn = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxFcKwOut = np.zeros(self.cyc.len, dtype=np.float64)
        self.essCapLimDischgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxEssKwOut = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxAvailElecKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.essCapLimChgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxEssChgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxElecKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mcElecInLimKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mcTransiLimKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxMcKwOut = np.zeros(self.cyc.len, dtype=np.float64)
        self.essLimMcRegenPercKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.essLimMcRegenKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxMechMcKwIn = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxTransKwOut = np.zeros(self.cyc.len, dtype=np.float64)

        ### Drive Train
        self.cycDragKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycAccelKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycAscentKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycTracKwReq = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxTracKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.spareTracKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycRrKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycWheelRadPerSec = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycTireInertiaKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycWheelKwReq = np.zeros(self.cyc.len, dtype=np.float64)
        self.regenContrLimKwPerc = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycRegenBrakeKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycFricBrakeKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycTransKwOutReq = np.zeros(self.cyc.len, dtype=np.float64)
        self.cycMet = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.transKwOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.transKwInAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.curSocTarget = np.zeros(self.cyc.len, dtype=np.float64)
        self.minMcKw2HelpFc = np.zeros(self.cyc.len, dtype=np.float64)
        self.mcMechKwOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.mcElecKwInAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.auxInKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.roadwayChgKwOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.minEssKw2HelpFc = np.zeros(self.cyc.len, dtype=np.float64)
        self.essKwOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcKwOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcKwOutAch_pct = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcKwInAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.fsKwOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.fsCumuMjOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.fsKwhOutAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.essCurKwh = np.zeros(self.cyc.len, dtype=np.float64)
        self.soc = np.zeros(self.cyc.len, dtype=np.float64)

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = np.zeros(self.cyc.len, dtype=np.float64)
        self.essRegenBufferDischgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.maxEssRegenBufferChgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.essAccelBufferChgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.accelBufferSoc = np.zeros(self.cyc.len, dtype=np.float64)
        self.maxEssAccelBufferDischgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.essAccelRegenDischgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mcElectInKwForMaxFcEff = np.zeros(self.cyc.len, dtype=np.float64)
        self.electKwReq4AE = np.zeros(self.cyc.len, dtype=np.float64)
        self.canPowerAllElectrically = np.array(
            [False] * self.cyc.len, dtype=np.bool_)
        self.desiredEssKwOutForAE = np.zeros(self.cyc.len, dtype=np.float64)
        self.essAEKwOut = np.zeros(self.cyc.len, dtype=np.float64)
        self.erAEKwOut = np.zeros(self.cyc.len, dtype=np.float64)
        self.essDesiredKw4FcEff = np.zeros(self.cyc.len, dtype=np.float64)
        self.essKwIfFcIsReq = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxMcElecKwIn = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcKwGapFrEff = np.zeros(self.cyc.len, dtype=np.float64)
        self.erKwIfFcIsReq = np.zeros(self.cyc.len, dtype=np.float64)
        self.mcElecKwInIfFcIsReq = np.zeros(self.cyc.len, dtype=np.float64)
        self.mcKwIfFcIsReq = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcForcedOn = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.fcForcedState = np.zeros(self.cyc.len, dtype=np.int32)
        self.mcMechKw4ForcedFc = np.zeros(self.cyc.len, dtype=np.float64)
        self.fcTimeOn = np.zeros(self.cyc.len, dtype=np.float64)
        self.prevfcTimeOn = np.zeros(self.cyc.len, dtype=np.float64)

        ### Additional Variables
        self.mpsAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.mphAch = np.zeros(self.cyc.len, dtype=np.float64)
        self.distMeters = np.zeros(self.cyc.len, dtype=np.float64)
        self.distMiles = np.zeros(self.cyc.len, dtype=np.float64)
        self.highAccFcOnTag = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.reachedBuff = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.maxTracMps = np.zeros(self.cyc.len, dtype=np.float64)
        self.addKwh = np.zeros(self.cyc.len, dtype=np.float64)
        self.dodCycs = np.zeros(self.cyc.len, dtype=np.float64)
        self.essPercDeadArray = np.zeros(self.cyc.len, dtype=np.float64)
        self.dragKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.essLossKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.accelKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ascentKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.rrKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.curMaxRoadwayChgKw = np.zeros(self.cyc.len, dtype=np.float64)
        self.trace_miss_iters = np.zeros(self.cyc.len, dtype=np.float64)
        self.newton_iters = np.zeros(self.cyc.len, dtype=np.float64)

    def sim_drive(self, initSoc=-1, auxInKwOverride=np.zeros(1, dtype=np.float64)):
        """
        Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        initSoc: initial SOC for electrified vehicles.  
        auxInKw: auxInKw override.  Array of same length as cyc.time_s.  
            Default of np.zeros(1) causes veh.auxKw to be used.
            If zero is actually desired as an override, either set
            veh.auxKw = 0 before instantiaton of SimDrive*, or use
            `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting the
            final value to non-zero prevents override mechanism.  
        """

        if (auxInKwOverride == 0).all():
            auxInKwOverride = self.auxInKw
        self.hev_sim_count = 0

        if initSoc != -1:
            if initSoc > 1.0 or initSoc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = None

        elif self.veh.vehPtType == CONV:  # Conventional
            # If no EV / Hybrid components, no SOC considerations.
            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0

        elif self.veh.vehPtType == HEV and initSoc == -1:  # HEV
            #####################################
            ### Charge Balancing Vehicle SOC ###
            #####################################
            # Charge balancing SOC for HEV vehicle types. Iterating initsoc and comparing to final SOC.
            # Iterating until tolerance met or 30 attempts made.
            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            ess2fuelKwh = 1.0
            while ess2fuelKwh > self.veh.essToFuelOkError and self.hev_sim_count < self.sim_params.sim_count_max:
                self.hev_sim_count += 1
                self.sim_drive_walk(initSoc, auxInKwOverride)
                fuelKj = np.sum(self.fsKwOutAch * self.cyc.dt_s)
                roadwayChgKj = np.sum(self.roadwayChgKwOutAch * self.cyc.dt_s)
                if (fuelKj + roadwayChgKj) > 0:
                    ess2fuelKwh = np.abs(
                        (self.soc[0] - self.soc[-1]) * self.veh.maxEssKwh * 3_600 / (fuelKj + roadwayChgKj)
                    )
                else:
                    ess2fuelKwh = 0.0
                initSoc = min(1.0, max(0.0, self.soc[-1]))

        elif (self.veh.vehPtType == PHEV and initSoc == -1) or (self.veh.vehPtType == BEV and initSoc == -1):  # PHEV and BEV
            # If EV, initializing initial SOC to maximum SOC.
            initSoc = self.veh.maxSoc

        self.sim_drive_walk(initSoc, auxInKwOverride)

        self.set_post_scalars()

    def sim_drive_walk(self, initSoc, auxInKwOverride=np.zeros(1, dtype=np.float64)):
        """Receives second-by-second cycle information, vehicle properties, 
        and an initial state of charge and runs sim_drive_step to perform a 
        backward facing powertrain simulation. Method 'sim_drive' runs this
        iteratively to achieve correct SOC initial and final conditions, as 
        needed.

        Arguments
        ------------
        initSoc (optional): initial battery state-of-charge (SOC) for electrified vehicles
        auxInKw: auxInKw override.  Array of same length as cyc.time_s.  
                Default of np.zeros(1) causes veh.auxKw to be used. If zero is actually
                desired as an override, either set veh.auxKw = 0 before instantiaton of
                SimDrive*, or use `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting
                the final value to non-zero prevents override mechanism.  
        """
        
        ############################
        ###   Loop Through Time  ###
        ############################

        ###  Assign First Values  ###
        ### Drive Train
        self.init_arrays() # reinitialize arrays for each new run
        if not((auxInKwOverride == 0).all()):
            self.auxInKw = auxInKwOverride
        
        self.cycMet[0] = True
        self.curSocTarget[0] = self.veh.maxSoc
        self.essCurKwh[0] = initSoc * self.veh.maxEssKwh
        self.soc[0] = initSoc
        self.mpsAch[0] = self.cyc0.mps[0]
        self.mphAch[0] = self.cyc0.mph[0]

        if self.sim_params.missed_trace_correction:
            self.cyc = cycle.copy_cycle(self.cyc0) # reset the cycle in case it has been manipulated
            print('copy type')
            print(type(self.cyc.time_s[1]))

        self.i = 1 # time step counter
        while self.i < len(self.cyc.time_s):
            if self.i <= 1:
                debug = True    
            self.sim_drive_step()
            if self.i <= 1:
                debug = True

        if self.sim_params.missed_trace_correction: 
            self.cyc.time_s = self.cyc.dt_s.cumsum() # correct time_s based on actual trace

        if (self.cyc.dt_s > 5).any() and self.sim_params.verbose:
            if self.sim_params.missed_trace_correction:
                print('Max time dilation factor =', (round((self.cyc.dt_s / self.cyc0.dt_s).max(), 3)))
            print("Warning: large time steps affect accuracy significantly.") 
            print("To suppress this message, view the doc string for simdrive.SimDriveParams.")
            print('Max time step =', (round(self.cyc.dt_s.max(), 3)))

    def sim_drive_step(self):
        """Step through 1 time step."""
        self.solve_step(self.i)
        if self.sim_params.missed_trace_correction and (self.cyc0.dist_m[:self.i].sum() > 0):
            self.set_time_dilation(self.i)
        # TODO: implement something for coasting here
        # if self.impose_coast[i] == True
            # self.set_coast_speeed(i)

        self.i += 1 # increment time step counter
    
    def solve_step(self, i):
        """Perform all the calculations to solve 1 time step."""
        self.set_misc_calcs(i)
        self.set_comp_lims(i)
        self.set_power_calcs(i)
        self.set_ach_speed(i)
        self.set_hybrid_cont_calcs(i)
        self.set_fc_forced_state(i) # can probably be *mostly* done with list comprehension in post processing
        self.set_hybrid_cont_decisions(i)
        self.set_fc_power(i)

    def set_misc_calcs(self, i):
        """Sets misc. calculations at time step 'i'
        Arguments:
        ----------
        i: index of time step"""
        # if cycle iteration is used, auxInKw must be re-zeroed to trigger the below if statement
        if (self.auxInKw[i:] == 0).all():
            # if all elements after i-1 are zero, trigger default behavior; otherwise, use override value 
            if self.veh.noElecAux:
                self.auxInKw[i] = self.veh.auxKw / self.veh.altEff
            else:
                self.auxInKw[i] = self.veh.auxKw            
        # Is SOC below min threshold?
        if self.soc[i-1] < (self.veh.minSoc + self.veh.percHighAccBuf):
            self.reachedBuff[i] = False
        else:
            self.reachedBuff[i] = True

        # Does the engine need to be on for low SOC or high acceleration
        if self.soc[i-1] < self.veh.minSoc or (self.highAccFcOnTag[i-1] and not(self.reachedBuff[i])):
            self.highAccFcOnTag[i] = True
        else:
            self.highAccFcOnTag[i] = False
        self.maxTracMps[i] = self.mpsAch[i-1] + (self.veh.maxTracMps2 * self.cyc.dt_s[i])

    def set_comp_lims(self, i):
        """Sets component limits for time step 'i'
        Arguments
        ------------
        i: index of time step
        initSoc: initial SOC for electrified vehicles"""

        # max fuel storage power output
        self.curMaxFsKwOut[i] = min(
            self.veh.maxFuelStorKw, 
            self.fsKwOutAch[i-1] + (
                (self.veh.maxFuelStorKw / self.veh.fuelStorSecsToPeakPwr) * (self.cyc.dt_s[i])))
        # maximum fuel storage power output rate of change
        self.fcTransLimKw[i] = self.fcKwOutAch[i-1] + (
            self.veh.maxFuelConvKw / self.veh.fuelConvSecsToPeakPwr * self.cyc.dt_s[i]
        )

        self.fcMaxKwIn[i] = min(self.curMaxFsKwOut[i], self.veh.maxFuelStorKw)
        self.fcFsLimKw[i] = self.veh.fcMaxOutkW
        self.curMaxFcKwOut[i] = min(
            self.veh.maxFuelConvKw, self.fcFsLimKw[i], self.fcTransLimKw[i])

        if self.veh.maxEssKwh == 0 or self.soc[i-1] < self.veh.minSoc:
            self.essCapLimDischgKw[i] = 0.0

        else:
            self.essCapLimDischgKw[i] = self.veh.maxEssKwh * np.sqrt(self.veh.essRoundTripEff) * 3.6e3 * (
                self.soc[i-1] - self.veh.minSoc) / self.cyc.dt_s[i]
        self.curMaxEssKwOut[i] = min(
            self.veh.maxEssKw, self.essCapLimDischgKw[i])

        if self.veh.maxEssKwh == 0 or self.veh.maxEssKw == 0:
            self.essCapLimChgKw[i] = 0

        else:
            self.essCapLimChgKw[i] = max(
                (self.veh.maxSoc - self.soc[i-1]) * self.veh.maxEssKwh * 1 / np.sqrt(self.veh.essRoundTripEff) / 
                (self.cyc.dt_s[i] * 1 / 3.6e3), 
                0
            )

        self.curMaxEssChgKw[i] = min(self.essCapLimChgKw[i], self.veh.maxEssKw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if self.veh.fcEffType == H2FC:
            self.curMaxElecKw[i] = self.curMaxFcKwOut[i] + self.curMaxRoadwayChgKw[i] + self.curMaxEssKwOut[i] - self.auxInKw[i]

        else:
            self.curMaxElecKw[i] = self.curMaxRoadwayChgKw[i] + self.curMaxEssKwOut[i] - self.auxInKw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        self.curMaxAvailElecKw[i] = min(self.curMaxElecKw[i], self.veh.mcMaxElecInKw)

        if self.curMaxElecKw[i] > 0:
            # limit power going into e-machine controller to
            if self.curMaxAvailElecKw[i] == max(self.veh.mcKwInArray):
                self.mcElecInLimKw[i] = min(self.veh.mcKwOutArray[-1], self.veh.maxMotorKw)
            else:
                self.mcElecInLimKw[i] = min(
                    self.veh.mcKwOutArray[
                        np.argmax(self.veh.mcKwInArray > min(
                            max(self.veh.mcKwInArray) - 0.01, 
                            self.curMaxAvailElecKw[i]
                        )) - 1],
                    self.veh.maxMotorKw)
        else:
            self.mcElecInLimKw[i] = 0.0

        # Motor transient power limit
        self.mcTransiLimKw[i] = abs(
            self.mcMechKwOutAch[i-1]) + self.veh.maxMotorKw / self.veh.motorSecsToPeakPwr * self.cyc.dt_s[i]

        self.curMaxMcKwOut[i] = max(
            min(
                self.mcElecInLimKw[i], 
                self.mcTransiLimKw[i], 
                np.float64(0 if self.veh.stopStart else 1) * self.veh.maxMotorKw),
            -self.veh.maxMotorKw
        )

        if self.curMaxMcKwOut[i] == 0:
            self.curMaxMcElecKwIn[i] = 0
        else:
            if self.curMaxMcKwOut[i] == self.veh.maxMotorKw:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / \
                    self.veh.mcFullEffArray[-1]
            else:
                self.curMaxMcElecKwIn[i] = (self.curMaxMcKwOut[i] / self.veh.mcFullEffArray[
                        max(1, np.argmax(
                            self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.curMaxMcKwOut[i])
                            ) - 1
                        )
                    ]
                )

        if self.veh.maxMotorKw == 0:
            self.essLimMcRegenPercKw[i] = 0.0

        else:
            self.essLimMcRegenPercKw[i] = min(
                (self.curMaxEssChgKw[i] + self.auxInKw[i]) / self.veh.maxMotorKw, 1)
        if self.curMaxEssChgKw[i] == 0:
            self.essLimMcRegenKw[i] = 0.0

        else:
            if self.veh.maxMotorKw == self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i]:
                self.essLimMcRegenKw[i] = min(
                    self.veh.maxMotorKw, self.curMaxEssChgKw[i] / self.veh.mcFullEffArray[-1])
            else:
                self.essLimMcRegenKw[i] = min(
                    self.veh.maxMotorKw, 
                    self.curMaxEssChgKw[i] / self.veh.mcFullEffArray[
                        max(1, 
                            np.argmax(
                                self.veh.mcKwOutArray > min(
                                    self.veh.maxMotorKw - 0.01, 
                                    self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i]
                                )
                            ) - 1
                        )
                    ]
                )

        self.curMaxMechMcKwIn[i] = min(
            self.essLimMcRegenKw[i], self.veh.maxMotorKw)
        self.curMaxTracKw[i] = (
            self.veh.wheelCoefOfFric * self.veh.driveAxleWeightFrac * self.veh.vehKg * self.props.gravityMPerSec2
            / (1 + self.veh.vehCgM * self.veh.wheelCoefOfFric / self.veh.wheelBaseM) / 1_000 * self.maxTracMps[i]
        )

        if self.veh.fcEffType == H2FC:

            if self.veh.noElecSys or self.veh.noElecAux or self.highAccFcOnTag[i]:
                self.curMaxTransKwOut[i] = min(
                    (self.curMaxMcKwOut[i] - self.auxInKw[i]) * self.veh.transEff, 
                    self.curMaxTracKw[i] / self.veh.transEff
                )

            else:
                self.curMaxTransKwOut[i] = min(
                    (self.curMaxMcKwOut[i] - min(self.curMaxElecKw[i], 0)) * self.veh.transEff, 
                    self.curMaxTracKw[i] / self.veh.transEff
                )

        else:

            if self.veh.noElecSys or self.veh.noElecAux or self.highAccFcOnTag[i]:
                self.curMaxTransKwOut[i] = min(
                    (self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] - self.auxInKw[i]) * self.veh.transEff, 
                    self.curMaxTracKw[i] / self.veh.transEff
                )

            else:
                self.curMaxTransKwOut[i] = min(
                    (self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] - min(self.curMaxElecKw[i], 0)) * self.veh.transEff, 
                    self.curMaxTracKw[i] / self.veh.transEff
                )
        
    def set_power_calcs(self, i):
        """Calculate power requirements to meet cycle and determine if
        cycle can be met.  
        Arguments
        ------------
        i: index of time step"""

        if self.newton_iters[i] > 0:
            mpsAch = self.mpsAch[i]
        else:
            mpsAch = self.cyc.mps[i]

        self.cycDragKw[i] = 0.5 * self.props.airDensityKgPerM3 * self.veh.dragCoef * self.veh.frontalAreaM2 * (
            (self.mpsAch[i-1] + mpsAch) / 2.0) ** 3 / 1_000
        self.cycAccelKw[i] = self.veh.vehKg / (2.0 * self.cyc.dt_s[i]) * (mpsAch ** 2 - self.mpsAch[i-1] ** 2) / 1_000
        self.cycAscentKw[i] = self.props.gravityMPerSec2 * np.sin(np.arctan(
            self.cyc.grade[i])) * self.veh.vehKg * ((self.mpsAch[i-1] + mpsAch) / 2.0) / 1_000
        self.cycTracKwReq[i] = self.cycDragKw[i] + \
            self.cycAccelKw[i] + self.cycAscentKw[i]
        self.spareTracKw[i] = self.curMaxTracKw[i] - self.cycTracKwReq[i]
        self.cycRrKw[i] = self.veh.vehKg * self.props.gravityMPerSec2 * self.veh.wheelRrCoef * np.cos(
            np.arctan(self.cyc.grade[i])) * (self.mpsAch[i-1] + mpsAch) / 2.0 / 1_000
        self.cycWheelRadPerSec[i] = mpsAch / self.veh.wheelRadiusM
        self.cycTireInertiaKw[i] = (
            0.5 * self.veh.wheelInertiaKgM2 * self.veh.numWheels * self.cycWheelRadPerSec[i] ** 2.0 / self.cyc.dt_s[i] -
            0.5 * self.veh.wheelInertiaKgM2 * self.veh.numWheels * (self.mpsAch[i-1] / self.veh.wheelRadiusM) ** 2.0 / self.cyc.dt_s[i]
        ) / 1_000

        self.cycWheelKwReq[i] = self.cycTracKwReq[i] + self.cycRrKw[i] + self.cycTireInertiaKw[i]
        self.regenContrLimKwPerc[i] = self.veh.maxRegen / (1 + self.veh.regenA * np.exp(-self.veh.regenB * (
            (self.cyc.mph[i] + self.mpsAch[i-1] * params.MPH_PER_MPS) / 2.0 + 1 - 0)))
        self.cycRegenBrakeKw[i] = max(min(
                self.curMaxMechMcKwIn[i] * self.veh.transEff, 
                self.regenContrLimKwPerc[i] * -self.cycWheelKwReq[i]), 
            0
        )
        self.cycFricBrakeKw[i] = -min(self.cycRegenBrakeKw[i] + self.cycWheelKwReq[i], 0)
        self.cycTransKwOutReq[i] = self.cycWheelKwReq[i] + self.cycFricBrakeKw[i]

        if self.cycTransKwOutReq[i] <= self.curMaxTransKwOut[i]:
            self.cycMet[i] = True
            self.transKwOutAch[i] = self.cycTransKwOutReq[i]

        else:
            self.cycMet[i] = False
            self.transKwOutAch[i] = self.curMaxTransKwOut[i]

        if self.transKwOutAch[i] > 0:
            self.transKwInAch[i] = self.transKwOutAch[i] / self.veh.transEff
        else:
            self.transKwInAch[i] = self.transKwOutAch[i] * self.veh.transEff

        if self.cycMet[i]:

            if self.veh.fcEffType == H2FC:
                self.minMcKw2HelpFc[i] = max(
                    self.transKwInAch[i], -self.curMaxMechMcKwIn[i])

            else:
                self.minMcKw2HelpFc[i] = max(
                    self.transKwInAch[i] - self.curMaxFcKwOut[i], -self.curMaxMechMcKwIn[i])
        else:
            self.minMcKw2HelpFc[i] = max(
                self.curMaxMcKwOut[i], -self.curMaxMechMcKwIn[i])

    def set_ach_speed(self, i):
        """Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
        Arguments
        ------------
        i: index of time step"""

        # Cycle is met
        if self.cycMet[i]:
            self.mpsAch[i] = self.cyc.mps[i]

        #Cycle is not met
        else:
            def newton_mps_estimate(Totals):
                t3 = Totals[0]
                t2 = Totals[1]
                t1 = Totals[2]
                t0 = Totals[3]
                xs = []
                ys = []
                ms = []
                bs = []
                # initial guess
                xi = max(1.0, self.mpsAch[i-1])
                # stop criteria
                max_iter = self.sim_params.newton_max_iter
                xtol = self.sim_params.newton_xtol
                # solver gain
                g = self.sim_params.newton_gain
                yi = t3 * xi ** 3 + t2 * xi ** 2 + t1 * xi + t0
                mi = 3 * t3 * xi ** 2 + 2 * t2 * xi + t1
                bi = yi - xi * mi
                xs.append(xi)
                ys.append(yi)
                ms.append(mi)
                bs.append(bi)
                iterate = 1
                converged = False
                while iterate < max_iter and not(converged):
                    xi = xs[-1] * (1 - g) - g * bs[-1] / ms[-1]
                    yi = t3 * xi ** 3 + t2 * xi ** 2 + t1 * xi + t0
                    mi = 3 * t3 * xi ** 2 + 2 * t2 * xi + t1
                    bi = yi - xi * mi
                    xs.append(xi)
                    ys.append(yi)
                    ms.append(mi)
                    bs.append(bi)
                    converged = abs((xs[-1] - xs[-2]) / xs[-2]) < xtol 
                    iterate += 1
                
                self.newton_iters[i] = iterate

                _ys = [abs(y) for y in ys]
                return xs[_ys.index(min(_ys))]

            Drag3 = 1.0 / 16.0 * self.props.airDensityKgPerM3 * \
                self.veh.dragCoef * self.veh.frontalAreaM2
            Accel2 = 0.5 * self.veh.vehKg / self.cyc.dt_s[i]
            Drag2 = 3.0 / 16.0 * self.props.airDensityKgPerM3 * \
                self.veh.dragCoef * self.veh.frontalAreaM2 * self.mpsAch[i-1]
            Wheel2 = 0.5 * self.veh.wheelInertiaKgM2 * \
                self.veh.numWheels / (self.cyc.dt_s[i] * self.veh.wheelRadiusM ** 2)
            Drag1 = 3.0 / 16.0 * self.props.airDensityKgPerM3 * self.veh.dragCoef * \
                self.veh.frontalAreaM2 * self.mpsAch[i-1] ** 2
            Roll1 = 0.5 * self.veh.vehKg * self.props.gravityMPerSec2 * self.veh.wheelRrCoef \
                * np.cos(np.arctan(self.cyc.grade[i])) 
            Ascent1 = 0.5 * self.props.gravityMPerSec2 * \
                np.sin(np.arctan(self.cyc.grade[i])) * self.veh.vehKg 
            Accel0 = -0.5 * self.veh.vehKg * self.mpsAch[i-1] ** 2 / self.cyc.dt_s[i]
            Drag0 = 1.0 / 16.0 * self.props.airDensityKgPerM3 * self.veh.dragCoef * \
                self.veh.frontalAreaM2 * self.mpsAch[i-1] ** 3
            Roll0 = 0.5 * self.veh.vehKg * self.props.gravityMPerSec2 * \
                self.veh.wheelRrCoef * np.cos(np.arctan(self.cyc.grade[i])) \
                * self.mpsAch[i-1]
            Ascent0 = 0.5 * self.props.gravityMPerSec2 * np.sin(np.arctan(self.cyc.grade[i])) \
                * self.veh.vehKg * self.mpsAch[i-1] 
            Wheel0 = -0.5 * self.veh.wheelInertiaKgM2 * self.veh.numWheels * \
                self.mpsAch[i-1] ** 2 / (self.cyc.dt_s[i] * self.veh.wheelRadiusM ** 2)

            Total3 = Drag3 / 1_000
            Total2 = (Accel2 + Drag2 + Wheel2) / 1_000
            Total1 = (Drag1 + Roll1 + Ascent1) / 1_000
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / 1_000 - self.curMaxTransKwOut[i]

            Total = np.array([Total3, Total2, Total1, Total0])
            self.mpsAch[i] = newton_mps_estimate(Total)
            self.set_power_calcs(i)

        self.mphAch[i] = self.mpsAch[i] * params.MPH_PER_MPS
        self.distMeters[i] = self.mpsAch[i] * self.cyc.dt_s[i]
        self.distMiles[i] = self.distMeters[i] * (1.0 / params.metersPerMile)

    def set_hybrid_cont_calcs(self, i):
        """Hybrid control calculations.  
        Arguments
        ------------
        i: index of time step"""

        if self.veh.noElecSys:
            self.regenBufferSoc[i] = 0

        elif self.veh.chargingOn:
            self.regenBufferSoc[i] = max(
                self.veh.maxSoc - (self.veh.maxRegenKwh / self.veh.maxEssKwh), (self.veh.maxSoc + self.veh.minSoc) / 2)

        else:
            self.regenBufferSoc[i] = max(
                (self.veh.maxEssKwh * self.veh.maxSoc - 
                    0.5 * self.veh.vehKg * (self.cyc.mps[i] ** 2) * (1.0 / 1_000) * (1.0 / 3_600) * 
                    self.veh.mcPeakEff * self.veh.maxRegen) / self.veh.maxEssKwh, 
                self.veh.minSoc
            )

            self.essRegenBufferDischgKw[i] = min(self.curMaxEssKwOut[i], max(
                0, (self.soc[i-1] - self.regenBufferSoc[i]) * self.veh.maxEssKwh * 3_600 / self.cyc.dt_s[i]))

            self.maxEssRegenBufferChgKw[i] = min(max(
                    0, 
                    (self.regenBufferSoc[i] - self.soc[i-1]) * self.veh.maxEssKwh * 3.6e3 / self.cyc.dt_s[i]), 
                self.curMaxEssChgKw[i]
            )

        if self.veh.noElecSys:
            self.accelBufferSoc[i] = 0

        else:
            self.accelBufferSoc[i] = min(
                max(
                    ((self.veh.maxAccelBufferMph / params.MPH_PER_MPS) ** 2 - self.cyc.mps[i] ** 2) / 
                    (self.veh.maxAccelBufferMph / params.MPH_PER_MPS) ** 2 * min(
                        self.veh.maxAccelBufferPercOfUseableSoc * (self.veh.maxSoc - self.veh.minSoc), 
                        self.veh.maxRegenKwh / self.veh.maxEssKwh
                    ) * self.veh.maxEssKwh / self.veh.maxEssKwh + self.veh.minSoc, 
                    self.veh.minSoc
                ), 
                self.veh.maxSoc
                )

            self.essAccelBufferChgKw[i] = max(
                0, (self.accelBufferSoc[i] - self.soc[i-1]) * self.veh.maxEssKwh * 3.6e3 / self.cyc.dt_s[i])
            self.maxEssAccelBufferDischgKw[i] = min(
                max(
                    0, 
                    (self.soc[i-1] - self.accelBufferSoc[i]) * self.veh.maxEssKwh * 3_600 / self.cyc.dt_s[i]), 
                self.curMaxEssKwOut[i]
            )

        if self.regenBufferSoc[i] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(
                min(
                    (self.soc[i-1] - (self.regenBufferSoc[i] + self.accelBufferSoc[i]) / 2) * self.veh.maxEssKwh * 3.6e3 / self.cyc.dt_s[i], 
                    self.curMaxEssKwOut[i]
                ), 
                -self.curMaxEssChgKw[i]
            )

        elif self.soc[i-1] > self.regenBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(
                min(
                    self.essRegenBufferDischgKw[i], 
                    self.curMaxEssKwOut[i]), 
                -self.curMaxEssChgKw[i]
            )

        elif self.soc[i-1] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(
                min(-1.0 * self.essAccelBufferChgKw[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        else:
            self.essAccelRegenDischgKw[i] = max(
                min(0, self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        self.fcKwGapFrEff[i] = abs(self.transKwOutAch[i] - self.veh.maxFcEffKw)

        if self.veh.noElecSys:
            self.mcElectInKwForMaxFcEff[i] = 0

        elif self.transKwOutAch[i] < self.veh.maxFcEffKw:
            if self.fcKwGapFrEff[i] == self.veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = -self.fcKwGapFrEff[i] / self.veh.mcFullEffArray[-1]
            else:
                self.mcElectInKwForMaxFcEff[i] = (-self.fcKwGapFrEff[i] / 
                    self.veh.mcFullEffArray[max(1, 
                        np.argmax(self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1)]
                )

        else:
            if self.fcKwGapFrEff[i] == self.veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = self.veh.mcKwInArray[len(
                    self.veh.mcKwInArray) - 1]
            else:
                self.mcElectInKwForMaxFcEff[i] = self.veh.mcKwInArray[np.argmax(
                    self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1]

        if self.veh.noElecSys:
            self.electKwReq4AE[i] = 0

        elif self.transKwInAch[i] > 0:
            if self.transKwInAch[i] == self.veh.maxMotorKw:
                self.electKwReq4AE[i] = self.transKwInAch[i] / self.veh.mcFullEffArray[-1] + self.auxInKw[i]
            else:
                self.electKwReq4AE[i] = (self.transKwInAch[i] / 
                    self.veh.mcFullEffArray[max(1, np.argmax(
                        self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.transKwInAch[i])) - 1)] + self.auxInKw[i]
                )

        else:
            self.electKwReq4AE[i] = 0

        self.prevfcTimeOn[i] = self.fcTimeOn[i-1]

        # some conditions in the following if statement have a buffer of 1e-6 to prevent false positives/negatives because these have been encountered in practice.   
        if self.veh.maxFuelConvKw == 0:
            self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and  \
                (self.transKwInAch[i] - 1e-6) <= self.curMaxMcKwOut[i] and \
                (self.electKwReq4AE[i] < self.curMaxElecKw[i] or self.veh.maxFuelConvKw == 0)

        else:
            self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and \
                (self.transKwInAch[i] - 1e-6) <= self.curMaxMcKwOut[i] and \
                (self.electKwReq4AE[i] < self.curMaxElecKw[i] or self.veh.maxFuelConvKw == 0) \
                and ((self.cyc.mph[i] - 1e-6) <= self.veh.mphFcOn or self.veh.chargingOn) and \
                self.electKwReq4AE[i] <= self.veh.kwDemandFcOn

        if self.canPowerAllElectrically[i]:

            if self.transKwInAch[i] < self.auxInKw[i]:
                self.desiredEssKwOutForAE[i] = self.auxInKw[i] + \
                    self.transKwInAch[i]

            elif self.regenBufferSoc[i] < self.accelBufferSoc[i]:
                self.desiredEssKwOutForAE[i] = self.essAccelRegenDischgKw[i]

            elif self.soc[i-1] > self.regenBufferSoc[i]:
                self.desiredEssKwOutForAE[i] = self.essRegenBufferDischgKw[i]

            elif self.soc[i-1] < self.accelBufferSoc[i]:
                self.desiredEssKwOutForAE[i] = -self.essAccelBufferChgKw[i]

            else:
                self.desiredEssKwOutForAE[i] = self.transKwInAch[i] + \
                    self.auxInKw[i] - self.curMaxRoadwayChgKw[i]

        else:   
            self.desiredEssKwOutForAE[i] = 0

        if self.canPowerAllElectrically[i]:
            self.essAEKwOut[i] = max(
                -self.curMaxEssChgKw[i], 
                -self.maxEssRegenBufferChgKw[i], 
                min(0, self.curMaxRoadwayChgKw[i] - self.transKwInAch[i] + self.auxInKw[i]), 
                min(self.curMaxEssKwOut[i], self.desiredEssKwOutForAE[i])
            )

        else:
            self.essAEKwOut[i] = 0

        self.erAEKwOut[i] = min(
            max(0, self.transKwInAch[i] + self.auxInKw[i] - self.essAEKwOut[i]), 
            self.curMaxRoadwayChgKw[i])
    
    def set_fc_forced_state(self, i):
        """Calculate control variables related to engine on/off state
        Arguments       
        ------------
        i: index of time step"""
        # force fuel converter on if it was on in the previous time step, but only if fc
        # has not been on longer than minFcTimeOn
        if self.prevfcTimeOn[i] > 0 and self.prevfcTimeOn[i] < self.veh.minFcTimeOn - self.cyc.dt_s[i]:
            self.fcForcedOn[i] = True
        else:
            self.fcForcedOn[i] = False

        if not(self.fcForcedOn[i]) or not(self.canPowerAllElectrically[i]):
            self.fcForcedState[i] = 1
            self.mcMechKw4ForcedFc[i] = 0

        elif self.transKwInAch[i] < 0:
            self.fcForcedState[i] = 2
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i]

        elif self.veh.maxFcEffKw == self.transKwInAch[i]:
            self.fcForcedState[i] = 3
            self.mcMechKw4ForcedFc[i] = 0

        elif self.veh.idleFcKw > self.transKwInAch[i] and self.cycAccelKw[i] >= 0:
            self.fcForcedState[i] = 4
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - self.veh.idleFcKw

        elif self.veh.maxFcEffKw > self.transKwInAch[i]:
            self.fcForcedState[i] = 5
            self.mcMechKw4ForcedFc[i] = 0

        else:
            self.fcForcedState[i] = 6
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - \
                self.veh.maxFcEffKw

    def set_hybrid_cont_decisions(self, i):
        """Hybrid control decisions.
        Arguments
        ------------
        i: index of time step"""

        if (-self.mcElectInKwForMaxFcEff[i] - self.curMaxRoadwayChgKw[i]) > 0:
            self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] -
                                            self.curMaxRoadwayChgKw[i]) * self.veh.essDischgToFcMaxEffPerc

        else:
            self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] - \
                                            self.curMaxRoadwayChgKw[i]) * self.veh.essChgToFcMaxEffPerc

        if self.accelBufferSoc[i] > self.regenBufferSoc[i]:
            self.essKwIfFcIsReq[i] = min(
                self.curMaxEssKwOut[i], 
                self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                max(-self.curMaxEssChgKw[i], self.essAccelRegenDischgKw[i]))

        elif self.essRegenBufferDischgKw[i] > 0:
            self.essKwIfFcIsReq[i] = min(
                self.curMaxEssKwOut[i], 
                self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                max(-self.curMaxEssChgKw[i], 
                    min(self.essAccelRegenDischgKw[i], 
                        self.mcElecInLimKw[i] + self.auxInKw[i], 
                        max(self.essRegenBufferDischgKw[i], self.essDesiredKw4FcEff[i])
                    )
                )
            )

        elif self.essAccelBufferChgKw[i] > 0:
            self.essKwIfFcIsReq[i] = min(
                self.curMaxEssKwOut[i], 
                self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                max(-self.curMaxEssChgKw[i], 
                    max(-1 * self.maxEssRegenBufferChgKw[i], 
                        min(-self.essAccelBufferChgKw[i], self.essDesiredKw4FcEff[i])
                    )
                )
            )

        elif self.essDesiredKw4FcEff[i] > 0:
            self.essKwIfFcIsReq[i] = min(
                self.curMaxEssKwOut[i], 
                self.veh.mcMaxElecInKw + self.auxInKw[i], 
                self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                max(-self.curMaxEssChgKw[i], 
                    min(self.essDesiredKw4FcEff[i], self.maxEssAccelBufferDischgKw[i])
                )
            )

        else:
            self.essKwIfFcIsReq[i] = min(
                self.curMaxEssKwOut[i], 
                self.veh.mcMaxElecInKw + self.auxInKw[i], 
                self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                max(-self.curMaxEssChgKw[i], 
                    max(self.essDesiredKw4FcEff[i], -self.maxEssRegenBufferChgKw[i])
                )
            )

        self.erKwIfFcIsReq[i] = max(0, 
            min(
                self.curMaxRoadwayChgKw[i], self.curMaxMechMcKwIn[i],
                self.essKwIfFcIsReq[i] - self.mcElecInLimKw[i] + self.auxInKw[i]
            )
        )

        self.mcElecKwInIfFcIsReq[i] = self.essKwIfFcIsReq[i] + self.erKwIfFcIsReq[i] - self.auxInKw[i]

        if self.veh.noElecSys:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] == 0:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] > 0:

            if self.mcElecKwInIfFcIsReq[i] == max(self.veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * self.veh.mcFullEffArray[-1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * self.veh.mcFullEffArray[
                    max(1, np.argmax(
                            self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i])
                        ) - 1
                    )
                ]

        else:
            if self.mcElecKwInIfFcIsReq[i] * -1 == max(self.veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / self.veh.mcFullEffArray[-1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / (self.veh.mcFullEffArray[
                        max(1, np.argmax(
                            self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i] * -1)) - 1
                        )
                    ]
                )

        if self.veh.maxMotorKw == 0:
            self.mcMechKwOutAch[i] = 0

        elif self.fcForcedOn[i] and self.canPowerAllElectrically[i] and (self.veh.vehPtType == HEV or self.veh.vehPtType == PHEV) and (self.veh.fcEffType != H2FC):
            self.mcMechKwOutAch[i] = self.mcMechKw4ForcedFc[i]

        elif self.transKwInAch[i] <= 0:

            if self.veh.fcEffType !=H2FC and self.veh.maxFuelConvKw > 0:
                if self.canPowerAllElectrically[i] == 1:
                    self.mcMechKwOutAch[i] = - \
                        min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i])
                else:
                    self.mcMechKwOutAch[i] = min(
                        -min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]),
                        max(-self.curMaxFcKwOut[i], self.mcKwIfFcIsReq[i])
                    )
            else:
                self.mcMechKwOutAch[i] = min(
                    -min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]), 
                    -self.transKwInAch[i]
                )

        elif self.canPowerAllElectrically[i] == 1:
            self.mcMechKwOutAch[i] = self.transKwInAch[i]

        else:
            self.mcMechKwOutAch[i] = max(self.minMcKw2HelpFc[i], self.mcKwIfFcIsReq[i])

        if self.mcMechKwOutAch[i] == 0:
            self.mcElecKwInAch[i] = 0.0

        elif self.mcMechKwOutAch[i] < 0:

            if self.mcMechKwOutAch[i] * -1 == max(self.veh.mcKwInArray):
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * self.veh.mcFullEffArray[-1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * self.veh.mcFullEffArray[
                    max(1, np.argmax(self.veh.mcKwInArray > min(
                        max(self.veh.mcKwInArray) - 0.01, 
                        self.mcMechKwOutAch[i] * -1)) - 1
                    )
                ]

        else:
            if self.veh.maxMotorKw == self.mcMechKwOutAch[i]:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / self.veh.mcFullEffArray[-1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / self.veh.mcFullEffArray[
                    max(1, np.argmax(self.veh.mcKwOutArray > min(
                        self.veh.maxMotorKw - 0.01, 
                        self.mcMechKwOutAch[i])) - 1
                    )
                ]

        if self.curMaxRoadwayChgKw[i] == 0:
            self.roadwayChgKwOutAch[i] = 0

        elif self.veh.fcEffType == H2FC:
            self.roadwayChgKwOutAch[i] = max(
                0, 
                self.mcElecKwInAch[i], 
                self.maxEssRegenBufferChgKw[i], 
                self.essRegenBufferDischgKw[i], 
                self.curMaxRoadwayChgKw[i])

        elif self.canPowerAllElectrically[i] == 1:
            self.roadwayChgKwOutAch[i] = self.erAEKwOut[i]

        else:
            self.roadwayChgKwOutAch[i] = self.erKwIfFcIsReq[i]

        self.minEssKw2HelpFc[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - \
            self.curMaxFcKwOut[i] - self.roadwayChgKwOutAch[i]

        if self.veh.maxEssKw == 0 or self.veh.maxEssKwh == 0:
            self.essKwOutAch[i] = 0

        elif self.veh.fcEffType == H2FC:

            if self.transKwOutAch[i] >=0:
                self.essKwOutAch[i] = min(max(
                        self.minEssKw2HelpFc[i], 
                        self.essDesiredKw4FcEff[i], 
                        self.essAccelRegenDischgKw[i]),
                    self.curMaxEssKwOut[i], 
                    self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i]
                )

            else:
                self.essKwOutAch[i] = self.mcElecKwInAch[i] + \
                    self.auxInKw[i] - self.roadwayChgKwOutAch[i]

        elif self.highAccFcOnTag[i] or self.veh.noElecAux:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] - self.roadwayChgKwOutAch[i]

        else:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i]

        if self.veh.noElecSys:
            self.essCurKwh[i] = 0

        elif self.essKwOutAch[i] < 0:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * self.cyc.dt_s[i] / 3.6e3 * np.sqrt(self.veh.essRoundTripEff)

        else:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * self.cyc.dt_s[i] / 3.6e3 * (1 / np.sqrt(self.veh.essRoundTripEff))

        if self.veh.maxEssKwh == 0:
            self.soc[i] = 0.0

        else:
            self.soc[i] = self.essCurKwh[i] / self.veh.maxEssKwh

        if self.canPowerAllElectrically[i] and not(self.fcForcedOn[i]) and self.fcKwOutAch[i] == 0.0:
            self.fcTimeOn[i] = 0
        else:
            self.fcTimeOn[i] = self.fcTimeOn[i-1] + self.cyc.dt_s[i]
    
    def set_fc_power(self, i):
        """Sets fcKwOutAch and fcKwInAch.
        Arguments
        ------------
        i: index of time step"""

        if self.veh.maxFuelConvKw == 0:
            self.fcKwOutAch[i] = 0

        elif self.veh.fcEffType == H2FC:
            self.fcKwOutAch[i] = min(
                self.curMaxFcKwOut[i], 
                max(0, 
                    self.mcElecKwInAch[i] + self.auxInKw[i] - self.essKwOutAch[i] - self.roadwayChgKwOutAch[i]
                )
            )

        elif self.veh.noElecSys or self.veh.noElecAux or self.highAccFcOnTag[i]:
            self.fcKwOutAch[i] = min(
                self.curMaxFcKwOut[i], 
                max(
                    0, 
                    self.transKwInAch[i] - self.mcMechKwOutAch[i] + self.auxInKw[i]
                )
            )

        else:
            self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(
                0, self.transKwInAch[i] - self.mcMechKwOutAch[i]))

        if self.veh.maxFuelConvKw == 0:
            self.fcKwOutAch_pct[i] = 0
        else:
            self.fcKwOutAch_pct[i] = self.fcKwOutAch[i] / self.veh.maxFuelConvKw

        if self.fcKwOutAch[i] == 0:
            self.fcKwInAch[i] = 0
            self.fcKwOutAch_pct[i] = 0

        else:
            self.fcKwInAch[i] = (
                self.fcKwOutAch[i] / (self.veh.fcEffArray[np.argmax(
                    self.veh.fcKwOutArray > min(self.fcKwOutAch[i], self.veh.fcMaxOutkW)) - 1]) 
                if self.veh.fcEffArray[np.argmax(
                    self.veh.fcKwOutArray > min(self.fcKwOutAch[i], self.veh.fcMaxOutkW)) - 1] != 0
                else 0)

        self.fsKwOutAch[i] = self.fcKwInAch[i]

        self.fsKwhOutAch[i] = self.fsKwOutAch[i] * \
            self.cyc.dt_s[i] * (1 / 3.6e3)

    def set_time_dilation(self, i):
        trace_met = (
            ((abs(self.cyc0.dist_m[:i+1].sum() - self.distMeters[:i+1].sum()) / self.cyc0.dist_m[:i+1].sum()
            ) < self.sim_params.time_dilation_tol) or 
            (self.cyc.mps[i] == 0) # if prescribed speed is zero, trace is met to avoid div-by-zero errors and other possible wackiness
        )

        if not(trace_met):
            self.trace_miss_iters[i] += 1

            d_short = [self.cyc0.dist_m[:i+1].sum() - self.distMeters[:i+1].sum()] # positive if behind trace
            t_dilation = [
                0.0, # no time dilation initially
                min(max(
                        d_short[-1] / self.cyc0.dt_s[i] / self.mpsAch[i], # initial guess, speed that needed to be achived per speed that was achieved
                        self.sim_params.min_time_dilation
                    ),
                    self.sim_params.max_time_dilation
                ) 
            ]

            # add time dilation factor * step size to current and subsequent times
            print('time_s')
            print(type(self.cyc.time_s[i]))
            print(self.cyc.time_s[i])
            print('dt_s')
            print(type(self.cyc.dt_s[i]))
            print(self.cyc.dt_s[i])
            print('t_dilation')
            print(type(t_dilation[-1]))
            print(t_dilation[-1])
            self.cyc.time_s[i:] += self.cyc.dt_s[i] * t_dilation[-1]
            self.solve_step(i)
            trace_met = (
                # convergence criteria
                (abs(self.cyc0.dist_m[:i+1].sum() - self.distMeters[:i+1].sum()) / 
                    self.cyc0.dist_m[:i+1].sum() < self.sim_params.time_dilation_tol) or
                # exceeding max time dilation
                (t_dilation[-1] >= self.sim_params.max_time_dilation) or
                # lower than min time dilation
                (t_dilation[-1] <= self.sim_params.min_time_dilation)                    
            )

        while not(trace_met):
            # iterate newton's method until time dilation has converged or other exit criteria trigger trace_met == True
            # distance shortfall [m]            
            # correct time steps
            d_short.append(self.cyc0.dist_m[:i+1].sum() - self.distMeters[:i+1].sum())
            t_dilation.append(
                min(
                    max(
                        t_dilation[-1] - (t_dilation[-1] - t_dilation[-2]) / (d_short[-1] - d_short[-2]) * d_short[-1],
                        self.sim_params.min_time_dilation,
                    ),
                    self.sim_params.max_time_dilation
                )
            )
            self.cyc.time_s[i:] += self.cyc.dt_s[i] * t_dilation[-1]

            self.solve_step(i)
            self.trace_miss_iters[i] += 1

            trace_met = (
                # convergence criteria
                (abs(self.cyc0.dist_m[:i+1].sum() - self.distMeters[:i+1].sum()) / 
                    self.cyc0.dist_m[:i+1].sum() < self.sim_params.time_dilation_tol) or
                # max iterations
                (self.trace_miss_iters[i] >= self.sim_params.max_trace_miss_iters) or
                # exceeding max time dilation
                (t_dilation[-1] >= self.sim_params.max_time_dilation) or
                # lower than min time dilation
                (t_dilation[-1] <= self.sim_params.min_time_dilation)
            )
    
    def set_coast_speed(self, i):
        """
        Placeholder for method to impose coasting.
        Might be good to include logic for deciding when to coast.
        """
        pass

    def set_post_scalars(self):
        """Sets scalar variables that can be calculated after a cycle is run. 
        This includes mpgge, various energy metrics, and others"""
        
        self.fsCumuMjOutAch = (self.fsKwOutAch * self.cyc.dt_s).cumsum() * 1e-3

        if self.fsKwhOutAch.sum() == 0:
            self.mpgge = 0.0

        else:
            self.mpgge = self.distMiles.sum() / (self.fsKwhOutAch.sum() / self.props.kWhPerGGE)

        self.roadwayChgKj = (self.roadwayChgKwOutAch * self.cyc.dt_s).sum()
        self.essDischgKj = - \
            (self.soc[-1] - self.soc[0]) * self.veh.maxEssKwh * 3.6e3
        self.battery_kWh_per_mi  = (
            self.essDischgKj / 3.6e3) / self.distMiles.sum()
        self.electric_kWh_per_mi  = (
            (self.roadwayChgKj + self.essDischgKj) / 3.6e3) / self.distMiles.sum()
        self.fuelKj = (self.fsKwOutAch * self.cyc.dt_s).sum()

        if (self.fuelKj + self.roadwayChgKj) == 0:
            self.ess2fuelKwh  = 1.0

        else:
            self.ess2fuelKwh  = self.essDischgKj / (self.fuelKj + self.roadwayChgKj)

        if self.mpgge == 0:
            # hardcoded conversion
            self.Gallons_gas_equivalent_per_mile = self.electric_kWh_per_mi / self.props.kWhPerGGE
            grid_Gallons_gas_equivalent_per_mile = self.electric_kWh_per_mi / self.props.kWhPerGGE / \
                self.veh.chgEff

        else:
            self.Gallons_gas_equivalent_per_mile = 1 / \
                self.mpgge + self.electric_kWh_per_mi  / self.props.kWhPerGGE
            grid_Gallons_gas_equivalent_per_mile = 1 / self.mpgge + \
                self.electric_kWh_per_mi / self.props.kWhPerGGE / self.veh.chgEff

        self.grid_mpgge_elec = 1 / grid_Gallons_gas_equivalent_per_mile
        self.mpgge_elec = 1 / self.Gallons_gas_equivalent_per_mile

        # energy audit calcs
        self.dragKw = self.cycDragKw 
        self.dragKj = (self.dragKw * self.cyc.dt_s).sum()
        self.ascentKw = self.cycAscentKw
        self.ascentKj = (self.ascentKw * self.cyc.dt_s).sum()
        self.rrKw = self.cycRrKw
        self.rrKj = (self.rrKw * self.cyc.dt_s).sum()

        self.essLossKw[1:] = np.array(
            [0 if (self.veh.maxEssKw == 0 or self.veh.maxEssKwh == 0)
            else -self.essKwOutAch[i] - (-self.essKwOutAch[i] * np.sqrt(self.veh.essRoundTripEff))
                if self.essKwOutAch[i] < 0
            else self.essKwOutAch[i] * (1.0 / np.sqrt(self.veh.essRoundTripEff)) - self.essKwOutAch[i]
            for i in range(1, len(self.cyc.time_s))]
        )
        
        self.brakeKj = (self.cycFricBrakeKw * self.cyc.dt_s).sum()
        self.transKj = ((self.transKwInAch - self.transKwOutAch) * self.cyc.dt_s).sum()
        self.mcKj = ((self.mcElecKwInAch - self.mcMechKwOutAch) * self.cyc.dt_s).sum()
        self.essEffKj = (self.essLossKw * self.cyc.dt_s).sum()
        self.auxKj = (self.auxInKw * self.cyc.dt_s).sum()
        self.fcKj = ((self.fcKwInAch - self.fcKwOutAch) * self.cyc.dt_s).sum()
        
        self.netKj = self.dragKj + self.ascentKj + self.rrKj + self.brakeKj + self.transKj \
            + self.mcKj + self.essEffKj + self.auxKj + self.fcKj

        self.keKj = 0.5 * self.veh.vehKg * \
            (self.mpsAch[0] ** 2 - self.mpsAch[-1] ** 2) / 1_000
        
        self.energyAuditError = ((self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj) - self.netKj
            ) / (self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj)

        if (np.abs(self.energyAuditError) > self.sim_params.energy_audit_error_tol) and \
            self.sim_params.verbose:
            print('Warning: There is a problem with conservation of energy.')
            print('Energy Audit Error:', np.round(self.energyAuditError, 5))

        self.accelKw[1:] = (self.veh.vehKg / (2.0 * (self.cyc.dt_s[1:]))) * (
            self.mpsAch[1:] ** 2 - self.mpsAch[:-1] ** 2) / 1_000

        self.trace_miss = False
        self.trace_miss_dist_frac = abs(self.distMeters.sum() - self.cyc0.dist_m.sum()) / self.cyc0.dist_m.sum()
        self.trace_miss_time_frac = abs(self.cyc.time_s[-1] - self.cyc0.time_s[-1]) / self.cyc0.time_s[-1]

        if not(self.sim_params.missed_trace_correction):
            if self.trace_miss_dist_frac > self.sim_params.trace_miss_dist_tol:
                self.trace_miss = True
                if self.sim_params.verbose:
                    print('Warning: Trace miss distance fraction:', np.round(self.trace_miss_dist_frac, 5))
                    print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_dist_tol, 5))
        else:
            if self.trace_miss_time_frac > self.sim_params.trace_miss_time_tol:
                self.trace_miss = True
                if self.sim_params.verbose:
                    print('Warning: Trace miss time fraction:', np.round(self.trace_miss_time_frac, 5))
                    print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_time_tol, 5))

        self.trace_miss_speed_mps = max([
            abs(self.mpsAch[i] - self.cyc.mps[i]) for i in range(len(self.cyc.time_s))
        ])
        if self.trace_miss_speed_mps > self.sim_params.trace_miss_speed_mps_tol:
            self.trace_miss = True
            if self.sim_params.verbose:
                print('Warning: Trace miss speed [m/s]:', np.round(self.trace_miss_speed_mps, 5))
                print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_speed_mps_tol, 5))
        

class SimAccelTest(SimDriveClassic):
    """Class for running FASTSim vehicle acceleration simulation."""

    def sim_drive(self):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType."""

        if self.veh.vehPtType == CONV:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_walk(initSoc)

        elif self.veh.vehPtType == HEV:  # HEV

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_walk(initSoc)

        else:

            # If EV, initializing initial SOC to maximum SOC.
            initSoc = self.veh.maxSoc
            self.sim_drive_walk(initSoc)

        self.set_post_scalars()


class SimDrivePost(object):
    """Class for post-processing of SimDrive instance.  Requires already-run 
    SimDriveJit or SimDriveClassic instance."""
    
    def __init__(self, sim_drive):
        """Arguments:
        ---------------
        sim_drive: solved sim_drive object"""
        
        for item in utils.get_attrs(sim_drive):
            self.__setattr__(item, sim_drive.__getattribute__(item))

    def get_output(self):
        """Calculate finalized results
        Arguments
        ------------
        initSoc: initial SOC for electrified vehicles
        
        Returns
        ------------
        output: dict of summary output variables"""

        output = {}

        output['mpgge'] = self.mpgge
        output['battery_kWh_per_mi'] = self.battery_kWh_per_mi
        output['electric_kWh_per_mi'] = self.electric_kWh_per_mi
        output['maxTraceMissMph'] = params.MPH_PER_MPS * max(abs(self.cyc.mps - self.mpsAch))
        self.maxTraceMissMph = output['maxTraceMissMph']

        output['ess2fuelKwh'] = self.ess2fuelKwh

        output['initial_soc'] = self.soc[0]
        output['final_soc'] = self.soc[-1]

        output['mpgge_elec'] = self.mpgge_elec
        output['soc'] = self.soc
        output['distance_mi'] = sum(self.distMiles)
        duration_sec = self.cyc.time_s[-1] - self.cyc.time_s[0]
        output['avg_speed_mph'] = sum(
            self.distMiles) / (duration_sec / 3.6e3)
        self.avg_speed_mph = output['avg_speed_mph']
        self.accel = np.diff(self.mphAch) / np.diff(self.cyc.time_s)
        output['avg_accel_mphps'] = np.mean(self.accel[self.accel > 0])
        self.avg_accel_mphps = output['avg_accel_mphps']

        if max(self.mphAch) > 60:
            output['ZeroToSixtyTime_secs'] = np.interp(60, self.mphAch, self.cyc.time_s)

        else:
            output['ZeroToSixtyTime_secs'] = 0.0

        output['fcKwOutAch'] = np.asarray(self.fcKwOutAch)
        output['fsKwhOutAch'] = np.asarray(self.fsKwhOutAch)
        output['fcKwInAch'] = np.asarray(self.fcKwInAch)
        output['time'] = np.asarray(self.cyc.time_s)

        return output

    # optional post-processing methods
    def get_diagnostics(self):
        """This method is to be run after runing sim_drive if diagnostic variables 
        are needed.  Diagnostic variables are returned in a dict.  Diagnostic variables include:
        - final integrated value of all positive powers
        - final integrated value of all negative powers
        - total distance traveled
        - miles per gallon gasoline equivalent (mpgge)"""
        
        base_var_list = list(self.__dict__.keys())
        pw_var_list = [var for var in base_var_list if re.search(
            '\w*Kw(?!h)\w*', var)] 
            # find all vars containing 'Kw' but not 'Kwh'
        
        prog = re.compile('(\w*)Kw(?!h)(\w*)') 
        # find all vars containing 'Kw' but not Kwh and capture parts before and after 'Kw'
        # using compile speeds up iteration

        # create positive and negative versions of all time series with units of kW
        # then integrate to find cycle end pos and negative energies
        tempvars = {} # dict for contaning intermediate variables
        output = {}
        for var in pw_var_list:
            tempvars[var + 'Pos'] = [x if x >= 0 
                                        else 0 
                                        for x in self.__getattribute__(var)]
            tempvars[var + 'Neg'] = [x if x < 0 
                                        else 0 
                                        for x in self.__getattribute__(var)]    
                        
            # Assign values to output dict for positive and negative energy variable names
            search = prog.search(var)
            output[search[1] + 'Kj' + search[2] + 'Pos'] = np.trapz(tempvars[var + 'Pos'], self.cyc.time_s)
            output[search[1] + 'Kj' + search[2] + 'Neg'] = np.trapz(tempvars[var + 'Neg'], self.cyc.time_s)
        
        output['distMilesFinal'] = sum(self.distMiles)
        if sum(self.fsKwhOutAch) > 0:
            output['mpgge'] = sum(self.distMiles) / sum(self.fsKwhOutAch) * self.props.kWhPerGGE
        else:
            output['mpgge'] = 0
    
        return output

    def set_battery_wear(self):
        """Battery wear calcs"""

        self.addKwh[1:] = np.array([
            (self.essCurKwh[i] - self.essCurKwh[i-1]) + self.addKwh[i-1]
            if self.essCurKwh[i] > self.essCurKwh[i-1]
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.dodCycs[1:] = np.array([
            self.addKwh[i-1] / self.veh.maxEssKwh if self.addKwh[i] == 0
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.essPercDeadArray = np.array([
            np.power(self.veh.essLifeCoefA, 1.0 / self.veh.essLifeCoefB) / np.power(self.dodCycs[i], 
            1.0 / self.veh.essLifeCoefB)
            if self.dodCycs[i] != 0
            else 0
            for i in range(0, len(self.essCurKwh))])


def SimDriveJit(cyc_jit, veh_jit):
    """
    deprecated
    """
    raise NotImplementedError("This function has been deprecated.")

def SimAccelTestJit(cyc_jit, veh_jit):
    """
    deprecated
    """
    raise NotImplementedError("This function has been deprecated")