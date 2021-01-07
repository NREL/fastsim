"""Module containing classes and methods for simulating vehicle drive
cycle. For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
import pandas as pd
import re
import sys
from fastsim import vehicle
from numba.experimental import jitclass                 # import the decorator
import warnings
warnings.simplefilter('ignore')

# local modules
from fastsim import parameters as params
from fastsim import cycle
from fastsim.cycle import CycleJit
from fastsim.vehicle import VehicleJit
from fastsim.buildspec import build_spec


class SimDriveParamsClassic(object):
    """Class containing attributes used for configuring sim_drive.
    Usually the defaults are ok, and there will be no need to use this.

    See comments in code for descriptions of various parameters that
    affect simulation behavior. If, for example, you want to suppress
    warning messages, use the following pastable code EXAMPLE:

    >>> cyc = cycle.Cycle('udds')
    >>> veh = vehicle.Vehicle(1)
    >>> sim_drive = simdrive.SimDriveClassic(cyc, veh)
    >>> sim_drive.sim_params.verbose = False # turn off error messages for large time steps
    >>> sim_drive.sim_drive()"""

    def __init__(self):
        """Default values that affect simulation behavior.  
        Can be modified after instantiation."""
        self.missed_trace_correction = False  # if True, missed trace correction is active, default = False
        self.max_time_dilation = 10  # maximum time dilation factor to "catch up" with trace
        self.min_time_dilation = 0.1  # minimum time dilation to let trace "catch up"
        self.time_dilation_tol = 1e-3  # convergence criteria for time dilation
        self.sim_count_max = 30  # max allowable number of HEV SOC iterations
        self.verbose = True  # show warning and other messages


param_spec = build_spec(SimDriveParamsClassic())

@jitclass(param_spec)
class SimDriveParams(SimDriveParamsClassic):
    pass


class SimDriveClassic(object):
    """Class containing methods for running FASTSim vehicle 
    fuel economy simulations. This class is not compiled and will 
    run slower for large batch runs.
    Arguments:
    ----------
    cyc: cycle.Cycle instance
    veh: vehicle.Vehicle instance"""

    def __init__(self, cyc, veh):
        """Initalizes arrays, given vehicle.Vehicle() and cycle.Cycle() as arguments.
        sim_params is needed only if non-default behavior is desired."""
        self.__init_objects__(cyc, veh)
        self.init_arrays()
        # initialized here for downstream classes that do not run sim_drive
        self.hev_sim_count = 0 

    def __init_objects__(self, cyc, veh):        
        self.veh = veh
        self.cyc = cyc.copy() # this cycle may be manipulated
        self.cyc0 = cyc.copy() # this cycle is not to be manipulated
        self.sim_params = SimDriveParamsClassic()
        self.props = params.PhysicalProperties()

    def init_arrays(self):
        len_cyc = len(self.cyc.cycSecs)
        self.i = 1 # initialize step counter for possible use outside sim_drive_walk()

        # Component Limits -- calculated dynamically"
        self.curMaxFsKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.fcTransLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcFsLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.fcMaxKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxFcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxAvailElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.essCapLimChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxEssChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxElecKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecInLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcTransiLimKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenPercKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLimMcRegenKw = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMechMcKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTransKwOut = np.zeros(len_cyc, dtype=np.float64)

        ### Drive Train
        self.cycDragKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAccelKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycAscentKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTracKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.spareTracKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycRrKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelRadPerSec = np.zeros(len_cyc, dtype=np.float64)
        self.cycTireInertiaKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycWheelKwReq = np.zeros(len_cyc, dtype=np.float64)
        self.regenContrLimKwPerc = np.zeros(len_cyc, dtype=np.float64)
        self.cycRegenBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycFricBrakeKw = np.zeros(len_cyc, dtype=np.float64)
        self.cycTransKwOutReq = np.zeros(len_cyc, dtype=np.float64)
        self.cycMet = np.zeros(len_cyc, dtype=np.float64)
        self.transKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.transKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.curSocTarget = np.zeros(len_cyc, dtype=np.float64)
        self.minMcKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.mcMechKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.auxInKw = np.zeros(len_cyc, dtype=np.float64)
        self.roadwayChgKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.minEssKw2HelpFc = np.zeros(len_cyc, dtype=np.float64)
        self.essKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwOutAch_pct = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwInAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsCumuMjOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.fsKwhOutAch = np.zeros(len_cyc, dtype=np.float64)
        self.essCurKwh = np.zeros(len_cyc, dtype=np.float64)
        self.soc = np.zeros(len_cyc, dtype=np.float64)

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.essRegenBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssRegenBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelBufferChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelBufferSoc = np.zeros(len_cyc, dtype=np.float64)
        self.maxEssAccelBufferDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.essAccelRegenDischgKw = np.zeros(len_cyc, dtype=np.float64)
        self.mcElectInKwForMaxFcEff = np.zeros(len_cyc, dtype=np.float64)
        self.electKwReq4AE = np.zeros(len_cyc, dtype=np.float64)
        self.canPowerAllElectrically = np.array(
            [False] * len_cyc, dtype=np.bool_)
        self.desiredEssKwOutForAE = np.zeros(len_cyc, dtype=np.float64)
        self.essAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.erAEKwOut = np.zeros(len_cyc, dtype=np.float64)
        self.essDesiredKw4FcEff = np.zeros(len_cyc, dtype=np.float64)
        self.essKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxMcElecKwIn = np.zeros(len_cyc, dtype=np.float64)
        self.fcKwGapFrEff = np.zeros(len_cyc, dtype=np.float64)
        self.erKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcElecKwInIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.mcKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float64)
        self.fcForcedOn = np.array([False] * len_cyc, dtype=np.bool_)
        self.fcForcedState = np.zeros(len_cyc, dtype=np.int32)
        self.mcMechKw4ForcedFc = np.zeros(len_cyc, dtype=np.float64)
        self.fcTimeOn = np.zeros(len_cyc, dtype=np.float64)
        self.prevfcTimeOn = np.zeros(len_cyc, dtype=np.float64)

        ### Additional Variables
        self.mpsAch = np.zeros(len_cyc, dtype=np.float64)
        self.mphAch = np.zeros(len_cyc, dtype=np.float64)
        self.distMeters = np.zeros(len_cyc, dtype=np.float64)
        self.distMiles = np.zeros(len_cyc, dtype=np.float64)
        self.highAccFcOnTag = np.zeros(len_cyc, dtype=np.float64)
        self.reachedBuff = np.zeros(len_cyc, dtype=np.float64)
        self.maxTracMps = np.zeros(len_cyc, dtype=np.float64)
        self.addKwh = np.zeros(len_cyc, dtype=np.float64)
        self.dodCycs = np.zeros(len_cyc, dtype=np.float64)
        self.essPercDeadArray = np.zeros(len_cyc, dtype=np.float64)
        self.dragKw = np.zeros(len_cyc, dtype=np.float64)
        self.essLossKw = np.zeros(len_cyc, dtype=np.float64)
        self.accelKw = np.zeros(len_cyc, dtype=np.float64)
        self.ascentKw = np.zeros(len_cyc, dtype=np.float64)
        self.rrKw = np.zeros(len_cyc, dtype=np.float64)
        self.motor_index_debug = np.zeros(len_cyc, dtype=np.float64)
        self.debug_flag = np.zeros(len_cyc, dtype=np.float64)
        self.curMaxRoadwayChgKw = np.zeros(len_cyc, dtype=np.float64)
        self.trace_miss_iters = np.zeros(len_cyc, dtype=np.float64)

    def sim_drive(self, initSoc=None, auxInKwOverride=np.zeros(1, dtype=np.float64)):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        initSoc: (optional) initial SOC for electrified vehicles.  
            Must be between 0 and 1.
        auxInKw: auxInKw override.  Array of same length as cyc.cycSecs.  
                Default of np.zeros(1) causes veh.auxKw to be used.
                If zero is actually desired as an override, either set
                veh.auxKw = 0 before instantiaton of SimDrive*, or use
                `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting the
                final value to non-zero prevents override mechanism.  
        """

        if (auxInKwOverride == 0).all():
            auxInKwOverride = self.auxInKw
        self.hev_sim_count = 0

        if initSoc != None:
            if initSoc > 1.0 or initSoc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = None
            else:
                self.sim_drive_walk(initSoc, auxInKwOverride)

        elif self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0

            self.sim_drive_walk(initSoc, auxInKwOverride)

        elif self.veh.vehPtType == 2 and initSoc == None:  # HEV

            #####################################
            ### Charge Balancing Vehicle SOC ###
            #####################################

            # Charge balancing SOC for PHEV vehicle types. Iterating initsoc and comparing to final SOC.
            # Iterating until tolerance met or 30 attempts made.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            ess2fuelKwh = 1.0
            while ess2fuelKwh > self.veh.essToFuelOkError and self.hev_sim_count < self.sim_params.sim_count_max:
                self.hev_sim_count += 1
                self.sim_drive_walk(initSoc, auxInKwOverride)
                fuelKj = np.sum(self.fsKwOutAch * self.cyc.secs)
                roadwayChgKj = np.sum(self.roadwayChgKwOutAch * self.cyc.secs)
                if (fuelKj + roadwayChgKj) > 0:
                    ess2fuelKwh = np.abs((self.soc[0] - self.soc[-1]) *
                                        self.veh.maxEssKwh * 3600 / (fuelKj + roadwayChgKj))
                else:
                    ess2fuelKwh = 0.0
                initSoc = min(1.0, max(0.0, self.soc[-1]))

            self.sim_drive_walk(initSoc, auxInKwOverride)

        elif (self.veh.vehPtType == 3 and initSoc == None) or (self.veh.vehPtType == 4 and initSoc == None):  # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = self.veh.maxSoc

            self.sim_drive_walk(initSoc, auxInKwOverride)

        else:

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
        auxInKw: auxInKw override.  Array of same length as cyc.cycSecs.  
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
        # in above, arguments must be explicit for numba
        if not((auxInKwOverride == 0).all()):
            self.auxInKw = auxInKwOverride
        
        self.cycMet[0] = 1
        self.curSocTarget[0] = self.veh.maxSoc
        self.essCurKwh[0] = initSoc * self.veh.maxEssKwh
        self.soc[0] = initSoc

        if self.sim_params.missed_trace_correction:
            self.cyc = self.cyc0.copy() # reset the cycle in case it has been manipulated

        self.i = 1 # time step counter
        while self.i < len(self.cyc.cycSecs):
            self.sim_drive_step()
        
        if self.sim_params.missed_trace_correction: 
            self.cyc.cycSecs = self.cyc.secs.cumsum() # correct cycSecs based on actual trace

        if (self.cyc.secs > 5).any() and self.sim_params.verbose:
            if self.sim_params.missed_trace_correction:
                print('Max time dilation factor =', (round(self.cyc.secs.max(), 3)))
            print("Warning: large time steps affect accuracy significantly.") 
            print("To suppress this message, view the doc string for simdrive.SimDriveParams.")
            print('Max time step =', (round(self.cyc.secs.max(), 3)))

    def sim_drive_step(self):
        """Step through 1 time step."""

        self.set_misc_calcs(self.i)
        self.set_comp_lims(self.i)
        self.set_power_calcs(self.i)
        self.set_ach_speed(self.i)
        self.set_hybrid_cont_calcs(self.i)
        self.set_fc_forced_state(self.i) # can probably be *mostly* done with list comprehension in post processing
        self.set_hybrid_cont_decisions(self.i)
        self.set_fc_power(self.i)

        if self.sim_params.missed_trace_correction:
            time_dilation_factor = [1, 1]
            if self.distMeters[self.i] > 0:
                time_dilation_factor[0] = 1.1
                time_dilation_factor[1] =  min(max(
                    (self.cyc0.cycDistMeters[:self.i].sum() - self.distMeters[:self.i].sum()
                     + self.distMeters[self.i]) / self.distMeters[self.i] + 1,
                    self.sim_params.min_time_dilation),
                    self.sim_params.max_time_dilation)

            # loop to iterate until time dilation factor converges
            while abs(time_dilation_factor[-1] - time_dilation_factor[-2]) / \
                abs(time_dilation_factor[-2]) > self.sim_params.time_dilation_tol:
                self.cyc.secs[self.i] = self.cyc0.secs[self.i] * \
                    time_dilation_factor[-1]
                self.cyc.cycDistMeters[self.i] = self.cyc.cycMps[self.i] * self.cyc.secs[self.i]
                self.set_misc_calcs(self.i)
                self.set_comp_lims(self.i)
                self.set_power_calcs(self.i)
                self.set_ach_speed(self.i)
                self.set_hybrid_cont_calcs(self.i)
                self.set_fc_forced_state(self.i)
                self.set_hybrid_cont_decisions(self.i)
                self.set_fc_power(self.i)
                time_dilation_factor.append(min(max(
                    (self.cyc0.cycDistMeters[:self.i].sum() - self.distMeters[:self.i].sum()
                     + self.distMeters[self.i]) / self.distMeters[self.i] + 1,
                    self.sim_params.min_time_dilation),
                    self.sim_params.max_time_dilation))
            
            self.trace_miss_iters[self.i] = len(time_dilation_factor)


        self.i += 1 # increment time step counter

    def set_misc_calcs(self, i):
        """Sets misc. calculations at time step 'i'
        Arguments:
        ----------
        i: index of time step"""
        # if cycle iteration is used, auxInKw must be re-zeroed to trigger the below if statement
        if (self.auxInKw[i:] == 0).all():
            # if all elements after i-1 are zero, trigger default behavior; otherwise, use override value 
            if self.veh.noElecAux == True:
                self.auxInKw[i] = self.veh.auxKw / self.veh.altEff
            else:
                self.auxInKw[i] = self.veh.auxKw            
        # Is SOC below min threshold?
        if self.soc[i-1] < (self.veh.minSoc + self.veh.percHighAccBuf):
            self.reachedBuff[i] = 0
        else:
            self.reachedBuff[i] = 1

        # Does the engine need to be on for low SOC or high acceleration
        if self.soc[i-1] < self.veh.minSoc or (self.highAccFcOnTag[i-1] == 1 and self.reachedBuff[i] == 0):
            self.highAccFcOnTag[i] = 1
        else:
            self.highAccFcOnTag[i] = 0
        self.maxTracMps[i] = self.mpsAch[i-1] + (self.veh.maxTracMps2 * self.cyc.secs[i])

    def set_comp_lims(self, i):
        """Sets component limits for time step 'i'
        Arguments
        ------------
        i: index of time step
        initSoc: initial SOC for electrified vehicles"""

        # max fuel storage power output
        self.curMaxFsKwOut[i] = min(self.veh.maxFuelStorKw, self.fsKwOutAch[i-1] + (
            (self.veh.maxFuelStorKw / self.veh.fuelStorSecsToPeakPwr) * (self.cyc.secs[i])))
        # maximum fuel storage power output rate of change
        self.fcTransLimKw[i] = self.fcKwOutAch[i-1] + \
            ((self.veh.maxFuelConvKw / self.veh.fuelConvSecsToPeakPwr) * (self.cyc.secs[i]))

        self.fcMaxKwIn[i] = min(self.curMaxFsKwOut[i], self.veh.maxFuelStorKw)
        self.fcFsLimKw[i] = self.veh.fcMaxOutkW
        self.curMaxFcKwOut[i] = min(
            self.veh.maxFuelConvKw, self.fcFsLimKw[i], self.fcTransLimKw[i])

        if self.veh.maxEssKwh == 0 or self.soc[i-1] < self.veh.minSoc:
            self.essCapLimDischgKw[i] = 0.0

        else:
            self.essCapLimDischgKw[i] = (
                self.veh.maxEssKwh * np.sqrt(self.veh.essRoundTripEff)) * 3600.0 * (self.soc[i-1] - self.veh.minSoc) / (self.cyc.secs[i])
        self.curMaxEssKwOut[i] = min(
            self.veh.maxEssKw, self.essCapLimDischgKw[i])

        if self.veh.maxEssKwh == 0 or self.veh.maxEssKw == 0:
            self.essCapLimChgKw[i] = 0

        else:
            self.essCapLimChgKw[i] = max(((self.veh.maxSoc - self.soc[i-1]) * self.veh.maxEssKwh * (1 /
                                                                                            np.sqrt(self.veh.essRoundTripEff))) / ((self.cyc.secs[i]) * (1 / 3600.0)), 0)

        self.curMaxEssChgKw[i] = min(self.essCapLimChgKw[i], self.veh.maxEssKw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if self.veh.fcEffType == 4:
            self.curMaxElecKw[i] = self.curMaxFcKwOut[i] + self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        else:
            self.curMaxElecKw[i] = self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        self.curMaxAvailElecKw[i] = min(
            self.curMaxElecKw[i], self.veh.mcMaxElecInKw)

        if self.curMaxElecKw[i] > 0:
            # limit power going into e-machine controller to
            if self.curMaxAvailElecKw[i] == max(self.veh.mcKwInArray):
                self.mcElecInLimKw[i] = min(
                    self.veh.mcKwOutArray[-1], self.veh.maxMotorKw)
            else:
                self.mcElecInLimKw[i] = min(
                    self.veh.mcKwOutArray[
                            np.argmax(self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) -
                                0.01, self.curMaxAvailElecKw[i])) - 1],
                    self.veh.maxMotorKw)
        else:
            self.mcElecInLimKw[i] = 0.0

        # Motor transient power limit
        self.mcTransiLimKw[i] = abs(
            self.mcMechKwOutAch[i-1]) + ((self.veh.maxMotorKw / self.veh.motorSecsToPeakPwr) * (self.cyc.secs[i]))

        self.curMaxMcKwOut[i] = max(
            min(self.mcElecInLimKw[i], self.mcTransiLimKw[i], 
            np.float64(0 if self.veh.stopStart else 1) * self.veh.maxMotorKw),
            -self.veh.maxMotorKw)

        if self.curMaxMcKwOut[i] == 0:
            self.curMaxMcElecKwIn[i] = 0
        else:
            if self.curMaxMcKwOut[i] == self.veh.maxMotorKw:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / \
                    self.veh.mcFullEffArray[-1]
            else:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / self.veh.mcFullEffArray[max(1, np.argmax(self.veh.mcKwOutArray
                                            > min(self.veh.maxMotorKw - 0.01, self.curMaxMcKwOut[i])) - 1)]

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
                self.essLimMcRegenKw[i] = min(self.veh.maxMotorKw, self.curMaxEssChgKw[i] / self.veh.mcFullEffArray
                                                [max(1, np.argmax(self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i])) - 1)])

        self.curMaxMechMcKwIn[i] = min(
            self.essLimMcRegenKw[i], self.veh.maxMotorKw)
        self.curMaxTracKw[i] = (((self.veh.wheelCoefOfFric * self.veh.driveAxleWeightFrac * self.veh.vehKg * self.props.gravityMPerSec2)
                                    / (1 + ((self.veh.vehCgM * self.veh.wheelCoefOfFric) / self.veh.wheelBaseM))) / 1000.0) * (self.maxTracMps[i])

        if self.veh.fcEffType == 4:

            if self.veh.noElecSys == True or self.veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = min(
                    (self.curMaxMcKwOut[i] - self.auxInKw[i]) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 1

            else:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] - min(
                    self.curMaxElecKw[i], 0)) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 2

        else:

            if self.veh.noElecSys == True or self.veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                self.auxInKw[i]) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 3

            else:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                min(self.curMaxElecKw[i], 0)) * self.veh.transEff, self.curMaxTracKw[i] / self.veh.transEff)
                self.debug_flag[i] = 4
        
    def set_power_calcs(self, i):
        """Calculate power requirements to meet cycle and determine if
        cycle can be met.  
        Arguments
        ------------
        i: index of time step"""

        self.cycDragKw[i] = 0.5 * self.props.airDensityKgPerM3 * self.veh.dragCoef * \
            self.veh.frontalAreaM2 * \
            (((self.mpsAch[i-1] + self.cyc.cycMps[i]) / 2.0)**3) / 1000.0
        self.cycAccelKw[i] = (self.veh.vehKg / (2.0 * (self.cyc.secs[i]))) * \
            ((self.cyc.cycMps[i]**2) - (self.mpsAch[i-1]**2)) / 1000.0
        self.cycAscentKw[i] = self.props.gravityMPerSec2 * np.sin(np.arctan(
            self.cyc.cycGrade[i])) * self.veh.vehKg * ((self.mpsAch[i-1] + self.cyc.cycMps[i]) / 2.0) / 1000.0
        self.cycTracKwReq[i] = self.cycDragKw[i] + \
            self.cycAccelKw[i] + self.cycAscentKw[i]
        self.spareTracKw[i] = self.curMaxTracKw[i] - self.cycTracKwReq[i]
        self.cycRrKw[i] = self.props.gravityMPerSec2 * self.veh.wheelRrCoef * \
            self.veh.vehKg * ((self.mpsAch[i-1] + self.cyc.cycMps[i]) / 2.0) / 1000.0
        self.cycWheelRadPerSec[i] = self.cyc.cycMps[i] / self.veh.wheelRadiusM
        self.cycTireInertiaKw[i] = (((0.5) * self.veh.wheelInertiaKgM2 * (self.veh.numWheels * (self.cycWheelRadPerSec[i]**2.0)) / self.cyc.secs[i]) -
                                    ((0.5) * self.veh.wheelInertiaKgM2 * (self.veh.numWheels * ((self.mpsAch[i-1] / self.veh.wheelRadiusM)**2.0)) / self.cyc.secs[i])) / 1000.0

        self.cycWheelKwReq[i] = self.cycTracKwReq[i] + \
            self.cycRrKw[i] + self.cycTireInertiaKw[i]
        self.regenContrLimKwPerc[i] = self.veh.maxRegen / (1 + self.veh.regenA * np.exp(-self.veh.regenB * (
            (self.cyc.cycMph[i] + self.mpsAch[i-1] * params.mphPerMps) / 2.0 + 1 - 0)))
        self.cycRegenBrakeKw[i] = max(min(
            self.curMaxMechMcKwIn[i] * self.veh.transEff, self.regenContrLimKwPerc[i] * -self.cycWheelKwReq[i]), 0)
        self.cycFricBrakeKw[i] = - \
            min(self.cycRegenBrakeKw[i] + self.cycWheelKwReq[i], 0)
        self.cycTransKwOutReq[i] = self.cycWheelKwReq[i] + \
            self.cycFricBrakeKw[i]

        if self.cycTransKwOutReq[i] <= self.curMaxTransKwOut[i]:
            self.cycMet[i] = 1
            self.transKwOutAch[i] = self.cycTransKwOutReq[i]

        else:
            self.cycMet[i] = -1
            self.transKwOutAch[i] = self.curMaxTransKwOut[i]
        
    def set_ach_speed(self, i):
        """Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
        Arguments
        ------------
        i: index of time step"""

        # Cycle is met
        if self.cycMet[i] == 1:
            self.mpsAch[i] = self.cyc.cycMps[i]

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
                max_iter = 100
                xtol = 1e-18
                # solver gain
                g = 0.8
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

                _ys = [abs(y) for y in ys]
                return xs[_ys.index(min(_ys))]

            Drag3 = 1.0 / 16.0 * self.props.airDensityKgPerM3 * \
                self.veh.dragCoef * self.veh.frontalAreaM2
            Accel2 = 0.5 * self.veh.vehKg / self.cyc.secs[i]
            Drag2 = 3.0 / 16.0 * self.props.airDensityKgPerM3 * \
                self.veh.dragCoef * self.veh.frontalAreaM2 * self.mpsAch[i-1]
            Wheel2 = 0.5 * self.veh.wheelInertiaKgM2 * \
                self.veh.numWheels / (self.cyc.secs[i] * self.veh.wheelRadiusM ** 2)
            Drag1 = 3.0 / 16.0 * self.props.airDensityKgPerM3 * self.veh.dragCoef * \
                self.veh.frontalAreaM2 * self.mpsAch[i-1] ** 2
            Roll1 = 0.5 * self.veh.vehKg * self.props.gravityMPerSec2 * self.veh.wheelRrCoef \
                * np.cos(np.arctan(self.cyc.cycGrade[i])) 
            Ascent1 = 0.5 * self.props.gravityMPerSec2 * \
                np.sin(np.arctan(self.cyc.cycGrade[i])) * self.veh.vehKg 
            Accel0 = -0.5 * self.veh.vehKg * self.mpsAch[i-1] ** 2 / self.cyc.secs[i]
            Drag0 = 1.0 / 16.0 * self.props.airDensityKgPerM3 * self.veh.dragCoef * \
                self.veh.frontalAreaM2 * self.mpsAch[i-1] ** 3
            Roll0 = 0.5 * self.veh.vehKg * self.props.gravityMPerSec2 * \
                self.veh.wheelRrCoef * np.cos(np.arctan(self.cyc.cycGrade[i])) \
                * self.mpsAch[i-1]
            Ascent0 = 0.5 * self.props.gravityMPerSec2 * np.sin(np.arctan(self.cyc.cycGrade[i])) \
                * self.veh.vehKg * self.mpsAch[i-1] 
            Wheel0 = -0.5 * self.veh.wheelInertiaKgM2 * self.veh.numWheels * \
                self.mpsAch[i-1] ** 2 / (self.cyc.secs[i] * self.veh.wheelRadiusM ** 2)

            Total3 = Drag3 / 1e3
            Total2 = (Accel2 + Drag2 + Wheel2) / 1e3
            Total1 = (Drag1 + Roll1 + Ascent1) / 1e3
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / \
                1e3 - self.curMaxTransKwOut[i]

            Total = np.array([Total3, Total2, Total1, Total0])
            self.mpsAch[i] = newton_mps_estimate(Total)

        self.mphAch[i] = self.mpsAch[i] * params.mphPerMps
        self.distMeters[i] = self.mpsAch[i] * self.cyc.secs[i]
        self.distMiles[i] = self.distMeters[i] * (1.0 / params.metersPerMile)
        
    def set_hybrid_cont_calcs(self, i):
        """Hybrid control calculations.  
        Arguments
        ------------
        i: index of time step"""

        if self.transKwOutAch[i] > 0:
            self.transKwInAch[i] = self.transKwOutAch[i] / self.veh.transEff
        else:
            self.transKwInAch[i] = self.transKwOutAch[i] * self.veh.transEff

        if self.cycMet[i] == 1:

            if self.veh.fcEffType == 4:
                self.minMcKw2HelpFc[i] = max(
                    self.transKwInAch[i], -self.curMaxMechMcKwIn[i])

            else:
                self.minMcKw2HelpFc[i] = max(
                    self.transKwInAch[i] - self.curMaxFcKwOut[i], -self.curMaxMechMcKwIn[i])
        else:
            self.minMcKw2HelpFc[i] = max(
                self.curMaxMcKwOut[i], -self.curMaxMechMcKwIn[i])

        if self.veh.noElecSys == True:
            self.regenBufferSoc[i] = 0

        elif self.veh.chargingOn:
            self.regenBufferSoc[i] = max(
                self.veh.maxSoc - (self.veh.maxRegenKwh / self.veh.maxEssKwh), (self.veh.maxSoc + self.veh.minSoc) / 2)

        else:
            self.regenBufferSoc[i] = max(((self.veh.maxEssKwh * self.veh.maxSoc) - (0.5 * self.veh.vehKg * (self.cyc.cycMps[i]**2) * (1.0 / 1000)
                                                                            * (1.0 / 3600) * self.veh.motorPeakEff * self.veh.maxRegen)) / self.veh.maxEssKwh, self.veh.minSoc)

            self.essRegenBufferDischgKw[i] = min(self.curMaxEssKwOut[i], max(
                0, (self.soc[i-1] - self.regenBufferSoc[i]) * self.veh.maxEssKwh * 3600 / self.cyc.secs[i]))

            self.maxEssRegenBufferChgKw[i] = min(max(
                0, (self.regenBufferSoc[i] - self.soc[i-1]) * self.veh.maxEssKwh * 3600.0 / self.cyc.secs[i]), self.curMaxEssChgKw[i])

        if self.veh.noElecSys == True:
            self.accelBufferSoc[i] = 0

        else:
            self.accelBufferSoc[i] = min(max((((((((self.veh.maxAccelBufferMph * (1 / params.mphPerMps))**2)) - ((self.cyc.cycMps[i]**2))) /
                                                (((self.veh.maxAccelBufferMph * (1 / params.mphPerMps))**2))) * (min(self.veh.maxAccelBufferPercOfUseableSoc * \
                                                                            (self.veh.maxSoc - self.veh.minSoc), self.veh.maxRegenKwh / self.veh.maxEssKwh) * self.veh.maxEssKwh)) / self.veh.maxEssKwh) + \
                self.veh.minSoc, self.veh.minSoc), self.veh.maxSoc)

            self.essAccelBufferChgKw[i] = max(
                0, (self.accelBufferSoc[i] - self.soc[i-1]) * self.veh.maxEssKwh * 3600.0 / self.cyc.secs[i])
            self.maxEssAccelBufferDischgKw[i] = min(max(
                0, (self.soc[i-1] - self.accelBufferSoc[i]) * self.veh.maxEssKwh * 3600 / self.cyc.secs[i]), self.curMaxEssKwOut[i])

        if self.regenBufferSoc[i] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(min(((self.soc[i-1] - (self.regenBufferSoc[i] + self.accelBufferSoc[i]) / 2) * self.veh.maxEssKwh * 3600.0) /
                                                    self.cyc.secs[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        elif self.soc[i-1] > self.regenBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(min(
                self.essRegenBufferDischgKw[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        elif self.soc[i-1] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = max(
                min(-1.0 * self.essAccelBufferChgKw[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        else:
            self.essAccelRegenDischgKw[i] = max(
                min(0, self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

        self.fcKwGapFrEff[i] = abs(self.transKwOutAch[i] - self.veh.maxFcEffKw)

        if self.veh.noElecSys == True:
            self.mcElectInKwForMaxFcEff[i] = 0

        elif self.transKwOutAch[i] < self.veh.maxFcEffKw:

            if self.fcKwGapFrEff[i] == self.veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / \
                    self.veh.mcFullEffArray[-1] * -1
            else:
                self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / self.veh.mcFullEffArray[max(
                    1, np.argmax(self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1)] * -1

        else:

            if self.fcKwGapFrEff[i] == self.veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = self.veh.mcKwInArray[len(
                    self.veh.mcKwInArray) - 1]
            else:
                self.mcElectInKwForMaxFcEff[i] = self.veh.mcKwInArray[np.argmax(
                    self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1]

        if self.veh.noElecSys == True:
            self.electKwReq4AE[i] = 0

        elif self.transKwInAch[i] > 0:
            if self.transKwInAch[i] == self.veh.maxMotorKw:

                self.electKwReq4AE[i] = self.transKwInAch[i] / \
                    self.veh.mcFullEffArray[-1] + self.auxInKw[i]
            else:
                self.electKwReq4AE[i] = self.transKwInAch[i] / self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.transKwInAch[i])) - 1)] + self.auxInKw[i]

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
                and ((self.cyc.cycMph[i] - 1e-6) <= self.veh.mphFcOn or self.veh.chargingOn) and \
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
            self.essAEKwOut[i] = max(-self.curMaxEssChgKw[i], -self.maxEssRegenBufferChgKw[i], min(0, self.curMaxRoadwayChgKw[i] - (
                self.transKwInAch[i] + self.auxInKw[i])), min(self.curMaxEssKwOut[i], self.desiredEssKwOutForAE[i]))

        else:
            self.essAEKwOut[i] = 0

        self.erAEKwOut[i] = min(max(0, self.transKwInAch[i] + self.auxInKw[i] - self.essAEKwOut[i]), self.curMaxRoadwayChgKw[i])
    
    def set_fc_forced_state(self, i):
        """Calculate control variables related to engine on/off state
        Arguments       
        ------------
        i: index of time step"""
        # force fuel converter on if it was on in the previous time step, but only if fc
        # has not been on longer than minFcTimeOn
        if self.prevfcTimeOn[i] > 0 and self.prevfcTimeOn[i] < self.veh.minFcTimeOn - self.cyc.secs[i]:
            self.fcForcedOn[i] = True
        else:
            self.fcForcedOn[i] = False

        if self.fcForcedOn[i] == False or self.canPowerAllElectrically[i] == False:
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
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], self.essAccelRegenDischgKw[i]))

        elif self.essRegenBufferDischgKw[i] > 0:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], min(self.essAccelRegenDischgKw[i], self.mcElecInLimKw[i] + self.auxInKw[i], max(self.essRegenBufferDischgKw[i], self.essDesiredKw4FcEff[i]))))

        elif self.essAccelBufferChgKw[i] > 0:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], max(-1 * self.maxEssRegenBufferChgKw[i], min(-self.essAccelBufferChgKw[i], self.essDesiredKw4FcEff[i]))))

        elif self.essDesiredKw4FcEff[i] > 0:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], min(self.essDesiredKw4FcEff[i], self.maxEssAccelBufferDischgKw[i])))

        else:
            self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], self.veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            max(-self.curMaxEssChgKw[i], max(self.essDesiredKw4FcEff[i], -self.maxEssRegenBufferChgKw[i])))

        self.erKwIfFcIsReq[i] = max(0, min(self.curMaxRoadwayChgKw[i], self.curMaxMechMcKwIn[i],
                                    self.essKwIfFcIsReq[i] - self.mcElecInLimKw[i] + self.auxInKw[i]))

        self.mcElecKwInIfFcIsReq[i] = self.essKwIfFcIsReq[i] + self.erKwIfFcIsReq[i] - self.auxInKw[i]

        if self.veh.noElecSys == True:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] == 0:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] > 0:

            if self.mcElecKwInIfFcIsReq[i] == max(self.veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * \
                    self.veh.mcFullEffArray[-1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i])) - 1)]

        else:
            if self.mcElecKwInIfFcIsReq[i] * -1 == max(self.veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / \
                    self.veh.mcFullEffArray[-1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / (self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i] * -1)) - 1)])

        if self.veh.maxMotorKw == 0:
            self.mcMechKwOutAch[i] = 0

        elif self.fcForcedOn[i] == True and self.canPowerAllElectrically[i] == True and (self.veh.vehPtType == 2.0 or self.veh.vehPtType == 3.0) and self.veh.fcEffType !=4:
            self.mcMechKwOutAch[i] = self.mcMechKw4ForcedFc[i]

        elif self.transKwInAch[i] <= 0:

            if self.veh.fcEffType !=4 and self.veh.maxFuelConvKw > 0:
                if self.canPowerAllElectrically[i] == 1:
                    self.mcMechKwOutAch[i] = - \
                        min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i])
                else:
                    self.mcMechKwOutAch[i] = min(-min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]),
                                                    max(-self.curMaxFcKwOut[i], self.mcKwIfFcIsReq[i]))
            else:
                self.mcMechKwOutAch[i] = min(
                    -min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]), -self.transKwInAch[i])

        elif self.canPowerAllElectrically[i] == 1:
            self.mcMechKwOutAch[i] = self.transKwInAch[i]

        else:
            self.mcMechKwOutAch[i] = max(
                self.minMcKw2HelpFc[i], self.mcKwIfFcIsReq[i])

        if self.mcMechKwOutAch[i] == 0:
            self.mcElecKwInAch[i] = 0.0
            self.motor_index_debug[i] = 0

        elif self.mcMechKwOutAch[i] < 0:

            if self.mcMechKwOutAch[i] * -1 == max(self.veh.mcKwInArray):
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * \
                    self.veh.mcFullEffArray[-1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwInArray > min(max(self.veh.mcKwInArray) - 0.01, self.mcMechKwOutAch[i] * -1)) - 1)]

        else:
            if self.veh.maxMotorKw == self.mcMechKwOutAch[i]:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / \
                    self.veh.mcFullEffArray[-1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / self.veh.mcFullEffArray[max(1, np.argmax(
                    self.veh.mcKwOutArray > min(self.veh.maxMotorKw - 0.01, self.mcMechKwOutAch[i])) - 1)]

        if self.curMaxRoadwayChgKw[i] == 0:
            self.roadwayChgKwOutAch[i] = 0

        elif self.veh.fcEffType == 4:
            self.roadwayChgKwOutAch[i] = max(
                0, self.mcElecKwInAch[i], self.maxEssRegenBufferChgKw[i], self.essRegenBufferDischgKw[i], self.curMaxRoadwayChgKw[i])

        elif self.canPowerAllElectrically[i] == 1:
            self.roadwayChgKwOutAch[i] = self.erAEKwOut[i]

        else:
            self.roadwayChgKwOutAch[i] = self.erKwIfFcIsReq[i]

        self.minEssKw2HelpFc[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - \
            self.curMaxFcKwOut[i] - self.roadwayChgKwOutAch[i]

        if self.veh.maxEssKw == 0 or self.veh.maxEssKwh == 0:
            self.essKwOutAch[i] = 0

        elif self.veh.fcEffType == 4:

            if self.transKwOutAch[i] >=0:
                self.essKwOutAch[i] = min(max(self.minEssKw2HelpFc[i], self.essDesiredKw4FcEff[i], self.essAccelRegenDischgKw[i]),
                                            self.curMaxEssKwOut[i], self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i])

            else:
                self.essKwOutAch[i] = self.mcElecKwInAch[i] + \
                    self.auxInKw[i] - self.roadwayChgKwOutAch[i]

        elif self.highAccFcOnTag[i] == 1 or self.veh.noElecAux == True:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] - \
                self.roadwayChgKwOutAch[i]

        else:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] + \
                self.auxInKw[i] - self.roadwayChgKwOutAch[i]

        if self.veh.noElecSys == True:
            self.essCurKwh[i] = 0

        elif self.essKwOutAch[i] < 0:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * \
                (self.cyc.secs[i] / 3600.0) * np.sqrt(self.veh.essRoundTripEff)

        else:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * \
                (self.cyc.secs[i] / 3600.0) * (1 / np.sqrt(self.veh.essRoundTripEff))

        if self.veh.maxEssKwh == 0:
            self.soc[i] = 0.0

        else:
            self.soc[i] = self.essCurKwh[i] / self.veh.maxEssKwh

        if self.canPowerAllElectrically[i] == True and self.fcForcedOn[i] == False and self.fcKwOutAch[i] == 0:
            self.fcTimeOn[i] = 0
        else:
            self.fcTimeOn[i] = self.fcTimeOn[i-1] + self.cyc.secs[i]
    
    def set_fc_power(self, i):
        """Sets fcKwOutAch and fcKwInAch.
        Arguments
        ------------
        i: index of time step"""

        if self.veh.maxFuelConvKw == 0:
            self.fcKwOutAch[i] = 0

        elif self.veh.fcEffType == 4:
            self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(
                0, self.mcElecKwInAch[i] + self.auxInKw[i] - self.essKwOutAch[i] - self.roadwayChgKwOutAch[i]))

        elif self.veh.noElecSys == True or self.veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
            self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(
                0, self.transKwInAch[i] - self.mcMechKwOutAch[i] + self.auxInKw[i]))

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
            self.fcKwInAch[i] = self.fcKwOutAch[i] / (self.veh.fcEffArray[np.argmax(
                self.veh.fcKwOutArray > min(self.fcKwOutAch[i], self.veh.fcMaxOutkW)) - 1])

        self.fsKwOutAch[i] = self.fcKwInAch[i]

        self.fsKwhOutAch[i] = self.fsKwOutAch[i] * \
            self.cyc.secs[i] * (1 / 3600.0)

    def set_post_scalars(self):
        """Sets scalar variables that can be calculated after a cycle is run. 
        This includes mpgge, various energy metrics, and others"""

        self.fsCumuMjOutAch = (self.fsKwOutAch * self.cyc.secs).cumsum() * 1e-3

        if self.fsKwhOutAch.sum() == 0:
            self.mpgge = 0

        else:
            self.mpgge = self.distMiles.sum() / \
                (self.fsKwhOutAch.sum() * (1 / params.kWhPerGGE))

        self.roadwayChgKj = (self.roadwayChgKwOutAch * self.cyc.secs).sum()
        self.essDischgKj = - \
            (self.soc[-1] - self.soc[0]) * self.veh.maxEssKwh * 3600.0
        self.battery_kWh_per_mi  = (
            self.essDischgKj / 3600.0) / self.distMiles.sum()
        self.electric_kWh_per_mi  = (
            (self.roadwayChgKj + self.essDischgKj) / 3600.0) / self.distMiles.sum()
        self.fuelKj = (self.fsKwOutAch * self.cyc.secs).sum()

        if (self.fuelKj + self.roadwayChgKj) == 0:
            self.ess2fuelKwh  = 1.0

        else:
            self.ess2fuelKwh  = self.essDischgKj / (self.fuelKj + self.roadwayChgKj)

        if self.mpgge == 0:
            # hardcoded conversion
            self.Gallons_gas_equivalent_per_mile = self.electric_kWh_per_mi / params.kWhPerGGE
            grid_Gallons_gas_equivalent_per_mile = self.electric_kWh_per_mi / 33.7 / self.veh.chgEff

        else:
            self.Gallons_gas_equivalent_per_mile = 1 / \
                self.mpgge + self.electric_kWh_per_mi  / params.kWhPerGGE
            grid_Gallons_gas_equivalent_per_mile = 1 / self.mpgge + self.electric_kWh_per_mi / 33.7 / self.veh.chgEff

        self.grid_mpgge_elec = 1 / grid_Gallons_gas_equivalent_per_mile
        self.mpgge_elec = 1 / self.Gallons_gas_equivalent_per_mile

        # energy audit calcs
        self.dragKw = self.cycDragKw
        self.dragKj = (self.dragKw * self.cyc.secs).sum()
        self.ascentKw = self.cycAscentKw
        self.ascentKj = (self.ascentKw * self.cyc.secs).sum()
        self.rrKw = self.cycRrKw
        self.rrKj = (self.rrKw * self.cyc.secs).sum()

        self.essLossKw[1:] = np.array(
            [0 if (self.veh.maxEssKw == 0 or self.veh.maxEssKwh == 0)
            else -self.essKwOutAch[i] - (-self.essKwOutAch[i] * np.sqrt(self.veh.essRoundTripEff))
                if self.essKwOutAch[i] < 0
            else self.essKwOutAch[i] * (1.0 / np.sqrt(self.veh.essRoundTripEff)) - self.essKwOutAch[i]
            for i in range(1, len(self.cyc.cycSecs))])
        
        self.brakeKj = (self.cycFricBrakeKw * self.cyc.secs).sum()
        self.transKj = ((self.transKwInAch - self.transKwOutAch) * self.cyc.secs).sum()
        self.mcKj = ((self.mcElecKwInAch - self.mcMechKwOutAch) * self.cyc.secs).sum()
        self.essEffKj = (self.essLossKw * self.cyc.secs).sum()
        self.auxKj = (self.auxInKw * self.cyc.secs).sum()
        self.fcKj = ((self.fcKwInAch - self.fcKwOutAch) * self.cyc.secs).sum()
        
        self.netKj = self.dragKj + self.ascentKj + self.rrKj + self.brakeKj + self.transKj \
            + self.mcKj + self.essEffKj + self.auxKj + self.fcKj

        self.keKj = 0.5 * self.veh.vehKg * \
            (self.mpsAch[0]**2 - self.mpsAch[-1]**2) / 1000
        
        self.energyAuditError = ((self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj) - self.netKj) /\
            (self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj)

        if np.abs(self.energyAuditError) > params.ENERGY_AUDIT_ERROR_TOLERANCE:
            print('Warning: There is a problem with conservation of energy.')

        self.accelKw[1:] = (self.veh.vehKg / (2.0 * (self.cyc.secs[1:]))) * \
            ((self.mpsAch[1:]**2) - (self.mpsAch[:-1]**2)) / 1000.0 

sim_drive_spec = build_spec(SimDriveClassic(cycle.Cycle('udds'), vehicle.Vehicle(1)))


@jitclass(sim_drive_spec)
class SimDriveJit(SimDriveClassic):
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle fuel economy simulations. This class will be 
    faster for large batch runs.
    Arguments:
    ----------
    cyc: cycle.CycleJit instance. Can come from cycle.Cycle.get_numba_cyc
    veh: vehicle.VehicleJit instance. Can come from vehicle.Vehicle.get_numba_veh"""

    def __init_objects__(self, cyc, veh):        
        self.veh = veh
        self.cyc = cyc.copy() # this cycle may be manipulated
        self.cyc0 = cyc.copy() # this cycle is not to be manipulated
        self.sim_params = SimDriveParams()
        self.props = params.PhysicalPropertiesJit()

    def sim_drive(self, initSoc=-1, auxInKwOverride=np.zeros(1, dtype=np.float64)):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        initSoc: initial SOC for electrified vehicles.  
            Leave empty for default value.  Otherwise, must be between 0 and 1.
            Numba's jitclass does not support keyword args so this allows for optionally
            passing initSoc as positional argument.
            auxInKw: auxInKw override.  Array of same length as cyc.cycSecs.  
                Default of np.zeros(1) causes veh.auxKw to be used. If zero is actually
                desired as an override, either set veh.auxKw = 0 before instantiaton of
                SimDrive*, or use `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting
                the final value to non-zero prevents override mechanism.  
        """

        if (auxInKwOverride == 0).all():
            auxInKwOverride = self.auxInKw

        self.hev_sim_count = 0

        if (initSoc != -1):
            if (initSoc > 1.0 or initSoc < 0.0):
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = -1 # override initSoc if invalid value is used
            else:
                self.sim_drive_walk(initSoc, auxInKwOverride)
    
        elif self.veh.vehPtType == 1: # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0 # this initSoc has no impact on results
            
            self.sim_drive_walk(initSoc, auxInKwOverride)

        elif self.veh.vehPtType == 2 and initSoc == -1:  # HEV 

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
                fuelKj = np.sum(self.fsKwOutAch * self.cyc.secs)
                roadwayChgKj = np.sum(self.roadwayChgKwOutAch * self.cyc.secs)
                if (fuelKj + roadwayChgKj) > 0:
                    ess2fuelKwh = np.abs((self.soc[0] - self.soc[-1]) *
                                         self.veh.maxEssKwh * 3600 / (fuelKj + roadwayChgKj))
                else:
                    ess2fuelKwh = 0.0
                initSoc = min(1.0, max(0.0, self.soc[-1]))
                        
            self.sim_drive_walk(initSoc, auxInKwOverride)

        elif (self.veh.vehPtType == 3 and initSoc == -1) or (self.veh.vehPtType == 4 and initSoc == -1): # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = self.veh.maxSoc
            
            self.sim_drive_walk(initSoc, auxInKwOverride)

        else:
            
            self.sim_drive_walk(initSoc, auxInKwOverride)
        
        self.set_post_scalars()            
            
@jitclass(sim_drive_spec)
class SimAccelTestJit(SimDriveClassic):
    """Class compiled using numba just-in-time compilation containing methods 
    for running FASTSim vehicle acceleration simulation. This class will be 
    faster for large batch runs."""

    def sim_drive(self):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType."""

        if self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_walk(initSoc)

        elif self.veh.vehPtType == 2:  # HEV

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_walk(initSoc)

        else:

            # If EV, initializing initial SOC to maximum SOC.
            initSoc = self.veh.maxSoc
            self.sim_drive_walk(initSoc)

        self.set_post_scalars()


class SimAccelTest(SimDriveClassic):
    """Class for running FASTSim vehicle acceleration simulation."""

    def sim_drive(self):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType."""

        if self.veh.vehPtType == 1:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = (self.veh.maxSoc + self.veh.minSoc) / 2.0
            self.sim_drive_walk(initSoc)

        elif self.veh.vehPtType == 2:  # HEV

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

        for item in sim_drive_spec:
            self.__setattr__(item[0], sim_drive.__getattribute__(item[0]))

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
        output['maxTraceMissMph'] = params.mphPerMps * \
            max(abs(self.cyc.cycMps - self.mpsAch))
        self.maxTraceMissMph = output['maxTraceMissMph']

        output['ess2fuelKwh'] = self.ess2fuelKwh

        output['initial_soc'] = self.soc[0]
        output['final_soc'] = self.soc[-1]

        output['mpgge_elec'] = self.mpgge_elec
        output['soc'] = self.soc
        output['distance_mi'] = sum(self.distMiles)
        duration_sec = self.cyc.cycSecs[-1] - self.cyc.cycSecs[0]
        output['avg_speed_mph'] = sum(
            self.distMiles) / (duration_sec / 3600.0)
        self.avg_speed_mph = output['avg_speed_mph']
        self.accel = np.diff(self.mphAch) / np.diff(self.cyc.cycSecs)
        output['avg_accel_mphps'] = np.mean(self.accel[self.accel > 0])
        self.avg_accel_mphps = output['avg_accel_mphps']

        if max(self.mphAch) > 60:
            output['ZeroToSixtyTime_secs'] = np.interp(60, self.mphAch, self.cyc.cycSecs)

        else:
            output['ZeroToSixtyTime_secs'] = 0.0

        #######################################################################
        ####  Time series information for additional analysis / debugging. ####
        ####             Add parameters of interest as needed.             ####
        #######################################################################

        output['fcKwOutAch'] = np.asarray(self.fcKwOutAch)
        output['fsKwhOutAch'] = np.asarray(self.fsKwhOutAch)
        output['fcKwInAch'] = np.asarray(self.fcKwInAch)
        output['time'] = np.asarray(self.cyc.cycSecs)

        return output

    # optional post-processing methods
    def get_diagnostics(self):
        """This method is to be run after runing sim_drive, if diagnostic variables 
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
            output[search[1] + 'Kj' + search[2] + 'Pos'] = np.trapz(tempvars[var + 'Pos'], self.cyc.cycSecs)
            output[search[1] + 'Kj' + search[2] + 'Neg'] = np.trapz(tempvars[var + 'Neg'], self.cyc.cycSecs)
        
        output['distMilesFinal'] = sum(self.distMiles)
        output['mpgge'] = sum(self.distMiles) / sum(self.fsKwhOutAch) * params.kWhPerGGE
    
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

