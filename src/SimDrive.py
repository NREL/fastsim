"""Module containing class and methods for simulating vehicle drive cycle.
For example usage, see ../README.md"""

### Import necessary python modules
import numpy as np
import pandas as pd
import re
from Globals import *
from numba import jitclass                 # import the decorator
from numba import float32, int32, bool_    # import the types
from numba import types, typed, deferred_type
import numba
import warnings
warnings.simplefilter('ignore')

import LoadData
import importlib
importlib.reload(LoadData)

# list of array attributes in SimDrive class
attr_list = ['curMaxFsKwOut', 'fcTransLimKw', 'fcFsLimKw', 'fcMaxKwIn', 'curMaxFcKwOut', 'essCapLimDischgKw', 'curMaxEssKwOut', 
            'curMaxAvailElecKw', 'essCapLimChgKw', 'curMaxEssChgKw', 'curMaxElecKw', 'mcElecInLimKw', 'mcTransiLimKw', 'curMaxMcKwOut', 
            'essLimMcRegenPercKw', 'essLimMcRegenKw', 'curMaxMechMcKwIn', 'curMaxTransKwOut', 'cycDragKw', 'cycAccelKw', 'cycAscentKw', 
            'cycTracKwReq', 'curMaxTracKw', 'spareTracKw', 'cycRrKw', 'cycWheelRadPerSec', 'cycTireInertiaKw', 'cycWheelKwReq', 
            'regenContrLimKwPerc', 'cycRegenBrakeKw', 'cycFricBrakeKw', 'cycTransKwOutReq', 'cycMet', 'transKwOutAch', 'transKwInAch', 
            'curSocTarget', 'minMcKw2HelpFc', 'mcMechKwOutAch', 'mcElecKwInAch', 'auxInKw', 'roadwayChgKwOutAch', 'minEssKw2HelpFc', 
            'essKwOutAch', 'fcKwOutAch', 'fcKwOutAch_pct', 'fcKwInAch', 'fsKwOutAch', 'fsKwhOutAch', 'essCurKwh', 'soc', 
            'regenBufferSoc', 'essRegenBufferDischgKw', 'maxEssRegenBufferChgKw', 'essAccelBufferChgKw', 'accelBufferSoc', 
            'maxEssAccelBufferDischgKw', 'essAccelRegenDischgKw', 'mcElectInKwForMaxFcEff', 'electKwReq4AE', 'canPowerAllElectrically', 
            'desiredEssKwOutForAE', 'essAEKwOut', 'erAEKwOut', 'essDesiredKw4FcEff', 'essKwIfFcIsReq', 'curMaxMcElecKwIn', 'fcKwGapFrEff', 
            'erKwIfFcIsReq', 'mcElecKwInIfFcIsReq', 'mcKwIfFcIsReq', 'mcMechKw4ForcedFc', 'fcTimeOn', 
            'prevfcTimeOn', 'mpsAch', 'mphAch', 'distMeters', 'distMiles', 'highAccFcOnTag', 'reachedBuff', 'maxTracMps', 'addKwh', 
            'dodCycs', 'essPercDeadArray', 'dragKw', 'essLossKw', 'accelKw', 'ascentKw', 'rrKw', 'motor_index_debug', 'debug_flag', 
             'curMaxRoadwayChgKw']

spec = [(attr, float32[:]) for attr in attr_list]
spec.append(('fcForcedOn', bool_[:]))
spec.append(('fcForcedState', int32[:]))
# spec.append(('len_cyc', int32))
# cyc_type = deferred_type()
# cyc_type.define(LoadData.TypedCycle.class_type.instance_type)
# spec.append(('cyc', cyc_type))
# veh_type = deferred_type()
# veh_type.define(LoadData.TypedVehicle.class_type.instance_type)
# spec.append(('veh', veh_type))

@jitclass(spec)
class SimDrive(object):
    """Class containing methods for running FASTSim vehicle fuel economy simulations."""
    def __init__(self, len_cyc):
        """Initializes arrays for specific cycle
        Arguments:
        -----------
        len_cyc: instance of LoadData.Cycle class
        """
        # Component Limits -- calculated dynamically"
        self.curMaxFsKwOut = np.zeros(len_cyc, dtype=np.float32)
        self.fcTransLimKw = np.zeros(len_cyc, dtype=np.float32)
        self.fcFsLimKw = np.zeros(len_cyc, dtype=np.float32)
        self.fcMaxKwIn = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxFcKwOut = np.zeros(len_cyc, dtype=np.float32)
        self.essCapLimDischgKw = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxEssKwOut = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxAvailElecKw = np.zeros(len_cyc, dtype=np.float32)
        self.essCapLimChgKw = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxEssChgKw = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxElecKw = np.zeros(len_cyc, dtype=np.float32)
        self.mcElecInLimKw = np.zeros(len_cyc, dtype=np.float32)
        self.mcTransiLimKw = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxMcKwOut = np.zeros(len_cyc, dtype=np.float32)
        self.essLimMcRegenPercKw = np.zeros(len_cyc, dtype=np.float32)
        self.essLimMcRegenKw = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxMechMcKwIn = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxTransKwOut = np.zeros(len_cyc, dtype=np.float32)

        ### Drive Train
        self.cycDragKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycAccelKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycAscentKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycTracKwReq = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxTracKw = np.zeros(len_cyc, dtype=np.float32)
        self.spareTracKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycRrKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycWheelRadPerSec = np.zeros(len_cyc, dtype=np.float32)
        self.cycTireInertiaKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycWheelKwReq = np.zeros(len_cyc, dtype=np.float32)
        self.regenContrLimKwPerc = np.zeros(len_cyc, dtype=np.float32)
        self.cycRegenBrakeKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycFricBrakeKw = np.zeros(len_cyc, dtype=np.float32)
        self.cycTransKwOutReq = np.zeros(len_cyc, dtype=np.float32)
        self.cycMet = np.zeros(len_cyc, dtype=np.float32)
        self.transKwOutAch = np.zeros(len_cyc, dtype=np.float32)
        self.transKwInAch = np.zeros(len_cyc, dtype=np.float32)
        self.curSocTarget = np.zeros(len_cyc, dtype=np.float32)
        self.minMcKw2HelpFc = np.zeros(len_cyc, dtype=np.float32)
        self.mcMechKwOutAch = np.zeros(len_cyc, dtype=np.float32)
        self.mcElecKwInAch = np.zeros(len_cyc, dtype=np.float32)
        self.auxInKw = np.zeros(len_cyc, dtype=np.float32)
        self.roadwayChgKwOutAch = np.zeros(len_cyc, dtype=np.float32)
        self.minEssKw2HelpFc = np.zeros(len_cyc, dtype=np.float32)
        self.essKwOutAch = np.zeros(len_cyc, dtype=np.float32)
        self.fcKwOutAch = np.zeros(len_cyc, dtype=np.float32)
        self.fcKwOutAch_pct = np.zeros(len_cyc, dtype=np.float32)
        self.fcKwInAch = np.zeros(len_cyc, dtype=np.float32)
        self.fsKwOutAch = np.zeros(len_cyc, dtype=np.float32)
        self.fsKwhOutAch = np.zeros(len_cyc, dtype=np.float32)
        self.essCurKwh = np.zeros(len_cyc, dtype=np.float32)
        self.soc = np.zeros(len_cyc, dtype=np.float32)

        # Vehicle Attributes, Control Variables
        self.regenBufferSoc = np.zeros(len_cyc, dtype=np.float32)
        self.essRegenBufferDischgKw = np.zeros(len_cyc, dtype=np.float32)
        self.maxEssRegenBufferChgKw = np.zeros(len_cyc, dtype=np.float32)
        self.essAccelBufferChgKw = np.zeros(len_cyc, dtype=np.float32)
        self.accelBufferSoc = np.zeros(len_cyc, dtype=np.float32)
        self.maxEssAccelBufferDischgKw = np.zeros(len_cyc, dtype=np.float32)
        self.essAccelRegenDischgKw = np.zeros(len_cyc, dtype=np.float32)
        self.mcElectInKwForMaxFcEff = np.zeros(len_cyc, dtype=np.float32)
        self.electKwReq4AE = np.zeros(len_cyc, dtype=np.float32)
        self.canPowerAllElectrically = np.zeros(len_cyc, dtype=np.float32)
        self.desiredEssKwOutForAE = np.zeros(len_cyc, dtype=np.float32)
        self.essAEKwOut = np.zeros(len_cyc, dtype=np.float32)
        self.erAEKwOut = np.zeros(len_cyc, dtype=np.float32)
        self.essDesiredKw4FcEff = np.zeros(len_cyc, dtype=np.float32)
        self.essKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxMcElecKwIn = np.zeros(len_cyc, dtype=np.float32)
        self.fcKwGapFrEff = np.zeros(len_cyc, dtype=np.float32)
        self.erKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float32)
        self.mcElecKwInIfFcIsReq = np.zeros(len_cyc, dtype=np.float32)
        self.mcKwIfFcIsReq = np.zeros(len_cyc, dtype=np.float32)
        self.fcForcedOn = np.array([False] * len_cyc, dtype=np.bool_)
        self.fcForcedState = np.zeros(len_cyc, dtype=np.int32)
        self.mcMechKw4ForcedFc = np.zeros(len_cyc, dtype=np.float32)
        self.fcTimeOn = np.zeros(len_cyc, dtype=np.float32)
        self.prevfcTimeOn = np.zeros(len_cyc, dtype=np.float32)
        
        ### Additional Variables
        self.mpsAch = np.zeros(len_cyc, dtype=np.float32)
        self.mphAch = np.zeros(len_cyc, dtype=np.float32)
        self.distMeters = np.zeros(len_cyc, dtype=np.float32)
        self.distMiles = np.zeros(len_cyc, dtype=np.float32)
        self.highAccFcOnTag = np.zeros(len_cyc, dtype=np.float32)
        self.reachedBuff = np.zeros(len_cyc, dtype=np.float32)
        self.maxTracMps = np.zeros(len_cyc, dtype=np.float32)
        self.addKwh = np.zeros(len_cyc, dtype=np.float32)
        self.dodCycs = np.zeros(len_cyc, dtype=np.float32)
        self.essPercDeadArray = np.zeros(len_cyc, dtype=np.float32)
        self.dragKw = np.zeros(len_cyc, dtype=np.float32)
        self.essLossKw = np.zeros(len_cyc, dtype=np.float32)
        self.accelKw = np.zeros(len_cyc, dtype=np.float32)
        self.ascentKw = np.zeros(len_cyc, dtype=np.float32)
        self.rrKw = np.zeros(len_cyc, dtype=np.float32)
        self.motor_index_debug = np.zeros(len_cyc, dtype=np.float32)
        self.debug_flag = np.zeros(len_cyc, dtype=np.float32)
        self.curMaxRoadwayChgKw = np.zeros(len_cyc, dtype=np.float32)
    
    def sim_drive(self, cyc, veh, initSoc=np.nan):
        """Initialize and run sim_drive_sub as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc(optional): initial SOC for electrified vehicles"""

        if initSoc != np.nan:
            if initSoc > 1.0 or initSoc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = np.nan
    
        if veh.vehPtType == 1: # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = 0.0
            
            self.sim_drive_sub(cyc, veh, initSoc)

        elif veh.vehPtType == 2 and initSoc == np.nan:  # HEV 

            #####################################
            ### Charge Balancing Vehicle SOC ###
            #####################################

            # Charge balancing SOC for PHEV vehicle types. Iterating initsoc and comparing to final SOC.
            # Iterating until tolerance met or 30 attempts made.

            initSoc = (veh.maxSoc + veh.minSoc) / 2.0
            ess2fuelKwh = 1.0
            sim_count = 0
            while ess2fuelKwh > veh.essToFuelOkError and sim_count < 30:
                sim_count += 1
                self.sim_drive_sub(cyc, veh, initSoc)
                output = self.get_output(cyc, veh)
                ess2fuelKwh = abs(output['ess2fuelKwh'])
                initSoc = np.min([1.0, np.max([0.0, output['final_soc']])])
                        
            self.sim_drive_sub(cyc, veh, initSoc)

        elif (veh.vehPtType == 3 and initSoc == np.nan) or (veh.vehPtType == 4 and initSoc == np.nan): # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = np.copy(veh.maxSoc)
            
            self.sim_drive_sub(cyc, veh, initSoc)

        else:
            
            self.sim_drive_sub(cyc, veh, initSoc)

    def sim_drive_sub(self, cyc, veh, initSoc):
        """Receives second-by-second cycle information, vehicle properties, 
        and an initial state of charge and performs a backward facing 
        powertrain simulation. Method 'sim_drive' runs this to 
        iterate through the time steps of 'cyc'.

        Arguments
        ------------
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial battery state-of-charge (SOC) for electrified vehicles"""
        
        ############################
        ###   Loop Through Time  ###
        ############################

        ###  Assign First ValueS  ###
        ### Drive Train
        self.cycMet[0] = 1
        self.curSocTarget[0] = veh.maxSoc
        self.essCurKwh[0] = initSoc * veh.maxEssKwh
        self.soc[0] = initSoc


        for i in range(1, len(cyc.cycSecs)):
            ### Misc calcs
            # If noElecAux, then the HV electrical system is not used to power aux loads 
            # and it must all come from the alternator.  This apparently assumes no belt-driven aux 
            # loads
            # *** 

            self.set_misc_calcs(i, cyc, veh)
            self.set_comp_lims(i, cyc, veh)
            self.set_power_calcs(i, cyc, veh)
            self.set_speed_dist_calcs(i, cyc, veh)
            self.set_hybrid_cont_calcs(i, cyc, veh)
            self.set_fc_forced_state(i, cyc, veh) # can probably be *mostly* done with list comprehension in post processing
            self.set_hybrid_cont_decisions(i, cyc, veh)

    def set_misc_calcs(self, i, cyc, veh):
        """Sets misc. calculations at time step 'i'
        Arguments
        ------------
        i: index of time step
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        if veh.noElecAux == True:
            self.auxInKw[i] = veh.auxKw / veh.altEff
        else:
            self.auxInKw[i] = veh.auxKw

        # Is SOC below min threshold?
        if self.soc[i-1] < (veh.minSoc + veh.percHighAccBuf):
            self.reachedBuff[i] = 0
        else:
            self.reachedBuff[i] = 1

        # Does the engine need to be on for low SOC or high acceleration
        if self.soc[i-1] < veh.minSoc or (self.highAccFcOnTag[i-1] == 1 and self.reachedBuff[i] == 0):
            self.highAccFcOnTag[i] = 1
        else:
            self.highAccFcOnTag[i] = 0
        self.maxTracMps[i] = self.mpsAch[i-1] + (veh.maxTracMps2 * cyc.secs[i])

    def set_comp_lims(self, i, cyc, veh):
        """Sets component limits for time step 'i'
        Arguments
        ------------
        i: index of time step
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        # max fuel storage power output
        self.curMaxFsKwOut[i] = np.min([veh.maxFuelStorKw, self.fsKwOutAch[i-1] + (
            (veh.maxFuelStorKw / veh.fuelStorSecsToPeakPwr) * (cyc.secs[i]))])
        # maximum fuel storage power output rate of change
        self.fcTransLimKw[i] = self.fcKwOutAch[i-1] + \
            ((veh.maxFuelConvKw / veh.fuelConvSecsToPeakPwr) * (cyc.secs[i]))

        self.fcMaxKwIn[i] = np.min([self.curMaxFsKwOut[i], veh.maxFuelStorKw])
        self.fcFsLimKw[i] = veh.fcMaxOutkW
        self.curMaxFcKwOut[i] = np.min([
            veh.maxFuelConvKw, self.fcFsLimKw[i], self.fcTransLimKw[i]])

        # *** I think veh.maxEssKw should also be in the following
        # boolean condition
        if veh.maxEssKwh == 0 or self.soc[i-1] < veh.minSoc:
            self.essCapLimDischgKw[i] = 0.0

        else:
            self.essCapLimDischgKw[i] = (
                veh.maxEssKwh * np.sqrt(veh.essRoundTripEff)) * 3600.0 * (self.soc[i-1] - veh.minSoc) / (cyc.secs[i])
        self.curMaxEssKwOut[i] = np.min([
            veh.maxEssKw, self.essCapLimDischgKw[i]])

        if veh.maxEssKwh == 0 or veh.maxEssKw == 0:
            self.essCapLimChgKw[i] = 0

        else:
            self.essCapLimChgKw[i] = np.max([((veh.maxSoc - self.soc[i-1]) * veh.maxEssKwh * (1 /
                                        np.sqrt(veh.essRoundTripEff))) / ((cyc.secs[i]) * (1 / 3600.0)), 0])

        self.curMaxEssChgKw[i] = np.min([self.essCapLimChgKw[i], veh.maxEssKw])

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if veh.fcEffType == 4:
            self.curMaxElecKw[i] = self.curMaxFcKwOut[i] + self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        else:
            self.curMaxElecKw[i] = self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        self.curMaxAvailElecKw[i] = np.min([
            self.curMaxElecKw[i], veh.mcMaxElecInKw])

        if self.curMaxElecKw[i] > 0:
            # limit power going into e-machine controller to
            if self.curMaxAvailElecKw[i] == np.max(veh.mcKwInArray):
                self.mcElecInLimKw[i] = np.min([
                    veh.mcKwOutArray[len(veh.mcKwOutArray) - 1], veh.maxMotorKw])
            else:
                self.mcElecInLimKw[i] = np.min([veh.mcKwOutArray[np.argmax(veh.mcKwInArray > np.min([np.max(veh.mcKwInArray) -
                                            0.01, self.curMaxAvailElecKw[i]])) - 1], veh.maxMotorKw])
        else:
            self.mcElecInLimKw[i] = 0.0

        # Motor transient power limit
        self.mcTransiLimKw[i] = abs(
            self.mcMechKwOutAch[i-1]) + ((veh.maxMotorKw / veh.motorSecsToPeakPwr) * (cyc.secs[i]))

        self.curMaxMcKwOut[i] = np.max([np.min([
            self.mcElecInLimKw[i], self.mcTransiLimKw[i], veh.maxMotorKw]), -veh.maxMotorKw])

        if self.curMaxMcKwOut[i] == 0:
            self.curMaxMcElecKwIn[i] = 0
        else:
            if self.curMaxMcKwOut[i] == veh.maxMotorKw:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / veh.mcFullEffArray[np.max([1, np.argmax(veh.mcKwOutArray
                                                > np.min([veh.maxMotorKw - 0.01, self.curMaxMcKwOut[i]])) - 1])]

        if veh.maxMotorKw == 0:
            self.essLimMcRegenPercKw[i] = 0.0

        else:
            self.essLimMcRegenPercKw[i] = np.min([
                (self.curMaxEssChgKw[i] + self.auxInKw[i]) / veh.maxMotorKw, 1])
        if self.curMaxEssChgKw[i] == 0:
            self.essLimMcRegenKw[i] = 0.0

        else:
            if veh.maxMotorKw == self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i]:
                self.essLimMcRegenKw[i] = np.min([
                    veh.maxMotorKw, self.curMaxEssChgKw[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]])
            else:
                self.essLimMcRegenKw[i] = np.min([veh.maxMotorKw, self.curMaxEssChgKw[i] / veh.mcFullEffArray
                                                [np.max([1, np.argmax(veh.mcKwOutArray > np.min([veh.maxMotorKw - 0.01, 
                                                self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i]])) - 1])]])

        self.curMaxMechMcKwIn[i] = np.min([
            self.essLimMcRegenKw[i], veh.maxMotorKw])
        self.curMaxTracKw[i] = (((veh.wheelCoefOfFric * veh.driveAxleWeightFrac * veh.vehKg * gravityMPerSec2)
                                    / (1 + ((veh.vehCgM * veh.wheelCoefOfFric) / veh.wheelBaseM))) / 1000.0) * (self.maxTracMps[i])

        if veh.fcEffType == 4:

            if veh.noElecSys == True or veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = np.min([
                    (self.curMaxMcKwOut[i] - self.auxInKw[i]) * veh.transEff, self.curMaxTracKw[i] / veh.transEff])
                self.debug_flag[i] = 1

            else:
                self.curMaxTransKwOut[i] = np.min([(self.curMaxMcKwOut[i] - np.min([
                    self.curMaxElecKw[i], 0])) * veh.transEff, self.curMaxTracKw[i] / veh.transEff])
                self.debug_flag[i] = 2

        else:

            if veh.noElecSys == True or veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = np.min([(self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                self.auxInKw[i]) * veh.transEff, self.curMaxTracKw[i] / veh.transEff])
                self.debug_flag[i] = 3

            else:
                self.curMaxTransKwOut[i] = np.min([(self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                np.min([self.curMaxElecKw[i], 0])) * veh.transEff, self.curMaxTracKw[i] / veh.transEff])
                self.debug_flag[i] = 4
        
    def set_power_calcs(self, i, cyc, veh):
        """Calculate and set power variables at time step 'i'.
        Arguments
        ------------
        i: index of time step
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        self.cycDragKw[i] = 0.5 * airDensityKgPerM3 * veh.dragCoef * \
            veh.frontalAreaM2 * \
            (((self.mpsAch[i-1] + cyc.cycMps[i]) / 2.0)**3) / 1000.0
        self.cycAccelKw[i] = (veh.vehKg / (2.0 * (cyc.secs[i]))) * \
            ((cyc.cycMps[i]**2) - (self.mpsAch[i-1]**2)) / 1000.0
        self.cycAscentKw[i] = gravityMPerSec2 * np.sin(np.arctan(
            cyc.cycGrade[i])) * veh.vehKg * ((self.mpsAch[i-1] + cyc.cycMps[i]) / 2.0) / 1000.0
        self.cycTracKwReq[i] = self.cycDragKw[i] + \
            self.cycAccelKw[i] + self.cycAscentKw[i]
        self.spareTracKw[i] = self.curMaxTracKw[i] - self.cycTracKwReq[i]
        self.cycRrKw[i] = gravityMPerSec2 * veh.wheelRrCoef * \
            veh.vehKg * ((self.mpsAch[i-1] + cyc.cycMps[i]) / 2.0) / 1000.0
        self.cycWheelRadPerSec[i] = cyc.cycMps[i] / veh.wheelRadiusM
        self.cycTireInertiaKw[i] = (((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * (self.cycWheelRadPerSec[i]**2.0)) / cyc.secs[i]) -
                                    ((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * ((self.mpsAch[i-1] / veh.wheelRadiusM)**2.0)) / cyc.secs[i])) / 1000.0

        self.cycWheelKwReq[i] = self.cycTracKwReq[i] + \
            self.cycRrKw[i] + self.cycTireInertiaKw[i]
        self.regenContrLimKwPerc[i] = veh.maxRegen / (1 + veh.regenA * np.exp(-veh.regenB * (
            (cyc.cycMph[i] + self.mpsAch[i-1] * mphPerMps) / 2.0 + 1 - 0)))
        self.cycRegenBrakeKw[i] = np.max([np.min([
            self.curMaxMechMcKwIn[i] * veh.transEff, self.regenContrLimKwPerc[i] * -self.cycWheelKwReq[i]]), 0])
        self.cycFricBrakeKw[i] = - \
            np.min([self.cycRegenBrakeKw[i] + self.cycWheelKwReq[i], 0])
        self.cycTransKwOutReq[i] = self.cycWheelKwReq[i] + \
            self.cycFricBrakeKw[i]

        if self.cycTransKwOutReq[i] <= self.curMaxTransKwOut[i]:
            self.cycMet[i] = 1
            self.transKwOutAch[i] = self.cycTransKwOutReq[i]

        else:
            self.cycMet[i] = -1
            self.transKwOutAch[i] = self.curMaxTransKwOut[i]
        
    def set_speed_dist_calcs(self, i, cyc, veh):
        """Calculate and set variables dependent on speed
        Arguments
        ------------
        i: index of time step
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        # Cycle is met
        if self.cycMet[i] == 1:
            self.mpsAch[i] = cyc.cycMps[i]

        #Cycle is not met
        else:
            Drag3 = (1.0 / 16.0) * airDensityKgPerM3 * \
                veh.dragCoef * veh.frontalAreaM2
            Accel2 = veh.vehKg / (2.0 * (cyc.secs[i]))
            Drag2 = (3.0 / 16.0) * airDensityKgPerM3 * \
                veh.dragCoef * veh.frontalAreaM2 * self.mpsAch[i-1]
            Wheel2 = 0.5 * veh.wheelInertiaKgM2 * \
                veh.numWheels / (cyc.secs[i] * (veh.wheelRadiusM**2))
            Drag1 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * \
                veh.frontalAreaM2 * ((self.mpsAch[i-1])**2)
            Roll1 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg / 2.0)
            Ascent1 = (gravityMPerSec2 *
                        np.sin(np.arctan(cyc.cycGrade[i])) * veh.vehKg / 2.0)
            Accel0 = - \
                (veh.vehKg * ((self.mpsAch[i-1])**2)) / (2.0 * (cyc.secs[i]))
            Drag0 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * \
                veh.frontalAreaM2 * ((self.mpsAch[i-1])**3)
            Roll0 = (gravityMPerSec2 * veh.wheelRrCoef *
                        veh.vehKg * self.mpsAch[i-1] / 2.0)
            Ascent0 = (
                gravityMPerSec2 * np.sin(np.arctan(cyc.cycGrade[i])) * veh.vehKg * self.mpsAch[i-1] / 2.0)
            Wheel0 = -((0.5 * veh.wheelInertiaKgM2 * veh.numWheels *
                        (self.mpsAch[i-1]**2)) / (cyc.secs[i] * (veh.wheelRadiusM**2)))

            Total3 = Drag3 / 1e3
            Total2 = (Accel2 + Drag2 + Wheel2) / 1e3
            Total1 = (Drag1 + Roll1 + Ascent1) / 1e3
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / \
                1e3 - self.curMaxTransKwOut[i]

            Total = [Total3, Total2, Total1, Total0]
            Total_roots = np.roots(Total)
            ind = np.argmin(abs(cyc.cycMps[i] - Total_roots))
            self.mpsAch[i] = Total_roots[ind]

        self.mphAch[i] = self.mpsAch[i] * mphPerMps
        self.distMeters[i] = self.mpsAch[i] * cyc.secs[i]
        self.distMiles[i] = self.distMeters[i] * (1.0 / metersPerMile)
        
    def set_hybrid_cont_calcs(self, i, cyc, veh):
        """Hybrid control calculations.  
        Arguments
        ------------
        i: index of time step
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        if self.transKwOutAch[i] > 0:
            self.transKwInAch[i] = self.transKwOutAch[i] / veh.transEff
        else:
            self.transKwInAch[i] = self.transKwOutAch[i] * veh.transEff

        if self.cycMet[i] == 1:

            if veh.fcEffType == 4:
                self.minMcKw2HelpFc[i] = np.max([
                    self.transKwInAch[i], -self.curMaxMechMcKwIn[i]])

            else:
                self.minMcKw2HelpFc[i] = np.max([
                    self.transKwInAch[i] - self.curMaxFcKwOut[i], -self.curMaxMechMcKwIn[i]])
        else:
            self.minMcKw2HelpFc[i] = np.max([
                self.curMaxMcKwOut[i], -self.curMaxMechMcKwIn[i]])

        if veh.noElecSys == True:
            self.regenBufferSoc[i] = 0

        elif veh.chargingOn:
            self.regenBufferSoc[i] = np.max([
                veh.maxSoc - (veh.maxRegenKwh / veh.maxEssKwh), (veh.maxSoc + veh.minSoc) / 2])

        else:
            self.regenBufferSoc[i] = np.max([((veh.maxEssKwh * veh.maxSoc) - (0.5 * veh.vehKg * (cyc.cycMps[i]**2) * (1.0 / 1000)
                                        * (1.0 / 3600) * veh.motorPeakEff * veh.maxRegen)) / veh.maxEssKwh, veh.minSoc])

            self.essRegenBufferDischgKw[i] = np.min([self.curMaxEssKwOut[i], np.max([
                0, (self.soc[i-1] - self.regenBufferSoc[i]) * veh.maxEssKwh * 3600 / cyc.secs[i]])])

            self.maxEssRegenBufferChgKw[i] = np.min([np.max([
                0, (self.regenBufferSoc[i] - self.soc[i-1]) * veh.maxEssKwh * 3600.0 / cyc.secs[i]]), self.curMaxEssChgKw[i]])

        if veh.noElecSys == True:
            self.accelBufferSoc[i] = 0

        else:
            self.accelBufferSoc[i] = np.min([np.max([(((((((veh.maxAccelBufferMph * (1 / mphPerMps))**2)) - ((cyc.cycMps[i]**2))) /
                                                (((veh.maxAccelBufferMph * (1 / mphPerMps))**2))) * (np.min([veh.maxAccelBufferPercOfUseableSoc * \
                                                (veh.maxSoc - veh.minSoc), veh.maxRegenKwh / veh.maxEssKwh]) * veh.maxEssKwh)) / veh.maxEssKwh) + \
                                                veh.minSoc, veh.minSoc]), veh.maxSoc])

            self.essAccelBufferChgKw[i] = np.max([
                0, (self.accelBufferSoc[i] - self.soc[i-1]) * veh.maxEssKwh * 3600.0 / cyc.secs[i]])
            self.maxEssAccelBufferDischgKw[i] = np.min([np.max([
                0, (self.soc[i-1] - self.accelBufferSoc[i]) * veh.maxEssKwh * 3600 / cyc.secs[i]]), self.curMaxEssKwOut[i]])

        if self.regenBufferSoc[i] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = np.max([np.min([((self.soc[i-1] - (self.regenBufferSoc[i] + self.accelBufferSoc[i]) / 2) * veh.maxEssKwh * 3600.0) /
                                                    cyc.secs[i], self.curMaxEssKwOut[i]]), -self.curMaxEssChgKw[i]])

        elif self.soc[i-1] > self.regenBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = np.max([np.min([
                self.essRegenBufferDischgKw[i], self.curMaxEssKwOut[i]]), -self.curMaxEssChgKw[i]])

        elif self.soc[i-1] < self.accelBufferSoc[i]:
            self.essAccelRegenDischgKw[i] = np.max([
                np.min([-1.0 * self.essAccelBufferChgKw[i], self.curMaxEssKwOut[i]]), -self.curMaxEssChgKw[i]])

        else:
            self.essAccelRegenDischgKw[i] = np.max([
                np.min([0, self.curMaxEssKwOut[i]]), -self.curMaxEssChgKw[i]])

        self.fcKwGapFrEff[i] = abs(self.transKwOutAch[i] - veh.maxFcEffKw)

        if veh.noElecSys == True:
            self.mcElectInKwForMaxFcEff[i] = 0

        elif self.transKwOutAch[i] < veh.maxFcEffKw:

            if self.fcKwGapFrEff[i] == veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1] * -1
            else:
                self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / veh.mcFullEffArray[np.max([
                    1, np.argmax(veh.mcKwOutArray > np.min([veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i]])) - 1])] * -1

        else:

            if self.fcKwGapFrEff[i] == veh.maxMotorKw:
                self.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[len(
                    veh.mcKwInArray) - 1]
            else:
                self.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[np.argmax(
                    veh.mcKwOutArray > np.min([veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i]])) - 1]

        if veh.noElecSys == True:
            self.electKwReq4AE[i] = 0

        elif self.transKwInAch[i] > 0:
            if self.transKwInAch[i] == veh.maxMotorKw:

                self.electKwReq4AE[i] = self.transKwInAch[i] / \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1] + self.auxInKw[i]
            else:
                self.electKwReq4AE[i] = self.transKwInAch[i] / veh.mcFullEffArray[np.max([1, np.argmax(
                    veh.mcKwOutArray > np.min([veh.maxMotorKw - 0.01, self.transKwInAch[i]])) - 1])] + self.auxInKw[i]

        else:
            self.electKwReq4AE[i] = 0

        self.prevfcTimeOn[i] = self.fcTimeOn[i-1]

        if veh.maxFuelConvKw == 0:
            self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and self.transKwInAch[i] <= self.curMaxMcKwOut[i] and (self.electKwReq4AE[i] < self.curMaxElecKw[i] or veh.maxFuelConvKw == 0)

        else:
            self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and self.transKwInAch[i] <=self.curMaxMcKwOut[i] and (self.electKwReq4AE[i] < self.curMaxElecKw[i] \
                or veh.maxFuelConvKw == 0) and (cyc.cycMph[i] - 0.00001 <=veh.mphFcOn or veh.chargingOn) and self.electKwReq4AE[i]<=veh.kwDemandFcOn

        if self.canPowerAllElectrically[i]:

            if self.transKwInAch[i] <+self.auxInKw[i]:
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
            self.essAEKwOut[i] = np.max([-self.curMaxEssChgKw[i], -self.maxEssRegenBufferChgKw[i], np.min([0, self.curMaxRoadwayChgKw[i] - (
                self.transKwInAch[i] + self.auxInKw[i])]), np.min([self.curMaxEssKwOut[i], self.desiredEssKwOutForAE[i]])])

        else:
            self.essAEKwOut[i] = 0

        self.erAEKwOut[i] = np.min([np.max([0, self.transKwInAch[i] + self.auxInKw[i] - self.essAEKwOut[i]]), self.curMaxRoadwayChgKw[i]])
    
    def set_fc_forced_state(self, i, cyc, veh):
        """Calculate control variables related to engine on/off state
        Arguments       
        ------------
        i: index of time step
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        # force fuel converter on if it was on in the previous time step, but only if fc
        # has not been on longer than minFcTimeOn
        if self.prevfcTimeOn[i] > 0 and self.prevfcTimeOn[i] < veh.minFcTimeOn - cyc.secs[i]:
            self.fcForcedOn[i] = True
        else:
            self.fcForcedOn[i] = False

        # Engine only mode
        if self.fcForcedOn[i] == False or self.canPowerAllElectrically[i] == False:
            self.fcForcedState[i] = 1
            self.mcMechKw4ForcedFc[i] = 0

        # Engine maximum efficiency mode
        elif self.transKwInAch[i] < 0:
            self.fcForcedState[i] = 2
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i]

        # All electric mode
        elif veh.maxFcEffKw == self.transKwInAch[i]:
            self.fcForcedState[i] = 3
            self.mcMechKw4ForcedFc[i] = 0

        # Engine forced on mode
        elif veh.idleFcKw > self.transKwInAch[i] and self.cycAccelKw[i] >= 0:
            self.fcForcedState[i] = 4
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - veh.idleFcKw

        # Engine + motor mode
        elif veh.maxFcEffKw > self.transKwInAch[i]:
            self.fcForcedState[i] = 5
            self.mcMechKw4ForcedFc[i] = 0

        # Fuel cell mode
        else:
            self.fcForcedState[i] = 6
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - \
                veh.maxFcEffKw

    def set_hybrid_cont_decisions(self, i, cyc, veh):
        """Hybrid control decisions.
        Arguments
        ------------
        i: index of time step
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        if (-self.mcElectInKwForMaxFcEff[i] - self.curMaxRoadwayChgKw[i]) > 0:
            self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] -
                                            self.curMaxRoadwayChgKw[i]) * veh.essDischgToFcMaxEffPerc

        else:
            self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] - \
                                            self.curMaxRoadwayChgKw[i]) * veh.essChgToFcMaxEffPerc

        if self.accelBufferSoc[i] > self.regenBufferSoc[i]:
            self.essKwIfFcIsReq[i] = np.min([self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            np.max([-self.curMaxEssChgKw[i], self.essAccelRegenDischgKw[i]])])

        elif self.essRegenBufferDischgKw[i] > 0:
            self.essKwIfFcIsReq[i] = np.min([self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            np.max([-self.curMaxEssChgKw[i], np.min([self.essAccelRegenDischgKw[i], self.mcElecInLimKw[i] + self.auxInKw[i], 
                                            np.max([self.essRegenBufferDischgKw[i], self.essDesiredKw4FcEff[i]])])])])

        elif self.essAccelBufferChgKw[i] > 0:
            self.essKwIfFcIsReq[i] = np.min([self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            np.max([-self.curMaxEssChgKw[i], np.max([-1 * self.maxEssRegenBufferChgKw[i], np.min([-self.essAccelBufferChgKw[i], 
                                            self.essDesiredKw4FcEff[i]])])])])

        elif self.essDesiredKw4FcEff[i] > 0:
            self.essKwIfFcIsReq[i] = np.min([self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            np.max([-self.curMaxEssChgKw[i], np.min([self.essDesiredKw4FcEff[i], self.maxEssAccelBufferDischgKw[i]])])])

        else:
            self.essKwIfFcIsReq[i] = np.min([self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i],
                                            np.max([-self.curMaxEssChgKw[i], np.max([self.essDesiredKw4FcEff[i], -self.maxEssRegenBufferChgKw[i]])])])

        self.erKwIfFcIsReq[i] = np.max([0, np.min([self.curMaxRoadwayChgKw[i], self.curMaxMechMcKwIn[i],
                                    self.essKwIfFcIsReq[i] - self.mcElecInLimKw[i] + self.auxInKw[i]])])

        self.mcElecKwInIfFcIsReq[i] = self.essKwIfFcIsReq[i] + \
            self.erKwIfFcIsReq[i] - self.auxInKw[i]

        if veh.noElecSys == True:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] == 0:
            self.mcKwIfFcIsReq[i] = 0

        elif self.mcElecKwInIfFcIsReq[i] > 0:

            if self.mcElecKwInIfFcIsReq[i] == np.max(veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * veh.mcFullEffArray[np.max([1, np.argmax(
                    veh.mcKwInArray > np.min([np.max(veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i]])) - 1])]

        else:
            if self.mcElecKwInIfFcIsReq[i] * -1 == np.max(veh.mcKwInArray):
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / (veh.mcFullEffArray[np.max([1, np.argmax(
                    veh.mcKwInArray > np.min([np.max(veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i] * -1])) - 1])])

        if veh.maxMotorKw == 0:
            self.mcMechKwOutAch[i] = 0

        elif self.fcForcedOn[i] == True and self.canPowerAllElectrically[i] == True and (veh.vehPtType == 2.0 or veh.vehPtType == 3.0) and veh.fcEffType !=4:
            self.mcMechKwOutAch[i] = self.mcMechKw4ForcedFc[i]

        elif self.transKwInAch[i] <=0:

            if veh.fcEffType !=4 and veh.maxFuelConvKw> 0:
                if self.canPowerAllElectrically[i] == 1:
                    self.mcMechKwOutAch[i] = - \
                        np.min([self.curMaxMechMcKwIn[i], -self.transKwInAch[i]])
                else:
                    self.mcMechKwOutAch[i] = np.min([-np.min([self.curMaxMechMcKwIn[i], -self.transKwInAch[i]]),
                                                    np.max([-self.curMaxFcKwOut[i], self.mcKwIfFcIsReq[i]])])
            else:
                self.mcMechKwOutAch[i] = np.min([
                    -np.min([self.curMaxMechMcKwIn[i], -self.transKwInAch[i]]), -self.transKwInAch[i]])

        elif self.canPowerAllElectrically[i] == 1:
            self.mcMechKwOutAch[i] = self.transKwInAch[i]

        else:
            self.mcMechKwOutAch[i] = np.max([
                self.minMcKw2HelpFc[i], self.mcKwIfFcIsReq[i]])

        if self.mcMechKwOutAch[i] == 0:
            self.mcElecKwInAch[i] = 0.0
            self.motor_index_debug[i] = 0

        elif self.mcMechKwOutAch[i] < 0:

            if self.mcMechKwOutAch[i] * -1 == np.max(veh.mcKwInArray):
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * veh.mcFullEffArray[np.max([1, np.argmax(
                    veh.mcKwInArray > np.min([np.max(veh.mcKwInArray) - 0.01, self.mcMechKwOutAch[i] * -1])) - 1])]

        else:
            if veh.maxMotorKw == self.mcMechKwOutAch[i]:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / veh.mcFullEffArray[np.max([1, np.argmax(
                    veh.mcKwOutArray > np.min([veh.maxMotorKw - 0.01, self.mcMechKwOutAch[i]])) - 1])]

        if self.curMaxRoadwayChgKw[i] == 0:
            self.roadwayChgKwOutAch[i] = 0

        elif veh.fcEffType == 4:
            self.roadwayChgKwOutAch[i] = np.max([
                0, self.mcElecKwInAch[i], self.maxEssRegenBufferChgKw[i], self.essRegenBufferDischgKw[i], self.curMaxRoadwayChgKw[i]])

        elif self.canPowerAllElectrically[i] == 1:
            self.roadwayChgKwOutAch[i] = self.erAEKwOut[i]

        else:
            self.roadwayChgKwOutAch[i] = self.erKwIfFcIsReq[i]

        self.minEssKw2HelpFc[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - \
            self.curMaxFcKwOut[i] - self.roadwayChgKwOutAch[i]

        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            self.essKwOutAch[i] = 0

        elif veh.fcEffType == 4:

            if self.transKwOutAch[i] >=0:
                self.essKwOutAch[i] = np.min([np.max([self.minEssKw2HelpFc[i], self.essDesiredKw4FcEff[i], self.essAccelRegenDischgKw[i]]),
                                            self.curMaxEssKwOut[i], self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i]])

            else:
                self.essKwOutAch[i] = self.mcElecKwInAch[i] + \
                    self.auxInKw[i] - self.roadwayChgKwOutAch[i]

        elif self.highAccFcOnTag[i] == 1 or veh.noElecAux == True:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] - \
                self.roadwayChgKwOutAch[i]

        else:
            self.essKwOutAch[i] = self.mcElecKwInAch[i] + \
                self.auxInKw[i] - self.roadwayChgKwOutAch[i]

        if veh.maxFuelConvKw == 0:
            self.fcKwOutAch[i] = 0

        elif veh.fcEffType == 4:
            self.fcKwOutAch[i] = np.min([self.curMaxFcKwOut[i], np.max([
                0, self.mcElecKwInAch[i] + self.auxInKw[i] - self.essKwOutAch[i] - self.roadwayChgKwOutAch[i]])])

        elif veh.noElecSys == True or veh.noElecAux == True or self.highAccFcOnTag[i] == 1:
            self.fcKwOutAch[i] = np.min([self.curMaxFcKwOut[i], np.max([
                0, self.transKwInAch[i] - self.mcMechKwOutAch[i] + self.auxInKw[i]])])

        else:
            self.fcKwOutAch[i] = np.min([self.curMaxFcKwOut[i], np.max([
                0, self.transKwInAch[i] - self.mcMechKwOutAch[i]])])

        if self.fcKwOutAch[i] == 0:
            self.fcKwInAch[i] = 0.0
            self.fcKwOutAch_pct[i] = 0

        if veh.maxFuelConvKw == 0:
            self.fcKwOutAch_pct[i] = 0
        else:
            self.fcKwOutAch_pct[i] = self.fcKwOutAch[i] / veh.maxFuelConvKw

        if self.fcKwOutAch[i] == 0:
            self.fcKwInAch[i] = 0
        else:
            if self.fcKwOutAch[i] == veh.fcMaxOutkW:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / \
                    veh.fcEffArray[len(veh.fcEffArray) - 1]
            else:
                self.fcKwInAch[i] = self.fcKwOutAch[i] / (veh.fcEffArray[np.max([1, np.argmax(
                    veh.fcKwOutArray > np.min([self.fcKwOutAch[i], veh.fcMaxOutkW - 0.001])) - 1])])

        self.fsKwOutAch[i] = np.copy(self.fcKwInAch[i])

        self.fsKwhOutAch[i] = self.fsKwOutAch[i] * \
            cyc.secs[i] * (1 / 3600.0)

        if veh.noElecSys == True:
            self.essCurKwh[i] = 0

        elif self.essKwOutAch[i] < 0:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * \
                (cyc.secs[i] / 3600.0) * np.sqrt(veh.essRoundTripEff)

        else:
            self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * \
                (cyc.secs[i] / 3600.0) * (1 / np.sqrt(veh.essRoundTripEff))

        if veh.maxEssKwh == 0:
            self.soc[i] = 0.0

        else:
            self.soc[i] = self.essCurKwh[i] / veh.maxEssKwh

        if self.canPowerAllElectrically[i] == True and self.fcForcedOn[i] == False and self.fcKwOutAch[i] == 0:
            self.fcTimeOn[i] = 0
        else:
            self.fcTimeOn[i] = self.fcTimeOn[i-1] + cyc.secs[i]

    # post-processing
    def get_output(self, cyc, veh):
        """Calculate finalized results
        Arguments
        ------------
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles
        
        Returns
        ------------
        output: dict of summary output variables"""

        output = {}

        if sum(self.fsKwhOutAch) == 0:
            output['mpgge'] = 0

        else:
            output['mpgge'] = sum(self.distMiles) / \
                (sum(self.fsKwhOutAch) * (1 / kWhPerGGE))

        self.roadwayChgKj = sum(self.roadwayChgKwOutAch * cyc.secs)
        self.essDischKj = - \
            (self.soc[-1] - self.soc[0]) * veh.maxEssKwh * 3600.0
        output['battery_kWh_per_mi'] = (
            self.essDischKj / 3600.0) / sum(self.distMiles)
        self.battery_kWh_per_mi = output['battery_kWh_per_mi']
        output['electric_kWh_per_mi'] = (
            (self.roadwayChgKj + self.essDischKj) / 3600.0) / sum(self.distMiles)
        self.electric_kWh_per_mi = output['electric_kWh_per_mi']
        output['maxTraceMissMph'] = mphPerMps * \
            np.max(abs(cyc.cycMps - self.mpsAch))
        self.maxTraceMissMph = output['maxTraceMissMph']
        self.fuelKj = sum(np.asarray(self.fsKwOutAch) * np.asarray(cyc.secs))
        self.roadwayChgKj = sum(np.asarray(
            self.roadwayChgKwOutAch) * np.asarray(cyc.secs))
        essDischgKj = -(self.soc[-1] - self.soc[0]) * veh.maxEssKwh * 3600.0

        if (self.fuelKj + self.roadwayChgKj) == 0:
            output['ess2fuelKwh'] = 1.0

        else:
            output['ess2fuelKwh'] = essDischgKj / \
                (self.fuelKj + self.roadwayChgKj)

        self.ess2fuelKwh = output['ess2fuelKwh']

        output['initial_soc'] = self.soc[0]
        output['final_soc'] = self.soc[-1]

        if output['mpgge'] == 0:
            # hardcoded conversion
            Gallons_gas_equivalent_per_mile = output['electric_kWh_per_mi'] / 33.7

        else:
            Gallons_gas_equivalent_per_mile = 1 / \
                output['mpgge'] + output['electric_kWh_per_mi'] / \
                33.7  # hardcoded conversion

        self.Gallons_gas_equivalent_per_mile = Gallons_gas_equivalent_per_mile

        output['mpgge_elec'] = 1 / Gallons_gas_equivalent_per_mile
        output['soc'] = np.asarray(self.soc)
        output['distance_mi'] = sum(self.distMiles)
        duration_sec = cyc.cycSecs[-1] - cyc.cycSecs[0]
        output['avg_speed_mph'] = sum(
            self.distMiles) / (duration_sec / 3600.0)
        self.avg_speed_mph = output['avg_speed_mph']
        self.accel = np.diff(self.mphAch) / np.diff(cyc.cycSecs)
        output['avg_accel_mphps'] = np.mean(self.accel[self.accel > 0])
        self.avg_accel_mphps = output['avg_accel_mphps']

        if np.max(self.mphAch) > 60:
            output['ZeroToSixtyTime_secs'] = np.interp(60, self.mphAch, cyc.cycSecs)

        else:
            output['ZeroToSixtyTime_secs'] = 0.0

        #######################################################################
        ####  Time series information for additional analysis / debugging. ####
        ####             Add parameters of interest as needed.             ####
        #######################################################################

        output['fcKwOutAch'] = np.asarray(self.fcKwOutAch)
        output['fsKwhOutAch'] = np.asarray(self.fsKwhOutAch)
        output['fcKwInAch'] = np.asarray(self.fcKwInAch)
        output['time'] = np.asarray(cyc.cycSecs)

        return output

    # optional post-processing methods
    def get_diagnostics(self, cyc):
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
            output[search[1] + 'Kj' + search[2] + 'Pos'] = np.trapz(tempvars[var + 'Pos'], cyc.cycSecs)
            output[search[1] + 'Kj' + search[2] + 'Neg'] = np.trapz(tempvars[var + 'Neg'], cyc.cycSecs)
        
        output['distMilesFinal'] = sum(self.distMiles)
        output['mpgge'] = sum(self.distMiles) / sum(self.fsKwhOutAch) * kWhPerGGE
    
        return output

    def set_battery_wear(self, veh):
        """Battery wear calcs
        Arguments:
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step
        
        Output: tarr"""

        self.addKwh[1:] = np.array([
            (self.essCurKwh[i] - self.essCurKwh[i-1]) + self.addKwh[i-1]
            if self.essCurKwh[i] > self.essCurKwh[i-1]
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.dodCycs[1:] = np.array([
            self.addKwh[i-1] / veh.maxEssKwh if self.addKwh[i] == 0
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.essPercDeadArray = np.array([
            np.power(veh.essLifeCoefA, 1.0 / veh.essLifeCoefB) / np.power(self.dodCycs[i], 
            1.0 / veh.essLifeCoefB)
            if self.dodCycs[i] != 0
            else 0
            for i in range(0, len(self.essCurKwh))])

    def set_energy_audit(self, cyc, veh):
        """Energy Audit Calculations
        Arguments
        ------------
        cyc: instance of LoadData.Cycle class
        veh: instance of LoadData.Vehicle class
        initSoc: initial SOC for electrified vehicles"""

        self.dragKw[1:] = 0.5 * airDensityKgPerM3 * veh.dragCoef * \
            veh.frontalAreaM2 * \
            (((self.mpsAch[:-1] + self.mpsAch[1:]) / 2.0)**3) / 1000.0
        
        self.essLossKw[1:] = np.array(
            [0 if (veh.maxEssKw == 0 or veh.maxEssKwh == 0) 
            else -self.essKwOutAch[i] - (-self.essKwOutAch[i] * np.sqrt(veh.essRoundTripEff)) 
                if self.essKwOutAch[i] < 0 
            else self.essKwOutAch[i] * (1.0 / np.sqrt(veh.essRoundTripEff)) - self.essKwOutAch[i] 
            for i in range(1, len(cyc.cycSecs))])

        self.accelKw[1:] = (veh.vehKg / (2.0 * (cyc.secs[1:]))) * \
            ((self.mpsAch[1:]**2) - (self.mpsAch[:-1]**2)) / 1000.0
        self.ascentKw[1:] = gravityMPerSec2 * np.sin(np.arctan(cyc.cycGrade[1:])) * veh.vehKg * (
            (self.mpsAch[:-1] + self.mpsAch[1:]) / 2.0) / 1000.0
        self.rrKw[1:] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * \
            ((self.mpsAch[:-1] + self.mpsAch[1:]) / 2.0) / 1000.0
