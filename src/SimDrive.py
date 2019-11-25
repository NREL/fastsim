"""
##############################################################################
##############################################################################
Pythonic copy of NREL's FASTSim
(Future Automotive Systems Technology Simulator)
Input Arguments
1) cyc: dictionary defining drive cycle to be simulated
        cyc['cycSecs']: drive cycle time in seconds (begins at zero)
        cyc['cycMps']: desired vehicle speed in meters per second
        cyc['cycGrade']: road grade
        cyc['cycRoadType']: Functional Class of GPS data
2) veh: dictionary defining vehicle parameters for simulation
Output Arguments
A dictionary containing scalar values summarizing simulation
(mpg, Wh/mi, avg engine power, etc) and/or time-series results for component
power and efficiency. Currently these values are user clarified in the code by assigning values to the 'output' dictionary.
    List of Abbreviations
    cur = current time step
    prev = previous time step

    cyc = drive cycle
    secs = seconds
    mps = meters per second
    mph = miles per hour
    kw = kilowatts, unit of power
    kwh = kilowatt-hour, unit of energy
    kg = kilograms, unit of mass
    max = maximum
    min = minimum
    avg = average
    fs = fuel storage (eg. gasoline/diesel tank, pressurized hydrogen tank)
    fc = fuel converter (eg. internal combustion engine, fuel cell)
    mc = electric motor/generator and controller
    ess = energy storage system (eg. high voltage traction battery)

    chg = charging of a component
    dis = discharging of a component
    lim = limit of a component
    regen = associated with regenerative braking
    des = desired value
    ach = achieved value
    in = component input
    out = component output

##############################################################################
##############################################################################
"""

### Import necessary python modules
import numpy as np
import pandas as pd
from collections import namedtuple
import warnings
warnings.simplefilter('ignore')

class TimeArrays(object):
    """Class for contaning time series data used by sim_drive_sub"""
    def __init__(self, cycSecs, veh, initSoc):
        """Initializes arrays of time dependent variables as attributes of self."""
        super().__init__()

        # Component Limits -- calculated dynamically"
        comp_lim_list = ['curMaxFsKwOut', 'fcTransLimKw', 'fcFsLimKw', 'fcMaxKwIn', 'curMaxFcKwOut', 
            'essCapLimDischgKw', 'curMaxEssKwOut', 'curMaxAvailElecKw', 'essCapLimChgKw', 'curMaxEssChgKw', 
            'curMaxElecKw', 'mcElecInLimKw', 'mcTransiLimKw', 'curMaxMcKwOut', 'essLimMcRegenPercKw', 
            'essLimMcRegenKw', 'curMaxMechMcKwIn', 'curMaxTransKwOut']

        ### Drive Train
        drivetrain_list = ['cycDragKw', 'cycAccelKw', 'cycAscentKw', 'cycTracKwReq', 'curMaxTracKw', 
            'spareTracKw', 'cycRrKw', 'cycWheelRadPerSec', 'cycTireInertiaKw', 'cycWheelKwReq', 
            'regenContrLimKwPerc', 'cycRegenBrakeKw', 'cycFricBrakeKw', 'cycTransKwOutReq', 'cycMet', 
            'transKwOutAch', 'transKwInAch', 'curSocTarget', 'minMcKw2HelpFc', 'mcMechKwOutAch', 
            'mcElecKwInAch', 'auxInKw', 'roadwayChgKwOutAch', 'minEssKw2HelpFc', 'essKwOutAch', 'fcKwOutAch', 
            'fcKwOutAch_pct', 'fcKwInAch', 'fsKwOutAch', 'fsKwhOutAch', 'essCurKwh', 'soc']

        #roadwayMaxEssChg  # *** CB is not sure why this is here
        
        # Vehicle Attributes, Control Variables
        control_list = ['regenBufferSoc' , 'essRegenBufferDischgKw', 'maxEssRegenBufferChgKw', 
            'essAccelBufferChgKw', 'accelBufferSoc', 'maxEssAccelBufferDischgKw', 'essAccelRegenDischgKw', 
            'mcElectInKwForMaxFcEff', 'electKwReq4AE', 'canPowerAllElectrically', 'desiredEssKwOutForAE', 
            'essAEKwOut', 'erAEKwOut', 'essDesiredKw4FcEff', 'essKwIfFcIsReq', 'curMaxMcElecKwIn', 
            'fcKwGapFrEff', 'erKwIfFcIsReq', 'mcElecKwInIfFcIsReq', 'mcKwIfFcIsReq', 'fcForcedOn', 
            'fcForcedState', 'mcMechKw4ForcedFc', 'fcTimeOn', 'prevfcTimeOn']

        ### Additional Variables
        misc_list = ['mpsAch', 'mphAch', 'distMeters', 'distMiles', 'highAccFcOnTag', 'reachedBuff', 
            'maxTracMps', 'addKwh', 'dodCycs', 'essPercDeadArray', 'dragKw', 'essLossKw', 'accelKw', 
            'ascentKw', 'rrKw', 'motor_index_debug', 'debug_flag', 'curMaxRoadwayChgKw']

        # create and initialize time array dataframe
        attributes = ['cycSecs'] + comp_lim_list + \
            drivetrain_list + control_list + misc_list

        # assign numpy.zeros of the same length as cycSecs to self attributes
        for attribute in attributes:
            self.__setattr__(attribute, np.zeros(len(cycSecs)))

        self.fcForcedOn = np.array([False] * len(cycSecs))
        # self.curMaxRoadwayChgKw = np.interp(
        #     cycRoadType, veh.MaxRoadwayChgKw_Roadway, veh.MaxRoadwayChgKw)  
            # *** this is just zeros, and I need to verify that it was zeros before and also 
            # verify that this is the correct behavior.  CB

        ###  Assign First Value  ###
        ### Drive Train
        self.cycMet[0] = 1
        self.curSocTarget[0] = veh.maxSoc
        self.essCurKwh[0] = initSoc * veh.maxEssKwh
        self.soc[0] = initSoc
  
def sim_drive(cyc , veh , initSoc=None):
    """Initialize and iterate sim_drive_sub as appropriate for vehicle attribute vehPtType."""    
    if initSoc != None:
        if initSoc>1.0 or initSoc<0.0:
            print('Must enter a valid initial SOC between 0.0 and 1.0')
            print('Running standard initial SOC controls')
            initSoc = None
   
    if veh.vehPtType == 1: # Conventional

        # If no EV / Hybrid components, no SOC considerations.

        initSoc = 0.0
        output = sim_drive_sub( cyc , veh , initSoc )

    elif veh.vehPtType == 2 and initSoc == None:  # HEV 

        #####################################
        ### Charge Balancing Vehicle SOC ###
        #####################################

        # Charge balancing SOC for PHEV vehicle types. Iterating initsoc and comparing to final SOC.
        # Iterating until tolerance met or 30 attempts made.

        initSoc = (veh.maxSoc + veh.minSoc) / 2.0
        ess2fuelKwh = 1.0
        sim_count = 0
        while ess2fuelKwh > veh.essToFuelOkError and sim_count<30:
            sim_count += 1
            output = sim_drive_sub(cyc, veh, initSoc)
            ess2fuelKwh = abs(output['ess2fuelKwh'])
            initSoc = min(1.0,max(0.0,output['final_soc']))
        np.copy(veh.maxSoc)
        output = sim_drive_sub(cyc, veh, initSoc)

    elif (veh.vehPtType == 3 and initSoc == None) or (veh.vehPtType == 4 and initSoc == None): # PHEV and BEV

        # If EV, initializing initial SOC to maximum SOC.

        initSoc = np.copy(veh.maxSoc)
        output = sim_drive_sub(cyc, veh, initSoc)
        
    else:
        output = sim_drive_sub(cyc, veh, initSoc)
        
    return output

def sim_drive_sub(cyc , veh , initSoc):
    """Function sim_drive_sub receives second-by-second cycle information,
    vehicle properties, and an initial state of charge and performs
    a backward facing powertrain simulation. The function returns an 
    output dictionary starting at approximately line 1030. Powertrain
    variables of interest (summary or time-series) can be added to the 
    output dictionary for reference."""
    
    ############################
    ###   Define Constants   ###
    ############################

    airDensityKgPerM3 = 1.2 # Sea level air density at approximately 20C
    gravityMPerSec2 = 9.81
    mphPerMps = 2.2369
    kWhPerGGE = 33.7
    metersPerMile = 1609.00
    maxTracMps2 = ((((veh.wheelCoefOfFric * veh.driveAxleWeightFrac * veh.vehKg * gravityMPerSec2)/\
        (1+((veh.vehCgM * veh.wheelCoefOfFric) / veh.wheelBaseM))))/(veh.vehKg * gravityMPerSec2)) * gravityMPerSec2
    maxRegenKwh = 0.5 * veh.vehKg*(27**2)/(3600 * 1000)

    #############################
    ### Initialize Variables  ###
    #############################

    ### Drive Cycle
    cycSecs = np.copy(cyc['cycSecs'])
    cycMps = np.copy(cyc['cycMps'])
    cycGrade = np.copy(cyc['cycGrade'])
    cycRoadType = np.copy(cyc['cycRoadType'])
    cycMph = np.copy([x * mphPerMps for x in cyc['cycMps']])
    secs = np.insert(np.diff(cycSecs), 0, 0)

    tarr = TimeArrays(cycSecs, veh, initSoc)

    ############################
    ###   Loop Through Time  ###
    ############################

    for i in range(1, len(cycSecs)):

        ### Misc calcs
        # If noElecAux, then the HV electrical system is not used to power aux loads 
        # and it must all come from the alternator.  This apparently assumes no belt-driven aux 
        # loads
        # *** 
        if veh.noElecAux == 'TRUE':
            tarr.auxInKw[i] = veh.auxKw / veh.altEff
        else:
            tarr.auxInKw[i] = veh.auxKw

        # Is SOC below min threshold?
        if tarr.soc[i-1] < (veh.minSoc + veh.percHighAccBuf):
            tarr.reachedBuff[i] = 0
        else:
            tarr.reachedBuff[i] = 1

        # Does the engine need to be on for low SOC or high acceleration
        if tarr.soc[i-1] < veh.minSoc or (tarr.highAccFcOnTag[i-1] == 1 and tarr.reachedBuff[i] == 0):
            tarr.highAccFcOnTag[i] = 1
        else:
            tarr.highAccFcOnTag[i] = 0
        tarr.maxTracMps[i] = tarr.mpsAch[i-1] + (maxTracMps2 * secs[i])

        ### Component Limits
        # max fuel storage power output
        tarr.curMaxFsKwOut[i] = min( veh.maxFuelStorKw , tarr.fsKwOutAch[i-1] + ((veh.maxFuelStorKw/veh.fuelStorSecsToPeakPwr) * (secs[i])))
        # maximum fuel storage power output rate of change
        tarr.fcTransLimKw[i] = tarr.fcKwOutAch[i-1] + ((veh.maxFuelConvKw / veh.fuelConvSecsToPeakPwr) * (secs[i]))

        tarr.fcMaxKwIn[i] = min(tarr.curMaxFsKwOut[i], veh.maxFuelStorKw) # *** this min seems redundant with line 518
        tarr.fcFsLimKw[i] = veh.fcMaxOutkW
        tarr.curMaxFcKwOut[i] = min(veh.maxFuelConvKw,tarr.fcFsLimKw[i],tarr.fcTransLimKw[i])

        # Does ESS discharge need to be limited? *** I think veh.maxEssKw should also be in the following
        # boolean condition
        if veh.maxEssKwh == 0 or tarr.soc[i-1] < veh.minSoc:
            tarr.essCapLimDischgKw[i] = 0.0

        else:
            tarr.essCapLimDischgKw[i] = (veh.maxEssKwh * np.sqrt(veh.essRoundTripEff)) * 3600.0 * (tarr.soc[i-1] - veh.minSoc) / (secs[i])
        tarr.curMaxEssKwOut[i] = min(veh.maxEssKw,tarr.essCapLimDischgKw[i])

        if  veh.maxEssKwh == 0 or veh.maxEssKw == 0:
            tarr.essCapLimChgKw[i] = 0

        else:
            tarr.essCapLimChgKw[i] = max(((veh.maxSoc - tarr.soc[i-1]) * veh.maxEssKwh * (1 / 
            np.sqrt(veh.essRoundTripEff))) / ((secs[i]) * (1 / 3600.0)), 0)

        tarr.curMaxEssChgKw[i] = min(tarr.essCapLimChgKw[i],veh.maxEssKw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if veh.fcEffType == 4:
            tarr.curMaxElecKw[i] = tarr.curMaxFcKwOut[i] + tarr.curMaxRoadwayChgKw[i] + \
                tarr.curMaxEssKwOut[i] - tarr.auxInKw[i]

        else:
            tarr.curMaxElecKw[i] = tarr.curMaxRoadwayChgKw[i] + tarr.curMaxEssKwOut[i] - tarr.auxInKw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        tarr.curMaxAvailElecKw[i] = min(tarr.curMaxElecKw[i], veh.mcMaxElecInKw)

        if tarr.curMaxElecKw[i] > 0:
            # limit power going into e-machine controller to 
            if tarr.curMaxAvailElecKw[i] == max(veh.mcKwInArray):
                tarr.mcElecInLimKw[i] = min(veh.mcKwOutArray[len(veh.mcKwOutArray) - 1],veh.maxMotorKw)
            else:
                tarr.mcElecInLimKw[i] = min(veh.mcKwOutArray[np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 
                0.01, tarr.curMaxAvailElecKw[i])) - 1],veh.maxMotorKw)
        else:
            tarr.mcElecInLimKw[i] = 0.0
        
        # Motor transient power limit
        tarr.mcTransiLimKw[i] = abs(tarr.mcMechKwOutAch[i-1]) + ((veh.maxMotorKw / veh.motorSecsToPeakPwr) * (secs[i]))
        
        tarr.curMaxMcKwOut[i] = max(min(tarr.mcElecInLimKw[i],tarr.mcTransiLimKw[i],veh.maxMotorKw),-veh.maxMotorKw)

        if tarr.curMaxMcKwOut[i] == 0:
            tarr.curMaxMcElecKwIn[i] = 0
        else:
            if tarr.curMaxMcKwOut[i] == veh.maxMotorKw:
                tarr.curMaxMcElecKwIn[i] = tarr.curMaxMcKwOut[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tarr.curMaxMcElecKwIn[i] = tarr.curMaxMcKwOut[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray 
                > min(veh.maxMotorKw - 0.01,tarr.curMaxMcKwOut[i])) - 1)]

        if veh.maxMotorKw == 0:
            tarr.essLimMcRegenPercKw[i] = 0.0

        else:
            tarr.essLimMcRegenPercKw[i] = min((tarr.curMaxEssChgKw[i] + tarr.auxInKw[i]) / veh.maxMotorKw,1)
        if tarr.curMaxEssChgKw[i] == 0:
            tarr.essLimMcRegenKw[i] = 0.0

        else:
            if veh.maxMotorKw == tarr.curMaxEssChgKw[i] - tarr.curMaxRoadwayChgKw[i]:
                tarr.essLimMcRegenKw[i] = min(veh.maxMotorKw,tarr.curMaxEssChgKw[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1])
            else:
                tarr.essLimMcRegenKw[i] = min(veh.maxMotorKw,tarr.curMaxEssChgKw[i] / veh.mcFullEffArray\
                    [max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tarr.curMaxEssChgKw[i] - tarr.curMaxRoadwayChgKw[i])) - 1)])

        tarr.curMaxMechMcKwIn[i] = min(tarr.essLimMcRegenKw[i],veh.maxMotorKw)
        tarr.curMaxTracKw[i] = (((veh.wheelCoefOfFric * veh.driveAxleWeightFrac * veh.vehKg * gravityMPerSec2)\
            / (1 + ((veh.vehCgM * veh.wheelCoefOfFric) / veh.wheelBaseM))) / 1000.0) * (tarr.maxTracMps[i])

        if veh.fcEffType == 4:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or tarr.highAccFcOnTag[i] == 1:
                tarr.curMaxTransKwOut[i] = min((tarr.curMaxMcKwOut[i] - tarr.auxInKw[i]) * veh.transEff,tarr.curMaxTracKw[i] / veh.transEff)
                tarr.debug_flag[i] = 1

            else:
                tarr.curMaxTransKwOut[i] = min((tarr.curMaxMcKwOut[i] - min(tarr.curMaxElecKw[i], 0)) * veh.transEff,tarr.curMaxTracKw[i] / veh.transEff)
                tarr.debug_flag[i] = 2

        else:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or tarr.highAccFcOnTag[i] == 1:
                tarr.curMaxTransKwOut[i] = min((tarr.curMaxMcKwOut[i] + tarr.curMaxFcKwOut[i] - \
                     tarr.auxInKw[i]) * veh.transEff,tarr.curMaxTracKw[i] / veh.transEff)
                tarr.debug_flag[i] = 3

            else:
                tarr.curMaxTransKwOut[i] = min((tarr.curMaxMcKwOut[i] + tarr.curMaxFcKwOut[i] - \
                    min(tarr.curMaxElecKw[i],0)) * veh.transEff, tarr.curMaxTracKw[i] / veh.transEff)
                tarr.debug_flag[i] = 4

        ### Cycle Power
        tarr.cycDragKw[i] = 0.5 * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * (((tarr.mpsAch[i-1] + cycMps[i]) / 2.0)**3) / 1000.0
        tarr.cycAccelKw[i] = (veh.vehKg / (2.0 * (secs[i]))) * ((cycMps[i]**2) - (tarr.mpsAch[i-1]**2)) / 1000.0
        tarr.cycAscentKw[i] = gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * ((tarr.mpsAch[i-1] + cycMps[i]) / 2.0) / 1000.0
        tarr.cycTracKwReq[i] = tarr.cycDragKw[i] + tarr.cycAccelKw[i] + tarr.cycAscentKw[i]
        tarr.spareTracKw[i] = tarr.curMaxTracKw[i] - tarr.cycTracKwReq[i]
        tarr.cycRrKw[i] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * ((tarr.mpsAch[i-1] + cycMps[i]) / 2.0) / 1000.0
        tarr.cycWheelRadPerSec[i] = cycMps[i] / veh.wheelRadiusM
        tarr.cycTireInertiaKw[i] = (((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * (tarr.cycWheelRadPerSec[i]**2.0)) / secs[i]) - \
            ((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * ((tarr.mpsAch[i-1] / veh.wheelRadiusM)**2.0)) / secs[i])) / 1000.0

        tarr.cycWheelKwReq[i] = tarr.cycTracKwReq[i] + tarr.cycRrKw[i] + tarr.cycTireInertiaKw[i]
        tarr.regenContrLimKwPerc[i] = veh.maxRegen / (1 + veh.regenA * np.exp(-veh.regenB * ((cycMph[i] + tarr.mpsAch[i-1] * mphPerMps) / 2.0 + 1 - 0)))
        tarr.cycRegenBrakeKw[i] = max(min(tarr.curMaxMechMcKwIn[i] * veh.transEff,tarr.regenContrLimKwPerc[i]*-tarr.cycWheelKwReq[i]),0)
        tarr.cycFricBrakeKw[i] = -min(tarr.cycRegenBrakeKw[i] + tarr.cycWheelKwReq[i],0)
        tarr.cycTransKwOutReq[i] = tarr.cycWheelKwReq[i] + tarr.cycFricBrakeKw[i]

        if tarr.cycTransKwOutReq[i]<=tarr.curMaxTransKwOut[i]:
            tarr.cycMet[i] = 1
            tarr.transKwOutAch[i] = tarr.cycTransKwOutReq[i]

        else:
            tarr.cycMet[i] = -1
            tarr.transKwOutAch[i] = tarr.curMaxTransKwOut[i]

        ################################
        ###   Speed/Distance Calcs   ###
        ################################

        #Cycle is met
        if tarr.cycMet[i] == 1:
            tarr.mpsAch[i] = cycMps[i]

        #Cycle is not met
        else:
            Drag3 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2
            Accel2 = veh.vehKg / (2.0 * (secs[i]))
            Drag2 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * tarr.mpsAch[i-1]
            Wheel2 = 0.5 * veh.wheelInertiaKgM2 * veh.numWheels / (secs[i] * (veh.wheelRadiusM**2))
            Drag1 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * ((tarr.mpsAch[i-1])**2)
            Roll1 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg / 2.0)
            Ascent1 = (gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg / 2.0)
            Accel0 = -(veh.vehKg * ((tarr.mpsAch[i-1])**2)) / (2.0 * (secs[i]))
            Drag0 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * ((tarr.mpsAch[i-1])**3)
            Roll0 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * tarr.mpsAch[i-1] / 2.0)
            Ascent0 = (gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * tarr.mpsAch[i-1] / 2.0)
            Wheel0 = -((0.5 * veh.wheelInertiaKgM2 * veh.numWheels * (tarr.mpsAch[i-1]**2)) / (secs[i] * (veh.wheelRadiusM**2)))

            Total3 = Drag3 / 1e3
            print(Accel2, Drag2, Wheel2)
            Total2 = (Accel2 + Drag2 + Wheel2) / 1e3
            Total1 = (Drag1 + Roll1 + Ascent1) / 1e3
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / 1e3 - tarr.curMaxTransKwOut[i]

            Total = [Total3, Total2, Total1, Total0]
            Total_roots = np.roots(Total)
            ind = np.argmin( abs(cycMps[i] - Total_roots) )
            tarr.mpsAch[i] = Total_roots[ind]

        tarr.mphAch[i] = tarr.mpsAch[i] * mphPerMps
        tarr.distMeters[i] = tarr.mpsAch[i] * secs[i]
        tarr.distMiles[i] = tarr.distMeters[i] * (1.0 / metersPerMile)

        ### Drive Train
        if tarr.transKwOutAch[i] > 0:
            tarr.transKwInAch[i] = tarr.transKwOutAch[i] / veh.transEff
        else:
            tarr.transKwInAch[i] = tarr.transKwOutAch[i] * veh.transEff

        if tarr.cycMet[i] == 1:

            if veh.fcEffType == 4:
                tarr.minMcKw2HelpFc[i] = max(tarr.transKwInAch[i], -tarr.curMaxMechMcKwIn[i])

            else:
                tarr.minMcKw2HelpFc[i] = max(tarr.transKwInAch[i] - tarr.curMaxFcKwOut[i], -tarr.curMaxMechMcKwIn[i])
        else:
            tarr.minMcKw2HelpFc[i] = max(tarr.curMaxMcKwOut[i], -tarr.curMaxMechMcKwIn[i])

        if veh.noElecSys == 'TRUE':
           tarr.regenBufferSoc[i] = 0

        elif veh.chargingOn:
           tarr.regenBufferSoc[i] = max(veh.maxSoc - (maxRegenKwh / veh.maxEssKwh), (veh.maxSoc + veh.minSoc) / 2)

        else:
           tarr.regenBufferSoc[i] = max(((veh.maxEssKwh * veh.maxSoc) - (0.5 * veh.vehKg * (cycMps[i]**2) * (1.0 / 1000) \
               * (1.0 / 3600) * veh.motorPeakEff * veh.maxRegen)) / veh.maxEssKwh,veh.minSoc)

        tarr.essRegenBufferDischgKw[i] = min(tarr.curMaxEssKwOut[i], max(0,(tarr.soc[i-1] - tarr.regenBufferSoc[i]) * veh.maxEssKwh * 3600 / secs[i]))

        tarr.maxEssRegenBufferChgKw[i] = min(max(0,(tarr.regenBufferSoc[i] - tarr.soc[i-1]) * veh.maxEssKwh * 3600.0 / secs[i]),tarr.curMaxEssChgKw[i])

        if veh.noElecSys == 'TRUE':
           tarr.accelBufferSoc[i] = 0

        else:
           tarr.accelBufferSoc[i] = min(max((((((((veh.maxAccelBufferMph * (1 / mphPerMps))**2)) - ((cycMps[i]**2))) / \
               (((veh.maxAccelBufferMph * (1 / mphPerMps))**2))) * (min(veh.maxAccelBufferPercOfUseableSoc * \
                   (veh.maxSoc - veh.minSoc),maxRegenKwh / veh.maxEssKwh) * veh.maxEssKwh)) / veh.maxEssKwh) + \
                       veh.minSoc,veh.minSoc), veh.maxSoc)

        tarr.essAccelBufferChgKw[i] = max(0,(tarr.accelBufferSoc[i] - tarr.soc[i-1]) * veh.maxEssKwh * 3600.0 / secs[i])
        tarr.maxEssAccelBufferDischgKw[i] = min(max(0, (tarr.soc[i-1] - tarr.accelBufferSoc[i]) * veh.maxEssKwh * 3600 / secs[i]),tarr.curMaxEssKwOut[i])

        if tarr.regenBufferSoc[i] < tarr.accelBufferSoc[i]:
            tarr.essAccelRegenDischgKw[i] = max(min(((tarr.soc[i-1] - (tarr.regenBufferSoc[i] + tarr.accelBufferSoc[i]) / 2) * veh.maxEssKwh * 3600.0) /\
                 secs[i],tarr.curMaxEssKwOut[i]),-tarr.curMaxEssChgKw[i])

        elif tarr.soc[i-1] > tarr.regenBufferSoc[i]:
            tarr.essAccelRegenDischgKw[i] = max(min(tarr.essRegenBufferDischgKw[i],tarr.curMaxEssKwOut[i]),-tarr.curMaxEssChgKw[i])

        elif tarr.soc[i-1] < tarr.accelBufferSoc[i]:
            tarr.essAccelRegenDischgKw[i] = max(min(-1.0 * tarr.essAccelBufferChgKw[i],tarr.curMaxEssKwOut[i]),-tarr.curMaxEssChgKw[i])

        else:
            tarr.essAccelRegenDischgKw[i] = max(min(0,tarr.curMaxEssKwOut[i]),-tarr.curMaxEssChgKw[i])

        tarr.fcKwGapFrEff[i] = abs(tarr.transKwOutAch[i] - veh.maxFcEffKw)

        if veh.noElecSys == 'TRUE':
            tarr.mcElectInKwForMaxFcEff[i] = 0

        elif tarr.transKwOutAch[i] < veh.maxFcEffKw:

            if tarr.fcKwGapFrEff[i] == veh.maxMotorKw:
                tarr.mcElectInKwForMaxFcEff[i] = tarr.fcKwGapFrEff[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]*-1
            else:
                tarr.mcElectInKwForMaxFcEff[i] = tarr.fcKwGapFrEff[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tarr.fcKwGapFrEff[i])) - 1)]*-1

        else:

            if tarr.fcKwGapFrEff[i] == veh.maxMotorKw:
                tarr.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[len(veh.mcKwInArray) - 1]
            else:
                tarr.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tarr.fcKwGapFrEff[i])) - 1]

        if veh.noElecSys == 'TRUE':
            tarr.electKwReq4AE[i] = 0

        elif tarr.transKwInAch[i] > 0:
            if tarr.transKwInAch[i] == veh.maxMotorKw:
        
                tarr.electKwReq4AE[i] = tarr.transKwInAch[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1] + tarr.auxInKw[i]
            else:
                tarr.electKwReq4AE[i] = tarr.transKwInAch[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tarr.transKwInAch[i])) - 1)] + tarr.auxInKw[i]

        else:
           tarr.electKwReq4AE[i] = 0

        tarr.prevfcTimeOn[i] = tarr.fcTimeOn[i-1]

        if veh.maxFuelConvKw == 0:
            tarr.canPowerAllElectrically[i] = tarr.accelBufferSoc[i] < tarr.soc[i-1] and tarr.transKwInAch[i]<=tarr.curMaxMcKwOut[i] and (tarr.electKwReq4AE[i] < tarr.curMaxElecKw[i] or veh.maxFuelConvKw == 0)

        else:
            tarr.canPowerAllElectrically[i] = tarr.accelBufferSoc[i] < tarr.soc[i-1] and tarr.transKwInAch[i]<=tarr.curMaxMcKwOut[i] and (tarr.electKwReq4AE[i] < tarr.curMaxElecKw[i] \
                or veh.maxFuelConvKw == 0) and (cycMph[i] - 0.00001<=veh.mphFcOn or veh.chargingOn) and tarr.electKwReq4AE[i]<=veh.kwDemandFcOn

        if tarr.canPowerAllElectrically[i]:

            if tarr.transKwInAch[i]<+tarr.auxInKw[i]:
                tarr.desiredEssKwOutForAE[i] = tarr.auxInKw[i] + tarr.transKwInAch[i]

            elif tarr.regenBufferSoc[i] < tarr.accelBufferSoc[i]:
                tarr.desiredEssKwOutForAE[i] = tarr.essAccelRegenDischgKw[i]

            elif tarr.soc[i-1] > tarr.regenBufferSoc[i]:
                tarr.desiredEssKwOutForAE[i] = tarr.essRegenBufferDischgKw[i]

            elif tarr.soc[i-1] < tarr.accelBufferSoc[i]:
                tarr.desiredEssKwOutForAE[i] = -tarr.essAccelBufferChgKw[i]

            else:
                tarr.desiredEssKwOutForAE[i] = tarr.transKwInAch[i] + tarr.auxInKw[i] - tarr.curMaxRoadwayChgKw[i]

        else:
            tarr.desiredEssKwOutForAE[i] = 0

        if tarr.canPowerAllElectrically[i]:
            tarr.essAEKwOut[i] = max(-tarr.curMaxEssChgKw[i],-tarr.maxEssRegenBufferChgKw[i],min(0,tarr.curMaxRoadwayChgKw[i] - (tarr.transKwInAch[i] + tarr.auxInKw[i])),min(tarr.curMaxEssKwOut[i],tarr.desiredEssKwOutForAE[i]))

        else:
            tarr.essAEKwOut[i] = 0

        tarr.erAEKwOut[i] = min(max(0,tarr.transKwInAch[i] + tarr.auxInKw[i] - tarr.essAEKwOut[i]),tarr.curMaxRoadwayChgKw[i])

        if tarr.prevfcTimeOn[i] > 0 and tarr.prevfcTimeOn[i] < veh.minFcTimeOn - secs[i]:
            tarr.fcForcedOn[i] = True
        else:
            tarr.fcForcedOn[i] = False

        if tarr.fcForcedOn[i] == False or tarr.canPowerAllElectrically[i] == False:
            tarr.fcForcedState[i] = 1
            tarr.mcMechKw4ForcedFc[i] = 0

        elif tarr.transKwInAch[i] < 0:
            tarr.fcForcedState[i] = 2
            tarr.mcMechKw4ForcedFc[i] = tarr.transKwInAch[i]

        elif veh.maxFcEffKw == tarr.transKwInAch[i]:
            tarr.fcForcedState[i] = 3
            tarr.mcMechKw4ForcedFc[i] = 0

        elif veh.idleFcKw > tarr.transKwInAch[i] and tarr.cycAccelKw[i] >=0:
            tarr.fcForcedState[i] = 4
            tarr.mcMechKw4ForcedFc[i] = tarr.transKwInAch[i] - veh.idleFcKw

        elif veh.maxFcEffKw > tarr.transKwInAch[i]:
            tarr.fcForcedState[i] = 5
            tarr.mcMechKw4ForcedFc[i] = 0

        else:
            tarr.fcForcedState[i] = 6
            tarr.mcMechKw4ForcedFc[i] = tarr.transKwInAch[i] - veh.maxFcEffKw

        if (-tarr.mcElectInKwForMaxFcEff[i] - tarr.curMaxRoadwayChgKw[i]) > 0:
            tarr.essDesiredKw4FcEff[i] = (-tarr.mcElectInKwForMaxFcEff[i] - tarr.curMaxRoadwayChgKw[i]) * veh.essDischgToFcMaxEffPerc

        else:
            tarr.essDesiredKw4FcEff[i] = (-tarr.mcElectInKwForMaxFcEff[i] - tarr.curMaxRoadwayChgKw[i]) * veh.essChgToFcMaxEffPerc

        if tarr.accelBufferSoc[i] > tarr.regenBufferSoc[i]:
            tarr.essKwIfFcIsReq[i] = min(tarr.curMaxEssKwOut[i],veh.mcMaxElecInKw + tarr.auxInKw[i],tarr.curMaxMcElecKwIn[i] + tarr.auxInKw[i], \
                max(-tarr.curMaxEssChgKw[i], tarr.essAccelRegenDischgKw[i]))

        elif tarr.essRegenBufferDischgKw[i] > 0:
            tarr.essKwIfFcIsReq[i] = min(tarr.curMaxEssKwOut[i],veh.mcMaxElecInKw + tarr.auxInKw[i],tarr.curMaxMcElecKwIn[i] + tarr.auxInKw[i], \
                max(-tarr.curMaxEssChgKw[i], min(tarr.essAccelRegenDischgKw[i],tarr.mcElecInLimKw[i] + tarr.auxInKw[i], max(tarr.essRegenBufferDischgKw[i],tarr.essDesiredKw4FcEff[i]))))

        elif tarr.essAccelBufferChgKw[i] > 0:
            tarr.essKwIfFcIsReq[i] = min(tarr.curMaxEssKwOut[i],veh.mcMaxElecInKw + tarr.auxInKw[i],tarr.curMaxMcElecKwIn[i] + tarr.auxInKw[i], \
                max(-tarr.curMaxEssChgKw[i], max(-1 * tarr.maxEssRegenBufferChgKw[i], min(-tarr.essAccelBufferChgKw[i],tarr.essDesiredKw4FcEff[i]))))


        elif tarr.essDesiredKw4FcEff[i] > 0:
            tarr.essKwIfFcIsReq[i] = min(tarr.curMaxEssKwOut[i],veh.mcMaxElecInKw + tarr.auxInKw[i],tarr.curMaxMcElecKwIn[i] + tarr.auxInKw[i], \
                max(-tarr.curMaxEssChgKw[i], min(tarr.essDesiredKw4FcEff[i],tarr.maxEssAccelBufferDischgKw[i])))

        else:
            tarr.essKwIfFcIsReq[i] = min(tarr.curMaxEssKwOut[i],veh.mcMaxElecInKw + tarr.auxInKw[i],tarr.curMaxMcElecKwIn[i] + tarr.auxInKw[i], \
                max(-tarr.curMaxEssChgKw[i], max(tarr.essDesiredKw4FcEff[i],-tarr.maxEssRegenBufferChgKw[i])))

        tarr.erKwIfFcIsReq[i] = max(0,min(tarr.curMaxRoadwayChgKw[i],tarr.curMaxMechMcKwIn[i],tarr.essKwIfFcIsReq[i] - tarr.mcElecInLimKw[i] + tarr.auxInKw[i]))

        tarr.mcElecKwInIfFcIsReq[i] = tarr.essKwIfFcIsReq[i] + tarr.erKwIfFcIsReq[i] - tarr.auxInKw[i]

        if veh.noElecSys == 'TRUE':
            tarr.mcKwIfFcIsReq[i] = 0

        elif  tarr.mcElecKwInIfFcIsReq[i] == 0:
            tarr.mcKwIfFcIsReq[i] = 0

        elif tarr.mcElecKwInIfFcIsReq[i] > 0:

            if tarr.mcElecKwInIfFcIsReq[i] == max(veh.mcKwInArray):
                 tarr.mcKwIfFcIsReq[i] = tarr.mcElecKwInIfFcIsReq[i] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                 tarr.mcKwIfFcIsReq[i] = tarr.mcElecKwInIfFcIsReq[i] * veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,tarr.mcElecKwInIfFcIsReq[i])) - 1)]

        else:
            if tarr.mcElecKwInIfFcIsReq[i]*-1 == max(veh.mcKwInArray):
                tarr.mcKwIfFcIsReq[i] = tarr.mcElecKwInIfFcIsReq[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tarr.mcKwIfFcIsReq[i] = tarr.mcElecKwInIfFcIsReq[i] / (veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,tarr.mcElecKwInIfFcIsReq[i]*-1)) - 1)])

        if veh.maxMotorKw == 0:
            tarr.mcMechKwOutAch[i] = 0

        elif tarr.fcForcedOn[i] == True and tarr.canPowerAllElectrically[i] == True and (veh.vehPtType == 2.0 or veh.vehPtType == 3.0) and veh.fcEffType!=4:
           tarr.mcMechKwOutAch[i] =  tarr.mcMechKw4ForcedFc[i]

        elif tarr.transKwInAch[i]<=0:

            if veh.fcEffType!=4 and veh.maxFuelConvKw> 0:
                if tarr.canPowerAllElectrically[i] == 1:
                    tarr.mcMechKwOutAch[i] = -min(tarr.curMaxMechMcKwIn[i],-tarr.transKwInAch[i])
                else:
                    tarr.mcMechKwOutAch[i] = min(-min(tarr.curMaxMechMcKwIn[i], -tarr.transKwInAch[i]),max(-tarr.curMaxFcKwOut[i], tarr.mcKwIfFcIsReq[i]))
            else:
                tarr.mcMechKwOutAch[i] = min(-min(tarr.curMaxMechMcKwIn[i],-tarr.transKwInAch[i]),-tarr.transKwInAch[i])

        elif tarr.canPowerAllElectrically[i] == 1:
            tarr.mcMechKwOutAch[i] = tarr.transKwInAch[i]

        else:
            tarr.mcMechKwOutAch[i] = max(tarr.minMcKw2HelpFc[i],tarr.mcKwIfFcIsReq[i])

        if tarr.mcMechKwOutAch[i] == 0:
            tarr.mcElecKwInAch[i] = 0.0
            tarr.motor_index_debug[i] = 0

        elif tarr.mcMechKwOutAch[i] < 0:

            if tarr.mcMechKwOutAch[i]*-1 == max(veh.mcKwInArray):
                tarr.mcElecKwInAch[i] = tarr.mcMechKwOutAch[i] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tarr.mcElecKwInAch[i] = tarr.mcMechKwOutAch[i] * veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,tarr.mcMechKwOutAch[i]*-1)) - 1)]

        else:
            if veh.maxMotorKw == tarr.mcMechKwOutAch[i]:
                tarr.mcElecKwInAch[i] = tarr.mcMechKwOutAch[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tarr.mcElecKwInAch[i] = tarr.mcMechKwOutAch[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tarr.mcMechKwOutAch[i])) - 1)]

        if tarr.curMaxRoadwayChgKw[i] == 0:
            tarr.roadwayChgKwOutAch[i] = 0

        elif veh.fcEffType == 4:
            tarr.roadwayChgKwOutAch[i] = max(0, tarr.mcElecKwInAch[i], tarr.maxEssRegenBufferChgKw[i], tarr.essRegenBufferDischgKw[i], tarr.curMaxRoadwayChgKw[i])

        elif tarr.canPowerAllElectrically[i] == 1:
            tarr.roadwayChgKwOutAch[i] = tarr.erAEKwOut[i]

        else:
            tarr.roadwayChgKwOutAch[i] = tarr.erKwIfFcIsReq[i]

        tarr.minEssKw2HelpFc[i] = tarr.mcElecKwInAch[i] + tarr.auxInKw[i] - tarr.curMaxFcKwOut[i] - tarr.roadwayChgKwOutAch[i]

        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            tarr.essKwOutAch[i]  = 0

        elif veh.fcEffType == 4:

            if tarr.transKwOutAch[i]>=0:
                tarr.essKwOutAch[i] = min(max(tarr.minEssKw2HelpFc[i],tarr.essDesiredKw4FcEff[i],tarr.essAccelRegenDischgKw[i]),tarr.curMaxEssKwOut[i],tarr.mcElecKwInAch[i] + tarr.auxInKw[i] - tarr.roadwayChgKwOutAch[i])

            else:
                tarr.essKwOutAch[i] = tarr.mcElecKwInAch[i] + tarr.auxInKw[i] - tarr.roadwayChgKwOutAch[i]

        elif tarr.highAccFcOnTag[i] == 1 or veh.noElecAux == 'TRUE':
            tarr.essKwOutAch[i] = tarr.mcElecKwInAch[i] - tarr.roadwayChgKwOutAch[i]

        else:
            tarr.essKwOutAch[i] = tarr.mcElecKwInAch[i] + tarr.auxInKw[i] - tarr.roadwayChgKwOutAch[i]

        if veh.maxFuelConvKw == 0:
            tarr.fcKwOutAch[i] = 0

        elif veh.fcEffType == 4:
            tarr.fcKwOutAch[i] = min(tarr.curMaxFcKwOut[i], max(0, tarr.mcElecKwInAch[i] + tarr.auxInKw[i] - tarr.essKwOutAch[i] - tarr.roadwayChgKwOutAch[i]))

        elif veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or tarr.highAccFcOnTag[i] == 1:
            tarr.fcKwOutAch[i] = min(tarr.curMaxFcKwOut[i], max(0, tarr.transKwInAch[i] - tarr.mcMechKwOutAch[i] + tarr.auxInKw[i]))

        else:
            tarr.fcKwOutAch[i] = min(tarr.curMaxFcKwOut[i], max(0, tarr.transKwInAch[i] - tarr.mcMechKwOutAch[i]))

        if tarr.fcKwOutAch[i] == 0:
            tarr.fcKwInAch[i] = 0.0
            tarr.fcKwOutAch_pct[i] = 0

        if veh.maxFuelConvKw == 0:
            tarr.fcKwOutAch_pct[i] = 0
        else:
            tarr.fcKwOutAch_pct[i] = tarr.fcKwOutAch[i] / veh.maxFuelConvKw

        if tarr.fcKwOutAch[i] == 0:
            tarr.fcKwInAch[i] = 0
        else:
            if tarr.fcKwOutAch[i] == veh.fcMaxOutkW:
                tarr.fcKwInAch[i] = tarr.fcKwOutAch[i] / veh.fcEffArray[len(veh.fcEffArray) - 1]
            else:
                tarr.fcKwInAch[i] = tarr.fcKwOutAch[i] / (veh.fcEffArray[max(1,np.argmax(veh.fcKwOutArray > min(tarr.fcKwOutAch[i],veh.fcMaxOutkW - 0.001)) - 1)])

        tarr.fsKwOutAch[i] = np.copy(tarr.fcKwInAch[i])

        tarr.fsKwhOutAch[i] = tarr.fsKwOutAch[i] * secs[i] * (1 / 3600.0)


        if veh.noElecSys == 'TRUE':
            tarr.essCurKwh[i] = 0

        elif tarr.essKwOutAch[i] < 0:
            tarr.essCurKwh[i] = tarr.essCurKwh[i-1] - tarr.essKwOutAch[i] * (secs[i] / 3600.0) * np.sqrt(veh.essRoundTripEff)

        else:
            tarr.essCurKwh[i] = tarr.essCurKwh[i-1] - tarr.essKwOutAch[i] * (secs[i] / 3600.0) * (1 / np.sqrt(veh.essRoundTripEff))

        if veh.maxEssKwh == 0:
            tarr.soc[i] = 0.0

        else:
            tarr.soc[i] = tarr.essCurKwh[i] / veh.maxEssKwh

        if tarr.canPowerAllElectrically[i] == True and tarr.fcForcedOn[i] == False and tarr.fcKwOutAch[i] == 0:
            tarr.fcTimeOn[i] = 0
        else:
            tarr.fcTimeOn[i] = tarr.fcTimeOn[i-1] + secs[i]

        ### Battery wear calcs

        if veh.noElecSys!='TRUE':

            if tarr.essCurKwh[i] > tarr.essCurKwh[i-1]:
                tarr.addKwh[i] = (tarr.essCurKwh[i] - tarr.essCurKwh[i-1]) + tarr.addKwh[i-1]
            else:
                tarr.addKwh[i] = 0

            if tarr.addKwh[i] == 0:
                tarr.dodCycs[i] = tarr.addKwh[i-1] / veh.maxEssKwh
            else:
                tarr.dodCycs[i] = 0

            if tarr.dodCycs[i]!=0:
                tarr.essPercDeadArray[i] = np.power(veh.essLifeCoefA,1.0 / veh.essLifeCoefB) / np.power(tarr.dodCycs[i],1.0 / veh.essLifeCoefB)
            else:
                tarr.essPercDeadArray[i] = 0

        ### Energy Audit Calculations
        tarr.dragKw[i] = 0.5 * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * (((tarr.mpsAch[i-1] + tarr.mpsAch[i]) / 2.0)**3) / 1000.0
        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            tarr.essLossKw[i] = 0
        elif tarr.essKwOutAch[i] < 0:
            tarr.essLossKw[i] = -tarr.essKwOutAch[i] - (-tarr.essKwOutAch[i] * np.sqrt(veh.essRoundTripEff))
        else:
            tarr.essLossKw[i] = tarr.essKwOutAch[i] * (1.0 / np.sqrt(veh.essRoundTripEff)) - tarr.essKwOutAch[i]
        tarr.accelKw[i] = (veh.vehKg / (2.0 * (secs[i]))) * ((tarr.mpsAch[i]**2) - (tarr.mpsAch[i-1]**2)) / 1000.0
        tarr.ascentKw[i] = gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * ((tarr.mpsAch[i-1] + tarr.mpsAch[i]) / 2.0) / 1000.0
        tarr.rrKw[i] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * ((tarr.mpsAch[i-1] + tarr.mpsAch[i]) / 2.0) / 1000.0

    ############################################
    ### Calculate Results and Assign Outputs ###
    ############################################

    output = dict()

    if sum(tarr.fsKwhOutAch) == 0:
        output['mpgge'] = 0

    else:
        output['mpgge'] = sum(tarr.distMiles) / (sum(tarr.fsKwhOutAch) * (1 / kWhPerGGE))

    roadwayChgKj = sum(tarr.roadwayChgKwOutAch * secs)
    essDischKj = -(tarr.soc[-1] - initSoc) * veh.maxEssKwh * 3600.0
    output['battery_kWh_per_mi'] = (essDischKj / 3600.0) / sum(tarr.distMiles)
    output['electric_kWh_per_mi'] = ((roadwayChgKj + essDischKj) / 3600.0) / sum(tarr.distMiles)
    output['maxTraceMissMph'] = mphPerMps * max(abs(cycMps - tarr.mpsAch))
    fuelKj = sum(np.asarray(tarr.fsKwOutAch) * np.asarray(secs))
    roadwayChgKj = sum(np.asarray(tarr.roadwayChgKwOutAch) * np.asarray(secs))
    essDischgKj = -(tarr.soc[-1] - initSoc) * veh.maxEssKwh * 3600.0

    if (fuelKj + roadwayChgKj) == 0:
        output['ess2fuelKwh'] = 1.0

    else:
        output['ess2fuelKwh'] = essDischgKj / (fuelKj + roadwayChgKj)

    output['initial_soc'] = tarr.soc[0]
    output['final_soc'] = tarr.soc[-1]


    if output['mpgge'] == 0:
        Gallons_gas_equivalent_per_mile = output['electric_kWh_per_mi'] / 33.7 # hardcoded conversion

    else:
         Gallons_gas_equivalent_per_mile = 1 / output['mpgge'] + output['electric_kWh_per_mi'] / 33.7 # hardcoded conversion

    output['mpgge_elec'] = 1 / Gallons_gas_equivalent_per_mile
    output['soc'] = np.asarray(tarr.soc)
    output['distance_mi'] = sum(tarr.distMiles)
    duration_sec = cycSecs[-1] - cycSecs[0]
    output['avg_speed_mph'] = sum(tarr.distMiles) / (duration_sec / 3600.0)
    accel = np.diff(tarr.mphAch) / np.diff(cycSecs)
    output['avg_accel_mphps'] = np.mean(accel[accel > 0])

    if max(tarr.mphAch) > 60:
        output['ZeroToSixtyTime_secs'] = np.interp(60, tarr.mphAch, cycSecs)

    else:
        output['ZeroToSixtyTime_secs'] = 0.0

    #######################################################################
    ####  Time series information for additional analysis / debugging. ####
    ####             Add parameters of interest as needed.             ####
    #######################################################################

    output['fcKwOutAch'] = np.asarray(tarr.fcKwOutAch)
    output['fsKwhOutAch'] = np.asarray(tarr.fsKwhOutAch)
    output['fcKwInAch'] = np.asarray(tarr.fcKwInAch)
    output['time'] = np.asarray(tarr.cycSecs)

    output['localvars'] = locals()

    return output
