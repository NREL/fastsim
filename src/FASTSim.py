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
import re
warnings.simplefilter('ignore')

def get_standard_cycle(cycle_name):
    """Load time trace of speed, grade, and road type."""
    csv_path = '..//cycles//'+cycle_name+'.csv'
    data = pd.read_csv(csv_path)
    return data

def get_veh(vnum):
    """Load vehicle attributes and assign to dict 'veh'."""
 
    vehdf = pd.read_csv('..//docs//FASTSim_py_veh_db.csv')
    vehdf.set_index('Selection', inplace=True, drop=False)

    ### selects specified vnum from vehdf
    veh = dict()
    for col in vehdf.columns:
        # convert all data to string types
        vehdf.loc[vnum, col] = str(vehdf.loc[vnum, col])
        # remove percent signs if any are found
        if vehdf.loc[vnum, col].find('%') != -1:
            vehdf.loc[vnum, col] = vehdf.loc[vnum, col].replace('%','')
            vehdf.loc[vnum, col] = float(vehdf.loc[vnum, col])
            vehdf.loc[vnum, col] = vehdf.loc[vnum, col] / 100.0
        # replace string for TRUE with Boolean True
        elif re.search('(?i)true', vehdf.loc[vnum, col]) != None:
            vehdf.loc[vnum, col] = True
        # replace string for FALSE with Boolean False
        elif re.search('(?i)false', vehdf.loc[vnum, col]) != None:
            vehdf.loc[vnum, col] = False
        else:
            try:
                vehdf.loc[vnum, col] = float(vehdf.loc[vnum, col])
            except:
                pass
        veh[col] = vehdf.loc[vnum, col]

    ######################################################################
    ### Append additional parameters to veh structure from calculation ###
    ######################################################################

    ### Build roadway power lookup table
    veh['MaxRoadwayChgKw_Roadway'] = range(6)
    veh['MaxRoadwayChgKw'] = [0] * len(veh['MaxRoadwayChgKw_Roadway'])
    veh['chargingOn'] = 0

     # Checking if a vehicle has any hybrid components
    if veh['maxEssKwh'] == 0 or veh['maxEssKw'] == 0 or veh['maxMotorKw'] == 0:
        veh['noElecSys'] = 'TRUE'

    else:
        veh['noElecSys'] = 'FALSE'

    # Checking if aux loads go through an alternator
    if veh['noElecSys'] == 'TRUE' or veh['maxMotorKw']<=veh['auxKw'] or veh['forceAuxOnFC'] == 'TRUE':
        veh['noElecAux'] = 'TRUE'

    else:
        veh['noElecAux'] = 'FALSE'

    veh['vehTypeSelection'] = np.copy(veh['vehPtType']) # Copying vehPtType to additional key
    # to be consistent with Excel version but not used in Python version

    ### Defining Fuel Converter efficiency curve as lookup table for %power_in vs power_out
    ### see "FC Model" tab in FASTSim for Excel

    if veh['maxFuelConvKw']>0:

        # Discrete power out percentages for assigning FC efficiencies
        fcPwrOutPerc = np.array([0, 0.005, 0.015, 0.04, 0.06, 0.10, 0.14, 0.20, 0.40, 0.60, 0.80, 1.00])

        # Efficiencies at different power out percentages by FC type
        eff_si = np.array([0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30])
        eff_atk = np.array([0.10, 0.12, 0.28, 0.35, 0.375, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35])
        eff_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])
        eff_fuel_cell = np.array([0.10, 0.30, 0.36, 0.45, 0.50, 0.56, 0.58, 0.60, 0.58, 0.57, 0.55, 0.54])
        eff_hd_diesel = np.array([0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])


        if veh['fcEffType'] == 1: # SI engine
            eff = np.copy(eff_si) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 2: # Atkinson cycle SI engine -- greater expansion
            eff = np.copy(eff_atk) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 3: # Diesel (compression ignition) engine
            eff = np.copy(eff_diesel) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 4: # H2 fuel cell
            eff = np.copy(eff_fuel_cell) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 5: # heavy duty Diesel engine
            eff = np.copy(eff_hd_diesel) + veh['fcAbsEffImpr']

        inputKwOutArray = fcPwrOutPerc * veh['maxFuelConvKw']  # discrete array of possible engine power outputs
        # Relatively continuous power out percentages for assigning FC efficiencies
        fcPercOutArray = np.r_[np.arange(0,3.0,0.1),np.arange(3.0,7.0,0.5),np.arange(7.0,60.0,1.0),np.arange(60.0,105.0,5.0)] / 100
        fcKwOutArray = veh['maxFuelConvKw'] * fcPercOutArray # Relatively continuous array of possible engine power outputs
        fcEffArray = np.array([0.0] * len(fcPercOutArray)) # Initializes relatively continuous array for fcEFF 

        # the following for loop populates fcEffArray 
        for j in range(0, len(fcPercOutArray) - 1):
            low_index = np.argmax(inputKwOutArray>=fcKwOutArray[j])
            fcinterp_x_1 = inputKwOutArray[low_index-1]
            fcinterp_x_2 = inputKwOutArray[low_index]
            fcinterp_y_1 = eff[low_index-1]
            fcinterp_y_2 = eff[low_index]
            fcEffArray[j] = (fcKwOutArray[j] - fcinterp_x_1)/(fcinterp_x_2 - fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        # populate final value 
        fcEffArray[-1] = eff[-1]

        # assign corresponding values in veh dict
        veh['fcEffArray'] = np.copy(fcEffArray)
        veh['fcKwOutArray'] = np.copy(fcKwOutArray)
        veh['maxFcEffKw'] = np.copy(veh['fcKwOutArray'][np.argmax(fcEffArray)])
        veh['fcMaxOutkW'] = np.copy(max(inputKwOutArray))
        veh['minFcTimeOn'] = 30 # hardcoded

    else:
        # these things are all zero for BEV powertrains
        # not sure why `veh['fcEffArray']` is not being assigned.  
        # Maybe it's not used anywhere in this condition.  *** delete this comment before public release
        veh['fcKwOutArray'] = np.array([0] * 101)
        veh['maxFcEffKw'] = 0
        veh['fcMaxOutkW'] = 0
        veh['minFcTimeOn'] = 30 # hardcoded

    ### Defining MC efficiency curve as lookup table for %power_in vs power_out
    ### see "Motor" tab in FASTSim for Excel
    if veh['maxMotorKw']>0:

        maxMotorKw = veh['maxMotorKw']

        mcPwrOutPerc = np.array([0.00, 0.02, 0.04, 0.06, 0.08,	0.10,	0.20,	0.40,	0.60,	0.80,	1.00])
        large_baseline_eff = np.array([0.83, 0.85,	0.87,	0.89,	0.90,	0.91,	0.93,	0.94,	0.94,	0.93,	0.92])
        small_baseline_eff = np.array([0.12,	0.16,	 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93,	0.93,	0.92])

        modern_max = 0.95
        modern_diff = modern_max - max(large_baseline_eff)

        large_baseline_eff_adj = large_baseline_eff + modern_diff

        mcKwAdjPerc = max(0.0,min((maxMotorKw - 7.5)/(75.0 - 7.5),1.0))
        mcEffArray = np.array([0.0] * len(mcPwrOutPerc))

        for k in range(0,len(mcPwrOutPerc)):
            mcEffArray[k] = mcKwAdjPerc * large_baseline_eff_adj[k] + (1 - mcKwAdjPerc)*(small_baseline_eff[k])

        mcInputKwOutArray = mcPwrOutPerc * maxMotorKw

        mcPercOutArray = np.linspace(0,1,101)
        mcKwOutArray = np.linspace(0,1,101) * maxMotorKw

        mcFullEffArray = np.array([0.0] * len(mcPercOutArray))

        for m in range(1, len(mcPercOutArray) - 1):
            low_index = np.argmax(mcInputKwOutArray>=mcKwOutArray[m])

            fcinterp_x_1 = mcInputKwOutArray[low_index-1]
            fcinterp_x_2 = mcInputKwOutArray[low_index]
            fcinterp_y_1 = mcEffArray[low_index-1]
            fcinterp_y_2 = mcEffArray[low_index]

            mcFullEffArray[m] = (mcKwOutArray[m] - fcinterp_x_1)/(fcinterp_x_2 - fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        mcFullEffArray[0] = 0
        mcFullEffArray[-1] = mcEffArray[-1]

        mcKwInArray = mcKwOutArray / mcFullEffArray
        mcKwInArray[0] = 0

        veh['mcKwInArray'] = np.copy(mcKwInArray)
        veh['mcKwOutArray'] = np.copy(mcKwOutArray)
        veh['mcMaxElecInKw'] = np.copy(max(mcKwInArray))
        veh['mcFullEffArray'] = np.copy(mcFullEffArray)
        veh['mcEffArray'] = np.copy(mcEffArray)

    else:
        veh['mcKwInArray'] = np.array([0.0] * 101)
        veh['mcKwOutArray'] = np.array([0.0]* 101)
        veh['mcMaxElecInKw'] = 0

    veh['mcMaxElecInKw'] = max(veh['mcKwInArray'])

    ### Specify shape of mc regen efficiency curve
    ### see "Regen" tab in FASTSim for Excel
    veh['regenA'] = 500.0 # hardcoded
    veh['regenB'] = 0.99 # hardcoded

    ### Calculate total vehicle mass
    # sum up component masses if positive real number is not specified for vehOverrideKg
    if not(veh['vehOverrideKg'] > 0):
        if veh['maxEssKwh'] == 0 or veh['maxEssKw'] == 0:
            ess_mass_kg = 0.0
        else:
            ess_mass_kg = ((veh['maxEssKwh'] * veh['essKgPerKwh']) + veh['essBaseKg']) * veh['compMassMultiplier']
        if veh['maxMotorKw'] == 0:
            mc_mass_kg = 0.0
        else:
            mc_mass_kg = (veh['mcPeBaseKg']+(veh['mcPeKgPerKw'] * veh['maxMotorKw'])) * veh['compMassMultiplier']
        if veh['maxFuelConvKw'] == 0:
            fc_mass_kg = 0.0
        else:
            fc_mass_kg = (((1 / veh['fuelConvKwPerKg']) * veh['maxFuelConvKw'] + veh['fuelConvBaseKg'])) * veh['compMassMultiplier']
        if veh['maxFuelStorKw'] == 0:
            fs_mass_kg = 0.0
        else:
            fs_mass_kg = ((1 / veh['fuelStorKwhPerKg']) * veh['fuelStorKwh']) * veh['compMassMultiplier']
        veh['vehKg'] = veh['cargoKg'] + veh['gliderKg'] + veh['transKg'] * veh['compMassMultiplier'] + ess_mass_kg + mc_mass_kg + fc_mass_kg + fs_mass_kg
    # if positive real number is specified for vehOverrideKg, use that
    else:
        veh['vehKg'] = np.copy(veh['vehOverrideKg'])

    # replace any spaces with underscores
    veh = dict(list(zip([key.replace(' ', '_') for key in veh.keys()], veh.values())))

    # convert veh dict to namedtuple 
    Veh = namedtuple('Veh', list(veh.keys()))
    veh = Veh(**veh)
    return veh

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

    def get_time_namedtuple():
        """Initializes arrays of time dependent variables as fields of namedtuple, 
        returned as tnt."""

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
        columns = ['cycSecs'] + comp_lim_list + \
            drivetrain_list + control_list + misc_list
        TNT = namedtuple('TNT', columns, defaults=[np.zeros(len(cycSecs))] * len(columns))
        tnt = TNT()

        tnt._replace(cycSecs=cycSecs)
        tnt._replace(fcForcedOn = np.array([False] * len(cycSecs)))
        # tnt.curMaxRoadwayChgKw = np.interp(
        #     cycRoadType, veh.MaxRoadwayChgKw_Roadway, veh.MaxRoadwayChgKw)  
            # *** this is just zeros, and I need to verify that it was zeros before and also 
            # verify that this is the correct behavior.  CB

        ###  Assign First Value  ###
        ### Drive Train
        tnt.cycMet[0] = 1
        tnt.curSocTarget[0] = veh.maxSoc
        tnt.essCurKwh[0] = initSoc * veh.maxEssKwh
        tnt.soc[0] = initSoc

        return tnt

    tnt = get_time_namedtuple()

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
            tnt.auxInKw[i] = veh.auxKw / veh.altEff
        else:
            tnt.auxInKw[i] = veh.auxKw

        # Is SOC below min threshold?
        if tnt.soc[i-1] < (veh.minSoc + veh.percHighAccBuf):
            tnt.reachedBuff[i] = 0
        else:
            tnt.reachedBuff[i] = 1

        # Does the engine need to be on for low SOC or high acceleration
        if tnt.soc[i-1] < veh.minSoc or (tnt.highAccFcOnTag[i-1] == 1 and tnt.reachedBuff[i] == 0):
            tnt.highAccFcOnTag[i] = 1
        else:
            tnt.highAccFcOnTag[i] = 0
        tnt.maxTracMps[i] = tnt.mpsAch[i-1] + (maxTracMps2 * secs[i])

        ### Component Limits
        # max fuel storage power output
        tnt.curMaxFsKwOut[i] = min( veh.maxFuelStorKw , tnt.fsKwOutAch[i-1] + ((veh.maxFuelStorKw/veh.fuelStorSecsToPeakPwr) * (secs[i])))
        # maximum fuel storage power output rate of change
        tnt.fcTransLimKw[i] = tnt.fcKwOutAch[i-1] + ((veh.maxFuelConvKw / veh.fuelConvSecsToPeakPwr) * (secs[i]))

        tnt.fcMaxKwIn[i] = min(tnt.curMaxFsKwOut[i], veh.maxFuelStorKw) # *** this min seems redundant with line 518
        tnt.fcFsLimKw[i] = veh.fcMaxOutkW
        tnt.curMaxFcKwOut[i] = min(veh.maxFuelConvKw,tnt.fcFsLimKw[i],tnt.fcTransLimKw[i])

        # Does ESS discharge need to be limited? *** I think veh.maxEssKw should also be in the following
        # boolean condition
        if veh.maxEssKwh == 0 or tnt.soc[i-1] < veh.minSoc:
            tnt.essCapLimDischgKw[i] = 0.0

        else:
            tnt.essCapLimDischgKw[i] = (veh.maxEssKwh * np.sqrt(veh.essRoundTripEff)) * 3600.0 * (tnt.soc[i-1] - veh.minSoc) / (secs[i])
        tnt.curMaxEssKwOut[i] = min(veh.maxEssKw,tnt.essCapLimDischgKw[i])

        if  veh.maxEssKwh == 0 or veh.maxEssKw == 0:
            tnt.essCapLimChgKw[i] = 0

        else:
            tnt.essCapLimChgKw[i] = max(((veh.maxSoc - tnt.soc[i-1]) * veh.maxEssKwh * (1 / 
            np.sqrt(veh.essRoundTripEff))) / ((secs[i]) * (1 / 3600.0)), 0)

        tnt.curMaxEssChgKw[i] = min(tnt.essCapLimChgKw[i],veh.maxEssKw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if veh.fcEffType == 4:
            tnt.curMaxElecKw[i] = tnt.curMaxFcKwOut[i] + tnt.curMaxRoadwayChgKw[i] + \
                tnt.curMaxEssKwOut[i] - tnt.auxInKw[i]

        else:
            tnt.curMaxElecKw[i] = tnt.curMaxRoadwayChgKw[i] + tnt.curMaxEssKwOut[i] - tnt.auxInKw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        tnt.curMaxAvailElecKw[i] = min(tnt.curMaxElecKw[i], veh.mcMaxElecInKw)

        if tnt.curMaxElecKw[i] > 0:
            # limit power going into e-machine controller to 
            if tnt.curMaxAvailElecKw[i] == max(veh.mcKwInArray):
                tnt.mcElecInLimKw[i] = min(veh.mcKwOutArray[len(veh.mcKwOutArray) - 1],veh.maxMotorKw)
            else:
                tnt.mcElecInLimKw[i] = min(veh.mcKwOutArray[np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 
                0.01, tnt.curMaxAvailElecKw[i])) - 1],veh.maxMotorKw)
        else:
            tnt.mcElecInLimKw[i] = 0.0
        
        # Motor transient power limit
        tnt.mcTransiLimKw[i] = abs(tnt.mcMechKwOutAch[i-1]) + ((veh.maxMotorKw / veh.motorSecsToPeakPwr) * (secs[i]))
        
        tnt.curMaxMcKwOut[i] = max(min(tnt.mcElecInLimKw[i],tnt.mcTransiLimKw[i],veh.maxMotorKw),-veh.maxMotorKw)

        if tnt.curMaxMcKwOut[i] == 0:
            tnt.curMaxMcElecKwIn[i] = 0
        else:
            if tnt.curMaxMcKwOut[i] == veh.maxMotorKw:
                tnt.curMaxMcElecKwIn[i] = tnt.curMaxMcKwOut[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tnt.curMaxMcElecKwIn[i] = tnt.curMaxMcKwOut[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray 
                > min(veh.maxMotorKw - 0.01,tnt.curMaxMcKwOut[i])) - 1)]

        if veh.maxMotorKw == 0:
            tnt.essLimMcRegenPercKw[i] = 0.0

        else:
            tnt.essLimMcRegenPercKw[i] = min((tnt.curMaxEssChgKw[i] + tnt.auxInKw[i]) / veh.maxMotorKw,1)
        if tnt.curMaxEssChgKw[i] == 0:
            tnt.essLimMcRegenKw[i] = 0.0

        else:
            if veh.maxMotorKw == tnt.curMaxEssChgKw[i] - tnt.curMaxRoadwayChgKw[i]:
                tnt.essLimMcRegenKw[i] = min(veh.maxMotorKw,tnt.curMaxEssChgKw[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1])
            else:
                tnt.essLimMcRegenKw[i] = min(veh.maxMotorKw,tnt.curMaxEssChgKw[i] / veh.mcFullEffArray\
                    [max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tnt.curMaxEssChgKw[i] - tnt.curMaxRoadwayChgKw[i])) - 1)])

        tnt.curMaxMechMcKwIn[i] = min(tnt.essLimMcRegenKw[i],veh.maxMotorKw)
        tnt.curMaxTracKw[i] = (((veh.wheelCoefOfFric * veh.driveAxleWeightFrac * veh.vehKg * gravityMPerSec2)\
            / (1 + ((veh.vehCgM * veh.wheelCoefOfFric) / veh.wheelBaseM))) / 1000.0) * (tnt.maxTracMps[i])

        if veh.fcEffType == 4:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or tnt.highAccFcOnTag[i] == 1:
                tnt.curMaxTransKwOut[i] = min((tnt.curMaxMcKwOut[i] - tnt.auxInKw[i]) * veh.transEff,tnt.curMaxTracKw[i] / veh.transEff)
                tnt.debug_flag[i] = 1

            else:
                tnt.curMaxTransKwOut[i] = min((tnt.curMaxMcKwOut[i] - min(tnt.curMaxElecKw[i], 0)) * veh.transEff,tnt.curMaxTracKw[i] / veh.transEff)
                tnt.debug_flag[i] = 2

        else:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or tnt.highAccFcOnTag[i] == 1:
                tnt.curMaxTransKwOut[i] = min((tnt.curMaxMcKwOut[i] + tnt.curMaxFcKwOut[i] - \
                     tnt.auxInKw[i]) * veh.transEff,tnt.curMaxTracKw[i] / veh.transEff)
                tnt.debug_flag[i] = 3

            else:
                tnt.curMaxTransKwOut[i] = min((tnt.curMaxMcKwOut[i] + tnt.curMaxFcKwOut[i] - \
                    min(tnt.curMaxElecKw[i],0)) * veh.transEff, tnt.curMaxTracKw[i] / veh.transEff)
                tnt.debug_flag[i] = 4

        ### Cycle Power
        tnt.cycDragKw[i] = 0.5 * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * (((tnt.mpsAch[i-1] + cycMps[i]) / 2.0)**3) / 1000.0
        tnt.cycAccelKw[i] = (veh.vehKg / (2.0 * (secs[i]))) * ((cycMps[i]**2) - (tnt.mpsAch[i-1]**2)) / 1000.0
        tnt.cycAscentKw[i] = gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * ((tnt.mpsAch[i-1] + cycMps[i]) / 2.0) / 1000.0
        tnt.cycTracKwReq[i] = tnt.cycDragKw[i] + tnt.cycAccelKw[i] + tnt.cycAscentKw[i]
        tnt.spareTracKw[i] = tnt.curMaxTracKw[i] - tnt.cycTracKwReq[i]
        tnt.cycRrKw[i] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * ((tnt.mpsAch[i-1] + cycMps[i]) / 2.0) / 1000.0
        tnt.cycWheelRadPerSec[i] = cycMps[i] / veh.wheelRadiusM
        tnt.cycTireInertiaKw[i] = (((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * (tnt.cycWheelRadPerSec[i]**2.0)) / secs[i]) - \
            ((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * ((tnt.mpsAch[i-1] / veh.wheelRadiusM)**2.0)) / secs[i])) / 1000.0

        tnt.cycWheelKwReq[i] = tnt.cycTracKwReq[i] + tnt.cycRrKw[i] + tnt.cycTireInertiaKw[i]
        tnt.regenContrLimKwPerc[i] = veh.maxRegen / (1 + veh.regenA * np.exp(-veh.regenB * ((cycMph[i] + tnt.mpsAch[i-1] * mphPerMps) / 2.0 + 1 - 0)))
        tnt.cycRegenBrakeKw[i] = max(min(tnt.curMaxMechMcKwIn[i] * veh.transEff,tnt.regenContrLimKwPerc[i]*-tnt.cycWheelKwReq[i]),0)
        tnt.cycFricBrakeKw[i] = -min(tnt.cycRegenBrakeKw[i] + tnt.cycWheelKwReq[i],0)
        tnt.cycTransKwOutReq[i] = tnt.cycWheelKwReq[i] + tnt.cycFricBrakeKw[i]

        if tnt.cycTransKwOutReq[i]<=tnt.curMaxTransKwOut[i]:
            tnt.cycMet[i] = 1
            tnt.transKwOutAch[i] = tnt.cycTransKwOutReq[i]

        else:
            tnt.cycMet[i] = -1
            tnt.transKwOutAch[i] = tnt.curMaxTransKwOut[i]

        ################################
        ###   Speed/Distance Calcs   ###
        ################################

        #Cycle is met
        if tnt.cycMet[i] == 1:
            tnt.mpsAch[i] = cycMps[i]

        #Cycle is not met
        else:
            Drag3 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2
            Accel2 = veh.vehKg / (2.0 * (secs[i]))
            Drag2 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * tnt.mpsAch[i-1]
            Wheel2 = 0.5 * veh.wheelInertiaKgM2 * veh.numWheels / (secs[i] * (veh.wheelRadiusM**2))
            Drag1 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * ((tnt.mpsAch[i-1])**2)
            Roll1 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg / 2.0)
            Ascent1 = (gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg / 2.0)
            Accel0 = -(veh.vehKg * ((tnt.mpsAch[i-1])**2)) / (2.0 * (secs[i]))
            Drag0 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * ((tnt.mpsAch[i-1])**3)
            Roll0 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * tnt.mpsAch[i-1] / 2.0)
            Ascent0 = (gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * tnt.mpsAch[i-1] / 2.0)
            Wheel0 = -((0.5 * veh.wheelInertiaKgM2 * veh.numWheels * (tnt.mpsAch[i-1]**2)) / (secs[i] * (veh.wheelRadiusM**2)))

            Total3 = Drag3 / 1e3
            print(Accel2, Drag2, Wheel2)
            Total2 = (Accel2 + Drag2 + Wheel2) / 1e3
            Total1 = (Drag1 + Roll1 + Ascent1) / 1e3
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / 1e3 - tnt.curMaxTransKwOut[i]

            Total = [Total3, Total2, Total1, Total0]
            Total_roots = np.roots(Total)
            ind = np.argmin( abs(cycMps[i] - Total_roots) )
            tnt.mpsAch[i] = Total_roots[ind]

        tnt.mphAch[i] = tnt.mpsAch[i] * mphPerMps
        tnt.distMeters[i] = tnt.mpsAch[i] * secs[i]
        tnt.distMiles[i] = tnt.distMeters[i] * (1.0 / metersPerMile)

        ### Drive Train
        if tnt.transKwOutAch[i] > 0:
            tnt.transKwInAch[i] = tnt.transKwOutAch[i] / veh.transEff
        else:
            tnt.transKwInAch[i] = tnt.transKwOutAch[i] * veh.transEff

        if tnt.cycMet[i] == 1:

            if veh.fcEffType == 4:
                tnt.minMcKw2HelpFc[i] = max(tnt.transKwInAch[i], -tnt.curMaxMechMcKwIn[i])

            else:
                tnt.minMcKw2HelpFc[i] = max(tnt.transKwInAch[i] - tnt.curMaxFcKwOut[i], -tnt.curMaxMechMcKwIn[i])
        else:
            tnt.minMcKw2HelpFc[i] = max(tnt.curMaxMcKwOut[i], -tnt.curMaxMechMcKwIn[i])

        if veh.noElecSys == 'TRUE':
           tnt.regenBufferSoc[i] = 0

        elif veh.chargingOn:
           tnt.regenBufferSoc[i] = max(veh.maxSoc - (maxRegenKwh / veh.maxEssKwh), (veh.maxSoc + veh.minSoc) / 2)

        else:
           tnt.regenBufferSoc[i] = max(((veh.maxEssKwh * veh.maxSoc) - (0.5 * veh.vehKg * (cycMps[i]**2) * (1.0 / 1000) \
               * (1.0 / 3600) * veh.motorPeakEff * veh.maxRegen)) / veh.maxEssKwh,veh.minSoc)

        tnt.essRegenBufferDischgKw[i] = min(tnt.curMaxEssKwOut[i], max(0,(tnt.soc[i-1] - tnt.regenBufferSoc[i]) * veh.maxEssKwh * 3600 / secs[i]))

        tnt.maxEssRegenBufferChgKw[i] = min(max(0,(tnt.regenBufferSoc[i] - tnt.soc[i-1]) * veh.maxEssKwh * 3600.0 / secs[i]),tnt.curMaxEssChgKw[i])

        if veh.noElecSys == 'TRUE':
           tnt.accelBufferSoc[i] = 0

        else:
           tnt.accelBufferSoc[i] = min(max((((((((veh.maxAccelBufferMph * (1 / mphPerMps))**2)) - ((cycMps[i]**2))) / \
               (((veh.maxAccelBufferMph * (1 / mphPerMps))**2))) * (min(veh.maxAccelBufferPercOfUseableSoc * \
                   (veh.maxSoc - veh.minSoc),maxRegenKwh / veh.maxEssKwh) * veh.maxEssKwh)) / veh.maxEssKwh) + \
                       veh.minSoc,veh.minSoc), veh.maxSoc)

        tnt.essAccelBufferChgKw[i] = max(0,(tnt.accelBufferSoc[i] - tnt.soc[i-1]) * veh.maxEssKwh * 3600.0 / secs[i])
        tnt.maxEssAccelBufferDischgKw[i] = min(max(0, (tnt.soc[i-1] - tnt.accelBufferSoc[i]) * veh.maxEssKwh * 3600 / secs[i]),tnt.curMaxEssKwOut[i])

        if tnt.regenBufferSoc[i] < tnt.accelBufferSoc[i]:
            tnt.essAccelRegenDischgKw[i] = max(min(((tnt.soc[i-1] - (tnt.regenBufferSoc[i] + tnt.accelBufferSoc[i]) / 2) * veh.maxEssKwh * 3600.0) /\
                 secs[i],tnt.curMaxEssKwOut[i]),-tnt.curMaxEssChgKw[i])

        elif tnt.soc[i-1] > tnt.regenBufferSoc[i]:
            tnt.essAccelRegenDischgKw[i] = max(min(tnt.essRegenBufferDischgKw[i],tnt.curMaxEssKwOut[i]),-tnt.curMaxEssChgKw[i])

        elif tnt.soc[i-1] < tnt.accelBufferSoc[i]:
            tnt.essAccelRegenDischgKw[i] = max(min(-1.0 * tnt.essAccelBufferChgKw[i],tnt.curMaxEssKwOut[i]),-tnt.curMaxEssChgKw[i])

        else:
            tnt.essAccelRegenDischgKw[i] = max(min(0,tnt.curMaxEssKwOut[i]),-tnt.curMaxEssChgKw[i])

        tnt.fcKwGapFrEff[i] = abs(tnt.transKwOutAch[i] - veh.maxFcEffKw)

        if veh.noElecSys == 'TRUE':
            tnt.mcElectInKwForMaxFcEff[i] = 0

        elif tnt.transKwOutAch[i] < veh.maxFcEffKw:

            if tnt.fcKwGapFrEff[i] == veh.maxMotorKw:
                tnt.mcElectInKwForMaxFcEff[i] = tnt.fcKwGapFrEff[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]*-1
            else:
                tnt.mcElectInKwForMaxFcEff[i] = tnt.fcKwGapFrEff[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tnt.fcKwGapFrEff[i])) - 1)]*-1

        else:

            if tnt.fcKwGapFrEff[i] == veh.maxMotorKw:
                tnt.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[len(veh.mcKwInArray) - 1]
            else:
                tnt.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tnt.fcKwGapFrEff[i])) - 1]

        if veh.noElecSys == 'TRUE':
            tnt.electKwReq4AE[i] = 0

        elif tnt.transKwInAch[i] > 0:
            if tnt.transKwInAch[i] == veh.maxMotorKw:
        
                tnt.electKwReq4AE[i] = tnt.transKwInAch[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1] + tnt.auxInKw[i]
            else:
                tnt.electKwReq4AE[i] = tnt.transKwInAch[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tnt.transKwInAch[i])) - 1)] + tnt.auxInKw[i]

        else:
           tnt.electKwReq4AE[i] = 0

        tnt.prevfcTimeOn[i] = tnt.fcTimeOn[i-1]

        if veh.maxFuelConvKw == 0:
            tnt.canPowerAllElectrically[i] = tnt.accelBufferSoc[i] < tnt.soc[i-1] and tnt.transKwInAch[i]<=tnt.curMaxMcKwOut[i] and (tnt.electKwReq4AE[i] < tnt.curMaxElecKw[i] or veh.maxFuelConvKw == 0)

        else:
            tnt.canPowerAllElectrically[i] = tnt.accelBufferSoc[i] < tnt.soc[i-1] and tnt.transKwInAch[i]<=tnt.curMaxMcKwOut[i] and (tnt.electKwReq4AE[i] < tnt.curMaxElecKw[i] \
                or veh.maxFuelConvKw == 0) and (cycMph[i] - 0.00001<=veh.mphFcOn or veh.chargingOn) and tnt.electKwReq4AE[i]<=veh.kwDemandFcOn

        if tnt.canPowerAllElectrically[i]:

            if tnt.transKwInAch[i]<+tnt.auxInKw[i]:
                tnt.desiredEssKwOutForAE[i] = tnt.auxInKw[i] + tnt.transKwInAch[i]

            elif tnt.regenBufferSoc[i] < tnt.accelBufferSoc[i]:
                tnt.desiredEssKwOutForAE[i] = tnt.essAccelRegenDischgKw[i]

            elif tnt.soc[i-1] > tnt.regenBufferSoc[i]:
                tnt.desiredEssKwOutForAE[i] = tnt.essRegenBufferDischgKw[i]

            elif tnt.soc[i-1] < tnt.accelBufferSoc[i]:
                tnt.desiredEssKwOutForAE[i] = -tnt.essAccelBufferChgKw[i]

            else:
                tnt.desiredEssKwOutForAE[i] = tnt.transKwInAch[i] + tnt.auxInKw[i] - tnt.curMaxRoadwayChgKw[i]

        else:
            tnt.desiredEssKwOutForAE[i] = 0

        if tnt.canPowerAllElectrically[i]:
            tnt.essAEKwOut[i] = max(-tnt.curMaxEssChgKw[i],-tnt.maxEssRegenBufferChgKw[i],min(0,tnt.curMaxRoadwayChgKw[i] - (tnt.transKwInAch[i] + tnt.auxInKw[i])),min(tnt.curMaxEssKwOut[i],tnt.desiredEssKwOutForAE[i]))

        else:
            tnt.essAEKwOut[i] = 0

        tnt.erAEKwOut[i] = min(max(0,tnt.transKwInAch[i] + tnt.auxInKw[i] - tnt.essAEKwOut[i]),tnt.curMaxRoadwayChgKw[i])

        if tnt.prevfcTimeOn[i] > 0 and tnt.prevfcTimeOn[i] < veh.minFcTimeOn - secs[i]:
            tnt.fcForcedOn[i] = True
        else:
            tnt.fcForcedOn[i] = False

        if tnt.fcForcedOn[i] == False or tnt.canPowerAllElectrically[i] == False:
            tnt.fcForcedState[i] = 1
            tnt.mcMechKw4ForcedFc[i] = 0

        elif tnt.transKwInAch[i] < 0:
            tnt.fcForcedState[i] = 2
            tnt.mcMechKw4ForcedFc[i] = tnt.transKwInAch[i]

        elif veh.maxFcEffKw == tnt.transKwInAch[i]:
            tnt.fcForcedState[i] = 3
            tnt.mcMechKw4ForcedFc[i] = 0

        elif veh.idleFcKw > tnt.transKwInAch[i] and tnt.cycAccelKw[i] >=0:
            tnt.fcForcedState[i] = 4
            tnt.mcMechKw4ForcedFc[i] = tnt.transKwInAch[i] - veh.idleFcKw

        elif veh.maxFcEffKw > tnt.transKwInAch[i]:
            tnt.fcForcedState[i] = 5
            tnt.mcMechKw4ForcedFc[i] = 0

        else:
            tnt.fcForcedState[i] = 6
            tnt.mcMechKw4ForcedFc[i] = tnt.transKwInAch[i] - veh.maxFcEffKw

        if (-tnt.mcElectInKwForMaxFcEff[i] - tnt.curMaxRoadwayChgKw[i]) > 0:
            tnt.essDesiredKw4FcEff[i] = (-tnt.mcElectInKwForMaxFcEff[i] - tnt.curMaxRoadwayChgKw[i]) * veh.essDischgToFcMaxEffPerc

        else:
            tnt.essDesiredKw4FcEff[i] = (-tnt.mcElectInKwForMaxFcEff[i] - tnt.curMaxRoadwayChgKw[i]) * veh.essChgToFcMaxEffPerc

        if tnt.accelBufferSoc[i] > tnt.regenBufferSoc[i]:
            tnt.essKwIfFcIsReq[i] = min(tnt.curMaxEssKwOut[i],veh.mcMaxElecInKw + tnt.auxInKw[i],tnt.curMaxMcElecKwIn[i] + tnt.auxInKw[i], \
                max(-tnt.curMaxEssChgKw[i], tnt.essAccelRegenDischgKw[i]))

        elif tnt.essRegenBufferDischgKw[i] > 0:
            tnt.essKwIfFcIsReq[i] = min(tnt.curMaxEssKwOut[i],veh.mcMaxElecInKw + tnt.auxInKw[i],tnt.curMaxMcElecKwIn[i] + tnt.auxInKw[i], \
                max(-tnt.curMaxEssChgKw[i], min(tnt.essAccelRegenDischgKw[i],tnt.mcElecInLimKw[i] + tnt.auxInKw[i], max(tnt.essRegenBufferDischgKw[i],tnt.essDesiredKw4FcEff[i]))))

        elif tnt.essAccelBufferChgKw[i] > 0:
            tnt.essKwIfFcIsReq[i] = min(tnt.curMaxEssKwOut[i],veh.mcMaxElecInKw + tnt.auxInKw[i],tnt.curMaxMcElecKwIn[i] + tnt.auxInKw[i], \
                max(-tnt.curMaxEssChgKw[i], max(-1 * tnt.maxEssRegenBufferChgKw[i], min(-tnt.essAccelBufferChgKw[i],tnt.essDesiredKw4FcEff[i]))))


        elif tnt.essDesiredKw4FcEff[i] > 0:
            tnt.essKwIfFcIsReq[i] = min(tnt.curMaxEssKwOut[i],veh.mcMaxElecInKw + tnt.auxInKw[i],tnt.curMaxMcElecKwIn[i] + tnt.auxInKw[i], \
                max(-tnt.curMaxEssChgKw[i], min(tnt.essDesiredKw4FcEff[i],tnt.maxEssAccelBufferDischgKw[i])))

        else:
            tnt.essKwIfFcIsReq[i] = min(tnt.curMaxEssKwOut[i],veh.mcMaxElecInKw + tnt.auxInKw[i],tnt.curMaxMcElecKwIn[i] + tnt.auxInKw[i], \
                max(-tnt.curMaxEssChgKw[i], max(tnt.essDesiredKw4FcEff[i],-tnt.maxEssRegenBufferChgKw[i])))

        tnt.erKwIfFcIsReq[i] = max(0,min(tnt.curMaxRoadwayChgKw[i],tnt.curMaxMechMcKwIn[i],tnt.essKwIfFcIsReq[i] - tnt.mcElecInLimKw[i] + tnt.auxInKw[i]))

        tnt.mcElecKwInIfFcIsReq[i] = tnt.essKwIfFcIsReq[i] + tnt.erKwIfFcIsReq[i] - tnt.auxInKw[i]

        if veh.noElecSys == 'TRUE':
            tnt.mcKwIfFcIsReq[i] = 0

        elif  tnt.mcElecKwInIfFcIsReq[i] == 0:
            tnt.mcKwIfFcIsReq[i] = 0

        elif tnt.mcElecKwInIfFcIsReq[i] > 0:

            if tnt.mcElecKwInIfFcIsReq[i] == max(veh.mcKwInArray):
                 tnt.mcKwIfFcIsReq[i] = tnt.mcElecKwInIfFcIsReq[i] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                 tnt.mcKwIfFcIsReq[i] = tnt.mcElecKwInIfFcIsReq[i] * veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,tnt.mcElecKwInIfFcIsReq[i])) - 1)]

        else:
            if tnt.mcElecKwInIfFcIsReq[i]*-1 == max(veh.mcKwInArray):
                tnt.mcKwIfFcIsReq[i] = tnt.mcElecKwInIfFcIsReq[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tnt.mcKwIfFcIsReq[i] = tnt.mcElecKwInIfFcIsReq[i] / (veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,tnt.mcElecKwInIfFcIsReq[i]*-1)) - 1)])

        if veh.maxMotorKw == 0:
            tnt.mcMechKwOutAch[i] = 0

        elif tnt.fcForcedOn[i] == True and tnt.canPowerAllElectrically[i] == True and (veh.vehPtType == 2.0 or veh.vehPtType == 3.0) and veh.fcEffType!=4:
           tnt.mcMechKwOutAch[i] =  tnt.mcMechKw4ForcedFc[i]

        elif tnt.transKwInAch[i]<=0:

            if veh.fcEffType!=4 and veh.maxFuelConvKw> 0:
                if tnt.canPowerAllElectrically[i] == 1:
                    tnt.mcMechKwOutAch[i] = -min(tnt.curMaxMechMcKwIn[i],-tnt.transKwInAch[i])
                else:
                    tnt.mcMechKwOutAch[i] = min(-min(tnt.curMaxMechMcKwIn[i], -tnt.transKwInAch[i]),max(-tnt.curMaxFcKwOut[i], tnt.mcKwIfFcIsReq[i]))
            else:
                tnt.mcMechKwOutAch[i] = min(-min(tnt.curMaxMechMcKwIn[i],-tnt.transKwInAch[i]),-tnt.transKwInAch[i])

        elif tnt.canPowerAllElectrically[i] == 1:
            tnt.mcMechKwOutAch[i] = tnt.transKwInAch[i]

        else:
            tnt.mcMechKwOutAch[i] = max(tnt.minMcKw2HelpFc[i],tnt.mcKwIfFcIsReq[i])

        if tnt.mcMechKwOutAch[i] == 0:
            tnt.mcElecKwInAch[i] = 0.0
            tnt.motor_index_debug[i] = 0

        elif tnt.mcMechKwOutAch[i] < 0:

            if tnt.mcMechKwOutAch[i]*-1 == max(veh.mcKwInArray):
                tnt.mcElecKwInAch[i] = tnt.mcMechKwOutAch[i] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tnt.mcElecKwInAch[i] = tnt.mcMechKwOutAch[i] * veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,tnt.mcMechKwOutAch[i]*-1)) - 1)]

        else:
            if veh.maxMotorKw == tnt.mcMechKwOutAch[i]:
                tnt.mcElecKwInAch[i] = tnt.mcMechKwOutAch[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                tnt.mcElecKwInAch[i] = tnt.mcMechKwOutAch[i] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,tnt.mcMechKwOutAch[i])) - 1)]

        if tnt.curMaxRoadwayChgKw[i] == 0:
            tnt.roadwayChgKwOutAch[i] = 0

        elif veh.fcEffType == 4:
            tnt.roadwayChgKwOutAch[i] = max(0, tnt.mcElecKwInAch[i], tnt.maxEssRegenBufferChgKw[i], tnt.essRegenBufferDischgKw[i], tnt.curMaxRoadwayChgKw[i])

        elif tnt.canPowerAllElectrically[i] == 1:
            tnt.roadwayChgKwOutAch[i] = tnt.erAEKwOut[i]

        else:
            tnt.roadwayChgKwOutAch[i] = tnt.erKwIfFcIsReq[i]

        tnt.minEssKw2HelpFc[i] = tnt.mcElecKwInAch[i] + tnt.auxInKw[i] - tnt.curMaxFcKwOut[i] - tnt.roadwayChgKwOutAch[i]

        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            tnt.essKwOutAch[i]  = 0

        elif veh.fcEffType == 4:

            if tnt.transKwOutAch[i]>=0:
                tnt.essKwOutAch[i] = min(max(tnt.minEssKw2HelpFc[i],tnt.essDesiredKw4FcEff[i],tnt.essAccelRegenDischgKw[i]),tnt.curMaxEssKwOut[i],tnt.mcElecKwInAch[i] + tnt.auxInKw[i] - tnt.roadwayChgKwOutAch[i])

            else:
                tnt.essKwOutAch[i] = tnt.mcElecKwInAch[i] + tnt.auxInKw[i] - tnt.roadwayChgKwOutAch[i]

        elif tnt.highAccFcOnTag[i] == 1 or veh.noElecAux == 'TRUE':
            tnt.essKwOutAch[i] = tnt.mcElecKwInAch[i] - tnt.roadwayChgKwOutAch[i]

        else:
            tnt.essKwOutAch[i] = tnt.mcElecKwInAch[i] + tnt.auxInKw[i] - tnt.roadwayChgKwOutAch[i]

        if veh.maxFuelConvKw == 0:
            tnt.fcKwOutAch[i] = 0

        elif veh.fcEffType == 4:
            tnt.fcKwOutAch[i] = min(tnt.curMaxFcKwOut[i], max(0, tnt.mcElecKwInAch[i] + tnt.auxInKw[i] - tnt.essKwOutAch[i] - tnt.roadwayChgKwOutAch[i]))

        elif veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or tnt.highAccFcOnTag[i] == 1:
            tnt.fcKwOutAch[i] = min(tnt.curMaxFcKwOut[i], max(0, tnt.transKwInAch[i] - tnt.mcMechKwOutAch[i] + tnt.auxInKw[i]))

        else:
            tnt.fcKwOutAch[i] = min(tnt.curMaxFcKwOut[i], max(0, tnt.transKwInAch[i] - tnt.mcMechKwOutAch[i]))

        if tnt.fcKwOutAch[i] == 0:
            tnt.fcKwInAch[i] = 0.0
            tnt.fcKwOutAch_pct[i] = 0

        if veh.maxFuelConvKw == 0:
            tnt.fcKwOutAch_pct[i] = 0
        else:
            tnt.fcKwOutAch_pct[i] = tnt.fcKwOutAch[i] / veh.maxFuelConvKw

        if tnt.fcKwOutAch[i] == 0:
            tnt.fcKwInAch[i] = 0
        else:
            if tnt.fcKwOutAch[i] == veh.fcMaxOutkW:
                tnt.fcKwInAch[i] = tnt.fcKwOutAch[i] / veh.fcEffArray[len(veh.fcEffArray) - 1]
            else:
                tnt.fcKwInAch[i] = tnt.fcKwOutAch[i] / (veh.fcEffArray[max(1,np.argmax(veh.fcKwOutArray > min(tnt.fcKwOutAch[i],veh.fcMaxOutkW - 0.001)) - 1)])

        tnt.fsKwOutAch[i] = np.copy(tnt.fcKwInAch[i])

        tnt.fsKwhOutAch[i] = tnt.fsKwOutAch[i] * secs[i] * (1 / 3600.0)


        if veh.noElecSys == 'TRUE':
            tnt.essCurKwh[i] = 0

        elif tnt.essKwOutAch[i] < 0:
            tnt.essCurKwh[i] = tnt.essCurKwh[i-1] - tnt.essKwOutAch[i] * (secs[i] / 3600.0) * np.sqrt(veh.essRoundTripEff)

        else:
            tnt.essCurKwh[i] = tnt.essCurKwh[i-1] - tnt.essKwOutAch[i] * (secs[i] / 3600.0) * (1 / np.sqrt(veh.essRoundTripEff))

        if veh.maxEssKwh == 0:
            tnt.soc[i] = 0.0

        else:
            tnt.soc[i] = tnt.essCurKwh[i] / veh.maxEssKwh

        if tnt.canPowerAllElectrically[i] == True and tnt.fcForcedOn[i] == False and tnt.fcKwOutAch[i] == 0:
            tnt.fcTimeOn[i] = 0
        else:
            tnt.fcTimeOn[i] = tnt.fcTimeOn[i-1] + secs[i]

        ### Battery wear calcs

        if veh.noElecSys!='TRUE':

            if tnt.essCurKwh[i] > tnt.essCurKwh[i-1]:
                tnt.addKwh[i] = (tnt.essCurKwh[i] - tnt.essCurKwh[i-1]) + tnt.addKwh[i-1]
            else:
                tnt.addKwh[i] = 0

            if tnt.addKwh[i] == 0:
                tnt.dodCycs[i] = tnt.addKwh[i-1] / veh.maxEssKwh
            else:
                tnt.dodCycs[i] = 0

            if tnt.dodCycs[i]!=0:
                tnt.essPercDeadArray[i] = np.power(veh.essLifeCoefA,1.0 / veh.essLifeCoefB) / np.power(tnt.dodCycs[i],1.0 / veh.essLifeCoefB)
            else:
                tnt.essPercDeadArray[i] = 0

        ### Energy Audit Calculations
        tnt.dragKw[i] = 0.5 * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * (((tnt.mpsAch[i-1] + tnt.mpsAch[i]) / 2.0)**3) / 1000.0
        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            tnt.essLossKw[i] = 0
        elif tnt.essKwOutAch[i] < 0:
            tnt.essLossKw[i] = -tnt.essKwOutAch[i] - (-tnt.essKwOutAch[i] * np.sqrt(veh.essRoundTripEff))
        else:
            tnt.essLossKw[i] = tnt.essKwOutAch[i] * (1.0 / np.sqrt(veh.essRoundTripEff)) - tnt.essKwOutAch[i]
        tnt.accelKw[i] = (veh.vehKg / (2.0 * (secs[i]))) * ((tnt.mpsAch[i]**2) - (tnt.mpsAch[i-1]**2)) / 1000.0
        tnt.ascentKw[i] = gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * ((tnt.mpsAch[i-1] + tnt.mpsAch[i]) / 2.0) / 1000.0
        tnt.rrKw[i] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * ((tnt.mpsAch[i-1] + tnt.mpsAch[i]) / 2.0) / 1000.0

    ############################################
    ### Calculate Results and Assign Outputs ###
    ############################################

    output = dict()

    if sum(tnt.fsKwhOutAch) == 0:
        output['mpgge'] = 0

    else:
        output['mpgge'] = sum(tnt.distMiles) / (sum(tnt.fsKwhOutAch) * (1 / kWhPerGGE))

    roadwayChgKj = sum(tnt.roadwayChgKwOutAch * secs)
    essDischKj = -(tnt.soc[-1] - initSoc) * veh.maxEssKwh * 3600.0
    output['battery_kWh_per_mi'] = (essDischKj / 3600.0) / sum(tnt.distMiles)
    output['electric_kWh_per_mi'] = ((roadwayChgKj + essDischKj) / 3600.0) / sum(tnt.distMiles)
    output['maxTraceMissMph'] = mphPerMps * max(abs(cycMps - tnt.mpsAch))
    fuelKj = sum(np.asarray(tnt.fsKwOutAch) * np.asarray(secs))
    roadwayChgKj = sum(np.asarray(tnt.roadwayChgKwOutAch) * np.asarray(secs))
    essDischgKj = -(tnt.soc[-1] - initSoc) * veh.maxEssKwh * 3600.0

    if (fuelKj + roadwayChgKj) == 0:
        output['ess2fuelKwh'] = 1.0

    else:
        output['ess2fuelKwh'] = essDischgKj / (fuelKj + roadwayChgKj)

    output['initial_soc'] = tnt.soc[0]
    output['final_soc'] = tnt.soc[-1]


    if output['mpgge'] == 0:
        Gallons_gas_equivalent_per_mile = output['electric_kWh_per_mi'] / 33.7

    else:
         Gallons_gas_equivalent_per_mile = 1 / output['mpgge'] + output['electric_kWh_per_mi'] / 33.7

    output['mpgge_elec'] = 1 / Gallons_gas_equivalent_per_mile
    output['soc'] = np.asarray(tnt.soc)
    output['distance_mi'] = sum(tnt.distMiles)
    duration_sec = cycSecs[-1] - cycSecs[0]
    output['avg_speed_mph'] = sum(tnt.distMiles) / (duration_sec / 3600.0)
    accel = np.diff(tnt.mphAch) / np.diff(cycSecs)
    output['avg_accel_mphps'] = np.mean(accel[accel > 0])

    if max(tnt.mphAch) > 60:
        output['ZeroToSixtyTime_secs'] = np.interp(60, tnt.mphAch, cycSecs)

    else:
        output['ZeroToSixtyTime_secs'] = 0.0

    #######################################################################
    ####  Time series information for additional analysis / debugging. ####
    ####             Add parameters of interest as needed.             ####
    #######################################################################

    output['fcKwOutAch'] = np.asarray(tnt.fcKwOutAch)
    output['fsKwhOutAch'] = np.asarray(tnt.fsKwhOutAch)
    output['fcKwInAch'] = np.asarray(tnt.fcKwInAch)
    output['time'] = np.asarray(tnt.cycSecs)

    output['localvars'] = locals()

    return output
