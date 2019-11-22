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
    cycMph = [x * mphPerMps for x in cyc['cycMps']]
    secs = np.insert(np.diff(cycSecs), 0, 0)

    def get_time_df():
        """Initializes arrays of time dependent variables as pandas dataframe columns.  
        Returns Pandas DataFrame called dft"""
    
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
        'ascentKw', 'rrKw', 'motor_index_debug', 'debug_flag']

        # create and initialize time array dataframe
        columns = ['cycSecs'] + comp_lim_list + \
            drivetrain_list + control_list + misc_list
        dft = pd.DataFrame(
            np.zeros((len(cycSecs), len(columns))), columns=columns)
        dft['cycSecs'] = cycSecs
        dft.set_index('cycSecs', inplace=True, drop=False)
        dft['fcForcedOn'] = False
        dft['curMaxRoadwayChgKw'] = np.interp(
            cycRoadType, veh.MaxRoadwayChgKw_Roadway, veh.MaxRoadwayChgKw)  
            # *** this is just zeros, and I need to verify that it was zeros before and also 
            # verify that this is the correct behavior.  CB

        ###  Assign First Value  ###
        ### Drive Train
        dft.loc[0, 'cycMet'] = 1
        dft.loc[0, 'curSocTarget'] = veh.maxSoc
        dft.loc[0, 'essCurKwh'] = initSoc * veh.maxEssKwh
        dft.loc[0, 'soc'] = initSoc

        return dft

    dft = get_time_df()

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
            dft.loc[i, 'auxInKw'] = veh.auxKw / veh.altEff
        else:
            dft.loc[i, 'auxInKw'] = veh.auxKw

        # Is SOC below min threshold?
        if dft.loc[i-1, 'soc'] < (veh.minSoc + veh.percHighAccBuf):
            dft.loc[i, 'reachedBuff'] = 0
        else:
            dft.loc[i, 'reachedBuff'] = 1

        # Does the engine need to be on for low SOC or high acceleration
        if dft.loc[i-1, 'soc'] < veh.minSoc or (dft.loc[i-1, 'highAccFcOnTag'] == 1 and dft.loc[i, 'reachedBuff'] == 0):
            dft.loc[i, 'highAccFcOnTag'] = 1
        else:
            dft.loc[i, 'highAccFcOnTag'] = 0
        dft.loc[i, 'maxTracMps'] = dft.loc[i-1, 'mpsAch'] + (maxTracMps2 * secs[i])

        ### Component Limits
        # max fuel storage power output
        dft.loc[i, 'curMaxFsKwOut'] = min( veh.maxFuelStorKw , dft.loc[i-1, 'fsKwOutAch'] + ((veh.maxFuelStorKw/veh.fuelStorSecsToPeakPwr) * (secs[i])))
        # maximum fuel storage power output rate of change
        dft.loc[i, 'fcTransLimKw'] = dft.loc[i-1, 'fcKwOutAch'] + ((veh.maxFuelConvKw / veh.fuelConvSecsToPeakPwr) * (secs[i]))

        dft.loc[i, 'fcMaxKwIn'] = min(dft.loc[i, 'curMaxFsKwOut'], veh.maxFuelStorKw) # *** this min seems redundant with line 518
        dft.loc[i, 'fcFsLimKw'] = veh.fcMaxOutkW
        dft.loc[i, 'curMaxFcKwOut'] = min(veh.maxFuelConvKw,dft.loc[i, 'fcFsLimKw'],dft.loc[i, 'fcTransLimKw'])

        # Does ESS discharge need to be limited? *** I think veh.maxEssKw should also be in the following
        # boolean condition
        if veh.maxEssKwh == 0 or dft.loc[i-1, 'soc'] < veh.minSoc:
            dft.loc[i, 'essCapLimDischgKw'] = 0.0

        else:
            dft.loc[i, 'essCapLimDischgKw'] = (veh.maxEssKwh * np.sqrt(veh.essRoundTripEff)) * 3600.0 * (dft.loc[i-1, 'soc'] - veh.minSoc) / (secs[i])
        dft.loc[i, 'curMaxEssKwOut'] = min(veh.maxEssKw,dft.loc[i, 'essCapLimDischgKw'])

        if  veh.maxEssKwh == 0 or veh.maxEssKw == 0:
            dft.loc[i, 'essCapLimChgKw'] = 0

        else:
            dft.loc[i, 'essCapLimChgKw'] = max(((veh.maxSoc - dft.loc[i-1, 'soc']) * veh.maxEssKwh * (1 / 
            np.sqrt(veh.essRoundTripEff))) / ((secs[i]) * (1 / 3600.0)), 0)

        dft.loc[i, 'curMaxEssChgKw'] = min(dft.loc[i, 'essCapLimChgKw'],veh.maxEssKw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if veh.fcEffType == 4:
            dft.loc[i, 'curMaxElecKw'] = dft.loc[i, 'curMaxFcKwOut'] + dft.loc[i, 'curMaxRoadwayChgKw'] + \
                dft.loc[i, 'curMaxEssKwOut'] - dft.loc[i, 'auxInKw']

        else:
            dft.loc[i, 'curMaxElecKw'] = dft.loc[i, 'curMaxRoadwayChgKw'] + dft.loc[i, 'curMaxEssKwOut'] - dft.loc[i, 'auxInKw']

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        dft.loc[i, 'curMaxAvailElecKw'] = min(dft.loc[i, 'curMaxElecKw'], veh.mcMaxElecInKw)

        if dft.loc[i, 'curMaxElecKw'] > 0:
            # limit power going into e-machine controller to 
            if dft.loc[i, 'curMaxAvailElecKw'] == max(veh.mcKwInArray):
                dft.loc[i, 'mcElecInLimKw'] = min(veh.mcKwOutArray[len(veh.mcKwOutArray) - 1],veh.maxMotorKw)
            else:
                dft.loc[i, 'mcElecInLimKw'] = min(veh.mcKwOutArray[np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 
                0.01, dft.loc[i, 'curMaxAvailElecKw'])) - 1],veh.maxMotorKw)
        else:
            dft.loc[i, 'mcElecInLimKw'] = 0.0
        
        # Motor transient power limit
        dft.loc[i, 'mcTransiLimKw'] = abs(dft.loc[i-1, 'mcMechKwOutAch']) + ((veh.maxMotorKw / veh.motorSecsToPeakPwr) * (secs[i]))
        
        dft.loc[i, 'curMaxMcKwOut'] = max(min(dft.loc[i, 'mcElecInLimKw'],dft.loc[i, 'mcTransiLimKw'],veh.maxMotorKw),-veh.maxMotorKw)

        if dft.loc[i, 'curMaxMcKwOut'] == 0:
            dft.loc[i, 'curMaxMcElecKwIn'] = 0
        else:
            if dft.loc[i, 'curMaxMcKwOut'] == veh.maxMotorKw:
                dft.loc[i, 'curMaxMcElecKwIn'] = dft.loc[i, 'curMaxMcKwOut'] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                dft.loc[i, 'curMaxMcElecKwIn'] = dft.loc[i, 'curMaxMcKwOut'] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray 
                > min(veh.maxMotorKw - 0.01,dft.loc[i, 'curMaxMcKwOut'])) - 1)]

        if veh.maxMotorKw == 0:
            dft.loc[i, 'essLimMcRegenPercKw'] = 0.0

        else:
            dft.loc[i, 'essLimMcRegenPercKw'] = min((dft.loc[i, 'curMaxEssChgKw'] + dft.loc[i, 'auxInKw']) / veh.maxMotorKw,1)
        if dft.loc[i, 'curMaxEssChgKw'] == 0:
            dft.loc[i, 'essLimMcRegenKw'] = 0.0

        else:
            if veh.maxMotorKw == dft.loc[i, 'curMaxEssChgKw'] - dft.loc[i, 'curMaxRoadwayChgKw']:
                dft.loc[i, 'essLimMcRegenKw'] = min(veh.maxMotorKw,dft.loc[i, 'curMaxEssChgKw'] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1])
            else:
                dft.loc[i, 'essLimMcRegenKw'] = min(veh.maxMotorKw,dft.loc[i, 'curMaxEssChgKw'] / veh.mcFullEffArray\
                    [max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,dft.loc[i, 'curMaxEssChgKw'] - dft.loc[i, 'curMaxRoadwayChgKw'])) - 1)])

        dft.loc[i, 'curMaxMechMcKwIn'] = min(dft.loc[i, 'essLimMcRegenKw'],veh.maxMotorKw)
        dft.loc[i, 'curMaxTracKw'] = (((veh.wheelCoefOfFric * veh.driveAxleWeightFrac * veh.vehKg * gravityMPerSec2)\
            / (1 + ((veh.vehCgM * veh.wheelCoefOfFric) / veh.wheelBaseM))) / 1000.0) * (dft.loc[i, 'maxTracMps'])

        if veh.fcEffType == 4:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or dft.loc[i, 'highAccFcOnTag'] == 1:
                dft.loc[i, 'curMaxTransKwOut'] = min((dft.loc[i, 'curMaxMcKwOut'] - dft.loc[i, 'auxInKw']) * veh.transEff,dft.loc[i, 'curMaxTracKw'] / veh.transEff)
                dft.loc[i, 'debug_flag'] = 1

            else:
                dft.loc[i, 'curMaxTransKwOut'] = min((dft.loc[i, 'curMaxMcKwOut'] - min(dft.loc[i, 'curMaxElecKw'], 0)) * veh.transEff,dft.loc[i, 'curMaxTracKw'] / veh.transEff)
                dft.loc[i, 'debug_flag'] = 2

        else:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or dft.loc[i, 'highAccFcOnTag'] == 1:
                dft.loc[i, 'curMaxTransKwOut'] = min((dft.loc[i, 'curMaxMcKwOut'] + dft.loc[i, 'curMaxFcKwOut'] - \
                     dft.loc[i, 'auxInKw']) * veh.transEff,dft.loc[i, 'curMaxTracKw'] / veh.transEff)
                dft.loc[i, 'debug_flag'] = 3

            else:
                dft.loc[i, 'curMaxTransKwOut'] = min((dft.loc[i, 'curMaxMcKwOut'] + dft.loc[i, 'curMaxFcKwOut'] - \
                    min(dft.loc[i, 'curMaxElecKw'],0)) * veh.transEff, dft.loc[i, 'curMaxTracKw'] / veh.transEff)
                dft.loc[i, 'debug_flag'] = 4

        ### Cycle Power
        dft.loc[i, 'cycDragKw'] = 0.5 * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * (((dft.loc[i-1, 'mpsAch'] + cycMps[i]) / 2.0)**3) / 1000.0
        dft.loc[i, 'cycAccelKw'] = (veh.vehKg / (2.0 * (secs[i]))) * ((cycMps[i]**2) - (dft.loc[i-1, 'mpsAch']**2)) / 1000.0
        dft.loc[i, 'cycAscentKw'] = gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * ((dft.loc[i-1, 'mpsAch'] + cycMps[i]) / 2.0) / 1000.0
        dft.loc[i, 'cycTracKwReq'] = dft.loc[i, 'cycDragKw'] + dft.loc[i, 'cycAccelKw'] + dft.loc[i, 'cycAscentKw']
        dft.loc[i, 'spareTracKw'] = dft.loc[i, 'curMaxTracKw'] - dft.loc[i, 'cycTracKwReq']
        dft.loc[i, 'cycRrKw'] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * ((dft.loc[i-1, 'mpsAch'] + cycMps[i]) / 2.0) / 1000.0
        dft.loc[i, 'cycWheelRadPerSec'] = cycMps[i] / veh.wheelRadiusM
        dft.loc[i, 'cycTireInertiaKw'] = (((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * (dft.loc[i, 'cycWheelRadPerSec']**2.0)) / secs[i]) - \
            ((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * ((dft.loc[i-1, 'mpsAch'] / veh.wheelRadiusM)**2.0)) / secs[i])) / 1000.0

        dft.loc[i, 'cycWheelKwReq'] = dft.loc[i, 'cycTracKwReq'] + dft.loc[i, 'cycRrKw'] + dft.loc[i, 'cycTireInertiaKw']
        dft.loc[i, 'regenContrLimKwPerc'] = veh.maxRegen / (1 + veh.regenA * np.exp(-veh.regenB * ((cycMph[i] + dft.loc[i-1, 'mpsAch'] * mphPerMps) / 2.0 + 1 - 0)))
        dft.loc[i, 'cycRegenBrakeKw'] = max(min(dft.loc[i, 'curMaxMechMcKwIn'] * veh.transEff,dft.loc[i, 'regenContrLimKwPerc']*-dft.loc[i, 'cycWheelKwReq']),0)
        dft.loc[i, 'cycFricBrakeKw'] = -min(dft.loc[i, 'cycRegenBrakeKw'] + dft.loc[i, 'cycWheelKwReq'],0)
        dft.loc[i, 'cycTransKwOutReq'] = dft.loc[i, 'cycWheelKwReq'] + dft.loc[i, 'cycFricBrakeKw']

        if dft.loc[i, 'cycTransKwOutReq']<=dft.loc[i, 'curMaxTransKwOut']:
            dft.loc[i, 'cycMet'] = 1
            dft.loc[i, 'transKwOutAch'] = dft.loc[i, 'cycTransKwOutReq']

        else:
            dft.loc[i, 'cycMet'] = -1
            dft.loc[i, 'transKwOutAch'] = dft.loc[i, 'curMaxTransKwOut']

        ################################
        ###   Speed/Distance Calcs   ###
        ################################

        #Cycle is met
        if dft.loc[i, 'cycMet'] == 1:
            dft.loc[i, 'mpsAch'] = cycMps[i]

        #Cycle is not met
        else:
            Drag3 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2
            Accel2 = veh.vehKg / (2.0 * (secs[i]))
            Drag2 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * dft.loc[i-1, 'mpsAch']
            Wheel2 = 0.5 * veh.wheelInertiaKgM2 * veh.numWheels / (secs[i] * (veh.wheelRadiusM**2))
            Drag1 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * ((dft.loc[i-1, 'mpsAch'])**2)
            Roll1 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg / 2.0)
            Ascent1 = (gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg / 2.0)
            Accel0 = -(veh.vehKg * ((dft.loc[i-1, 'mpsAch'])**2)) / (2.0 * (secs[i]))
            Drag0 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * ((dft.loc[i-1, 'mpsAch'])**3)
            Roll0 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * dft.loc[i-1, 'mpsAch'] / 2.0)
            Ascent0 = (gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * dft.loc[i-1, 'mpsAch'] / 2.0)
            Wheel0 = -((0.5 * veh.wheelInertiaKgM2 * veh.numWheels * (dft.loc[i-1, 'mpsAch']**2)) / (secs[i] * (veh.wheelRadiusM**2)))

            Total3 = Drag3 / 1e3
            print(Accel2, Drag2, Wheel2)
            Total2 = (Accel2 + Drag2 + Wheel2) / 1e3
            Total1 = (Drag1 + Roll1 + Ascent1) / 1e3
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / 1e3 - dft.loc[i, 'curMaxTransKwOut']

            Total = [Total3, Total2, Total1, Total0]
            Total_roots = np.roots(Total)
            ind = np.argmin( abs(cycMps[i] - Total_roots) )
            dft.loc[i, 'mpsAch'] = Total_roots[ind]

        dft.loc[i, 'mphAch'] = dft.loc[i, 'mpsAch'] * mphPerMps
        dft.loc[i, 'distMeters'] = dft.loc[i, 'mpsAch'] * secs[i]
        dft.loc[i, 'distMiles'] = dft.loc[i, 'distMeters'] * (1.0 / metersPerMile)

        ### Drive Train
        if dft.loc[i, 'transKwOutAch'] > 0:
            dft.loc[i, 'transKwInAch'] = dft.loc[i, 'transKwOutAch'] / veh.transEff
        else:
            dft.loc[i, 'transKwInAch'] = dft.loc[i, 'transKwOutAch'] * veh.transEff

        if dft.loc[i, 'cycMet'] == 1:

            if veh.fcEffType == 4:
                dft.loc[i, 'minMcKw2HelpFc'] = max(dft.loc[i, 'transKwInAch'], -dft.loc[i, 'curMaxMechMcKwIn'])

            else:
                dft.loc[i, 'minMcKw2HelpFc'] = max(dft.loc[i, 'transKwInAch'] - dft.loc[i, 'curMaxFcKwOut'], -dft.loc[i, 'curMaxMechMcKwIn'])
        else:
            dft.loc[i, 'minMcKw2HelpFc'] = max(dft.loc[i, 'curMaxMcKwOut'], -dft.loc[i, 'curMaxMechMcKwIn'])

        if veh.noElecSys == 'TRUE':
           dft.loc[i, 'regenBufferSoc'] = 0

        elif veh.chargingOn:
           dft.loc[i, 'regenBufferSoc'] = max(veh.maxSoc - (maxRegenKwh / veh.maxEssKwh), (veh.maxSoc + veh.minSoc) / 2)

        else:
           dft.loc[i, 'regenBufferSoc'] = max(((veh.maxEssKwh * veh.maxSoc) - (0.5 * veh.vehKg * (cycMps[i]**2) * (1.0 / 1000) \
               * (1.0 / 3600) * veh.motorPeakEff * veh.maxRegen)) / veh.maxEssKwh,veh.minSoc)

        dft.loc[i, 'essRegenBufferDischgKw'] = min(dft.loc[i, 'curMaxEssKwOut'], max(0,(dft.loc[i-1, 'soc'] - dft.loc[i, 'regenBufferSoc']) * veh.maxEssKwh * 3600 / secs[i]))

        dft.loc[i, 'maxEssRegenBufferChgKw'] = min(max(0,(dft.loc[i, 'regenBufferSoc'] - dft.loc[i-1, 'soc']) * veh.maxEssKwh * 3600.0 / secs[i]),dft.loc[i, 'curMaxEssChgKw'])

        if veh.noElecSys == 'TRUE':
           dft.loc[i, 'accelBufferSoc'] = 0

        else:
           dft.loc[i, 'accelBufferSoc'] = min(max((((((((veh.maxAccelBufferMph * (1 / mphPerMps))**2)) - ((cycMps[i]**2))) / \
               (((veh.maxAccelBufferMph * (1 / mphPerMps))**2))) * (min(veh.maxAccelBufferPercOfUseableSoc * \
                   (veh.maxSoc - veh.minSoc),maxRegenKwh / veh.maxEssKwh) * veh.maxEssKwh)) / veh.maxEssKwh) + \
                       veh.minSoc,veh.minSoc), veh.maxSoc)

        dft.loc[i, 'essAccelBufferChgKw'] = max(0,(dft.loc[i, 'accelBufferSoc'] - dft.loc[i-1, 'soc']) * veh.maxEssKwh * 3600.0 / secs[i])
        dft.loc[i, 'maxEssAccelBufferDischgKw'] = min(max(0, (dft.loc[i-1, 'soc'] - dft.loc[i, 'accelBufferSoc']) * veh.maxEssKwh * 3600 / secs[i]),dft.loc[i, 'curMaxEssKwOut'])

        if dft.loc[i, 'regenBufferSoc'] < dft.loc[i, 'accelBufferSoc']:
            dft.loc[i, 'essAccelRegenDischgKw'] = max(min(((dft.loc[i-1, 'soc'] - (dft.loc[i, 'regenBufferSoc'] + dft.loc[i, 'accelBufferSoc']) / 2) * veh.maxEssKwh * 3600.0) /\
                 secs[i],dft.loc[i, 'curMaxEssKwOut']),-dft.loc[i, 'curMaxEssChgKw'])

        elif dft.loc[i-1, 'soc'] > dft.loc[i, 'regenBufferSoc']:
            dft.loc[i, 'essAccelRegenDischgKw'] = max(min(dft.loc[i, 'essRegenBufferDischgKw'],dft.loc[i, 'curMaxEssKwOut']),-dft.loc[i, 'curMaxEssChgKw'])

        elif dft.loc[i-1, 'soc'] < dft.loc[i, 'accelBufferSoc']:
            dft.loc[i, 'essAccelRegenDischgKw'] = max(min(-1.0 * dft.loc[i, 'essAccelBufferChgKw'],dft.loc[i, 'curMaxEssKwOut']),-dft.loc[i, 'curMaxEssChgKw'])

        else:
            dft.loc[i, 'essAccelRegenDischgKw'] = max(min(0,dft.loc[i, 'curMaxEssKwOut']),-dft.loc[i, 'curMaxEssChgKw'])

        dft.loc[i, 'fcKwGapFrEff'] = abs(dft.loc[i, 'transKwOutAch'] - veh.maxFcEffKw)

        if veh.noElecSys == 'TRUE':
            dft.loc[i, 'mcElectInKwForMaxFcEff'] = 0

        elif dft.loc[i, 'transKwOutAch'] < veh.maxFcEffKw:

            if dft.loc[i, 'fcKwGapFrEff'] == veh.maxMotorKw:
                dft.loc[i, 'mcElectInKwForMaxFcEff'] = dft.loc[i, 'fcKwGapFrEff'] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]*-1
            else:
                dft.loc[i, 'mcElectInKwForMaxFcEff'] = dft.loc[i, 'fcKwGapFrEff'] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,dft.loc[i, 'fcKwGapFrEff'])) - 1)]*-1

        else:

            if dft.loc[i, 'fcKwGapFrEff'] == veh.maxMotorKw:
                dft.loc[i, 'mcElectInKwForMaxFcEff'] = veh.mcKwInArray[len(veh.mcKwInArray) - 1]
            else:
                dft.loc[i, 'mcElectInKwForMaxFcEff'] = veh.mcKwInArray[np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,dft.loc[i, 'fcKwGapFrEff'])) - 1]

        if veh.noElecSys == 'TRUE':
            dft.loc[i, 'electKwReq4AE'] = 0

        elif dft.loc[i, 'transKwInAch'] > 0:
            if dft.loc[i, 'transKwInAch'] == veh.maxMotorKw:
        
                dft.loc[i, 'electKwReq4AE'] = dft.loc[i, 'transKwInAch'] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1] + dft.loc[i, 'auxInKw']
            else:
                dft.loc[i, 'electKwReq4AE'] = dft.loc[i, 'transKwInAch'] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,dft.loc[i, 'transKwInAch'])) - 1)] + dft.loc[i, 'auxInKw']

        else:
           dft.loc[i, 'electKwReq4AE'] = 0

        dft.loc[i, 'prevfcTimeOn'] = dft.loc[i-1, 'fcTimeOn']

        if veh.maxFuelConvKw == 0:
            dft.loc[i, 'canPowerAllElectrically'] = dft.loc[i, 'accelBufferSoc'] < dft.loc[i-1, 'soc'] and dft.loc[i, 'transKwInAch']<=dft.loc[i, 'curMaxMcKwOut'] and (dft.loc[i, 'electKwReq4AE'] < dft.loc[i, 'curMaxElecKw'] or veh.maxFuelConvKw == 0)

        else:
            dft.loc[i, 'canPowerAllElectrically'] = dft.loc[i, 'accelBufferSoc'] < dft.loc[i-1, 'soc'] and dft.loc[i, 'transKwInAch']<=dft.loc[i, 'curMaxMcKwOut'] and (dft.loc[i, 'electKwReq4AE'] < dft.loc[i, 'curMaxElecKw'] \
                or veh.maxFuelConvKw == 0) and (cycMph[i] - 0.00001<=veh.mphFcOn or veh.chargingOn) and dft.loc[i, 'electKwReq4AE']<=veh.kwDemandFcOn

        if dft.loc[i, 'canPowerAllElectrically']:

            if dft.loc[i, 'transKwInAch']<+dft.loc[i, 'auxInKw']:
                dft.loc[i, 'desiredEssKwOutForAE'] = dft.loc[i, 'auxInKw'] + dft.loc[i, 'transKwInAch']

            elif dft.loc[i, 'regenBufferSoc'] < dft.loc[i, 'accelBufferSoc']:
                dft.loc[i, 'desiredEssKwOutForAE'] = dft.loc[i, 'essAccelRegenDischgKw']

            elif dft.loc[i-1, 'soc'] > dft.loc[i, 'regenBufferSoc']:
                dft.loc[i, 'desiredEssKwOutForAE'] = dft.loc[i, 'essRegenBufferDischgKw']

            elif dft.loc[i-1, 'soc'] < dft.loc[i, 'accelBufferSoc']:
                dft.loc[i, 'desiredEssKwOutForAE'] = -dft.loc[i, 'essAccelBufferChgKw']

            else:
                dft.loc[i, 'desiredEssKwOutForAE'] = dft.loc[i, 'transKwInAch'] + dft.loc[i, 'auxInKw'] - dft.loc[i, 'curMaxRoadwayChgKw']

        else:
            dft.loc[i, 'desiredEssKwOutForAE'] = 0

        if dft.loc[i, 'canPowerAllElectrically']:
            dft.loc[i, 'essAEKwOut'] = max(-dft.loc[i, 'curMaxEssChgKw'],-dft.loc[i, 'maxEssRegenBufferChgKw'],min(0,dft.loc[i, 'curMaxRoadwayChgKw'] - (dft.loc[i, 'transKwInAch'] + dft.loc[i, 'auxInKw'])),min(dft.loc[i, 'curMaxEssKwOut'],dft.loc[i, 'desiredEssKwOutForAE']))

        else:
            dft.loc[i, 'essAEKwOut'] = 0

        dft.loc[i, 'erAEKwOut'] = min(max(0,dft.loc[i, 'transKwInAch'] + dft.loc[i, 'auxInKw'] - dft.loc[i, 'essAEKwOut']),dft.loc[i, 'curMaxRoadwayChgKw'])

        if dft.loc[i, 'prevfcTimeOn'] > 0 and dft.loc[i, 'prevfcTimeOn'] < veh.minFcTimeOn - secs[i]:
            dft.loc[i, 'fcForcedOn'] = True
        else:
            dft.loc[i, 'fcForcedOn'] = False

        if dft.loc[i, 'fcForcedOn'] == False or dft.loc[i, 'canPowerAllElectrically'] == False:
            dft.loc[i, 'fcForcedState'] = 1
            dft.loc[i, 'mcMechKw4ForcedFc'] = 0

        elif dft.loc[i, 'transKwInAch'] < 0:
            dft.loc[i, 'fcForcedState'] = 2
            dft.loc[i, 'mcMechKw4ForcedFc'] = dft.loc[i, 'transKwInAch']

        elif veh.maxFcEffKw == dft.loc[i, 'transKwInAch']:
            dft.loc[i, 'fcForcedState'] = 3
            dft.loc[i, 'mcMechKw4ForcedFc'] = 0

        elif veh.idleFcKw > dft.loc[i, 'transKwInAch'] and dft.loc[i, 'cycAccelKw'] >=0:
            dft.loc[i, 'fcForcedState'] = 4
            dft.loc[i, 'mcMechKw4ForcedFc'] = dft.loc[i, 'transKwInAch'] - veh.idleFcKw

        elif veh.maxFcEffKw > dft.loc[i, 'transKwInAch']:
            dft.loc[i, 'fcForcedState'] = 5
            dft.loc[i, 'mcMechKw4ForcedFc'] = 0

        else:
            dft.loc[i, 'fcForcedState'] = 6
            dft.loc[i, 'mcMechKw4ForcedFc'] = dft.loc[i, 'transKwInAch'] - veh.maxFcEffKw

        if (-dft.loc[i, 'mcElectInKwForMaxFcEff'] - dft.loc[i, 'curMaxRoadwayChgKw']) > 0:
            dft.loc[i, 'essDesiredKw4FcEff'] = (-dft.loc[i, 'mcElectInKwForMaxFcEff'] - dft.loc[i, 'curMaxRoadwayChgKw']) * veh.essDischgToFcMaxEffPerc

        else:
            dft.loc[i, 'essDesiredKw4FcEff'] = (-dft.loc[i, 'mcElectInKwForMaxFcEff'] - dft.loc[i, 'curMaxRoadwayChgKw']) * veh.essChgToFcMaxEffPerc

        if dft.loc[i, 'accelBufferSoc'] > dft.loc[i, 'regenBufferSoc']:
            dft.loc[i, 'essKwIfFcIsReq'] = min(dft.loc[i, 'curMaxEssKwOut'],veh.mcMaxElecInKw + dft.loc[i, 'auxInKw'],dft.loc[i, 'curMaxMcElecKwIn'] + dft.loc[i, 'auxInKw'], \
                max(-dft.loc[i, 'curMaxEssChgKw'], dft.loc[i, 'essAccelRegenDischgKw']))

        elif dft.loc[i, 'essRegenBufferDischgKw'] > 0:
            dft.loc[i, 'essKwIfFcIsReq'] = min(dft.loc[i, 'curMaxEssKwOut'],veh.mcMaxElecInKw + dft.loc[i, 'auxInKw'],dft.loc[i, 'curMaxMcElecKwIn'] + dft.loc[i, 'auxInKw'], \
                max(-dft.loc[i, 'curMaxEssChgKw'], min(dft.loc[i, 'essAccelRegenDischgKw'],dft.loc[i, 'mcElecInLimKw'] + dft.loc[i, 'auxInKw'], max(dft.loc[i, 'essRegenBufferDischgKw'],dft.loc[i, 'essDesiredKw4FcEff']))))

        elif dft.loc[i, 'essAccelBufferChgKw'] > 0:
            dft.loc[i, 'essKwIfFcIsReq'] = min(dft.loc[i, 'curMaxEssKwOut'],veh.mcMaxElecInKw + dft.loc[i, 'auxInKw'],dft.loc[i, 'curMaxMcElecKwIn'] + dft.loc[i, 'auxInKw'], \
                max(-dft.loc[i, 'curMaxEssChgKw'], max(-1 * dft.loc[i, 'maxEssRegenBufferChgKw'], min(-dft.loc[i, 'essAccelBufferChgKw'],dft.loc[i, 'essDesiredKw4FcEff']))))


        elif dft.loc[i, 'essDesiredKw4FcEff'] > 0:
            dft.loc[i, 'essKwIfFcIsReq'] = min(dft.loc[i, 'curMaxEssKwOut'],veh.mcMaxElecInKw + dft.loc[i, 'auxInKw'],dft.loc[i, 'curMaxMcElecKwIn'] + dft.loc[i, 'auxInKw'], \
                max(-dft.loc[i, 'curMaxEssChgKw'], min(dft.loc[i, 'essDesiredKw4FcEff'],dft.loc[i, 'maxEssAccelBufferDischgKw'])))

        else:
            dft.loc[i, 'essKwIfFcIsReq'] = min(dft.loc[i, 'curMaxEssKwOut'],veh.mcMaxElecInKw + dft.loc[i, 'auxInKw'],dft.loc[i, 'curMaxMcElecKwIn'] + dft.loc[i, 'auxInKw'], \
                max(-dft.loc[i, 'curMaxEssChgKw'], max(dft.loc[i, 'essDesiredKw4FcEff'],-dft.loc[i, 'maxEssRegenBufferChgKw'])))

        dft.loc[i, 'erKwIfFcIsReq'] = max(0,min(dft.loc[i, 'curMaxRoadwayChgKw'],dft.loc[i, 'curMaxMechMcKwIn'],dft.loc[i, 'essKwIfFcIsReq'] - dft.loc[i, 'mcElecInLimKw'] + dft.loc[i, 'auxInKw']))

        dft.loc[i, 'mcElecKwInIfFcIsReq'] = dft.loc[i, 'essKwIfFcIsReq'] + dft.loc[i, 'erKwIfFcIsReq'] - dft.loc[i, 'auxInKw']

        if veh.noElecSys == 'TRUE':
            dft.loc[i, 'mcKwIfFcIsReq'] = 0

        elif  dft.loc[i, 'mcElecKwInIfFcIsReq'] == 0:
            dft.loc[i, 'mcKwIfFcIsReq'] = 0

        elif dft.loc[i, 'mcElecKwInIfFcIsReq'] > 0:

            if dft.loc[i, 'mcElecKwInIfFcIsReq'] == max(veh.mcKwInArray):
                 dft.loc[i, 'mcKwIfFcIsReq'] = dft.loc[i, 'mcElecKwInIfFcIsReq'] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                 dft.loc[i, 'mcKwIfFcIsReq'] = dft.loc[i, 'mcElecKwInIfFcIsReq'] * veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,dft.loc[i, 'mcElecKwInIfFcIsReq'])) - 1)]

        else:
            if dft.loc[i, 'mcElecKwInIfFcIsReq']*-1 == max(veh.mcKwInArray):
                dft.loc[i, 'mcKwIfFcIsReq'] = dft.loc[i, 'mcElecKwInIfFcIsReq'] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                dft.loc[i, 'mcKwIfFcIsReq'] = dft.loc[i, 'mcElecKwInIfFcIsReq'] / (veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,dft.loc[i, 'mcElecKwInIfFcIsReq']*-1)) - 1)])

        if veh.maxMotorKw == 0:
            dft.loc[i, 'mcMechKwOutAch'] = 0

        elif dft.loc[i, 'fcForcedOn'] == True and dft.loc[i, 'canPowerAllElectrically'] == True and (veh.vehPtType == 2.0 or veh.vehPtType == 3.0) and veh.fcEffType!=4:
           dft.loc[i, 'mcMechKwOutAch'] =  dft.loc[i, 'mcMechKw4ForcedFc']

        elif dft.loc[i, 'transKwInAch']<=0:

            if veh.fcEffType!=4 and veh.maxFuelConvKw> 0:
                if dft.loc[i, 'canPowerAllElectrically'] == 1:
                    dft.loc[i, 'mcMechKwOutAch'] = -min(dft.loc[i, 'curMaxMechMcKwIn'],-dft.loc[i, 'transKwInAch'])
                else:
                    dft.loc[i, 'mcMechKwOutAch'] = min(-min(dft.loc[i, 'curMaxMechMcKwIn'], -dft.loc[i, 'transKwInAch']),max(-dft.loc[i, 'curMaxFcKwOut'], dft.loc[i, 'mcKwIfFcIsReq']))
            else:
                dft.loc[i, 'mcMechKwOutAch'] = min(-min(dft.loc[i, 'curMaxMechMcKwIn'],-dft.loc[i, 'transKwInAch']),-dft.loc[i, 'transKwInAch'])

        elif dft.loc[i, 'canPowerAllElectrically'] == 1:
            dft.loc[i, 'mcMechKwOutAch'] = dft.loc[i, 'transKwInAch']

        else:
            dft.loc[i, 'mcMechKwOutAch'] = max(dft.loc[i, 'minMcKw2HelpFc'],dft.loc[i, 'mcKwIfFcIsReq'])

        if dft.loc[i, 'mcMechKwOutAch'] == 0:
            dft.loc[i, 'mcElecKwInAch'] = 0.0
            dft.loc[i, 'motor_index_debug'] = 0

        elif dft.loc[i, 'mcMechKwOutAch'] < 0:

            if dft.loc[i, 'mcMechKwOutAch']*-1 == max(veh.mcKwInArray):
                dft.loc[i, 'mcElecKwInAch'] = dft.loc[i, 'mcMechKwOutAch'] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                dft.loc[i, 'mcElecKwInAch'] = dft.loc[i, 'mcMechKwOutAch'] * veh.mcFullEffArray[max(1,np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01,dft.loc[i, 'mcMechKwOutAch']*-1)) - 1)]

        else:
            if veh.maxMotorKw == dft.loc[i, 'mcMechKwOutAch']:
                dft.loc[i, 'mcElecKwInAch'] = dft.loc[i, 'mcMechKwOutAch'] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                dft.loc[i, 'mcElecKwInAch'] = dft.loc[i, 'mcMechKwOutAch'] / veh.mcFullEffArray[max(1,np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01,dft.loc[i, 'mcMechKwOutAch'])) - 1)]

        if dft.loc[i, 'curMaxRoadwayChgKw'] == 0:
            dft.loc[i, 'roadwayChgKwOutAch'] = 0

        elif veh.fcEffType == 4:
            dft.loc[i, 'roadwayChgKwOutAch'] = max(0, dft.loc[i, 'mcElecKwInAch'], dft.loc[i, 'maxEssRegenBufferChgKw'], dft.loc[i, 'essRegenBufferDischgKw'], dft.loc[i, 'curMaxRoadwayChgKw'])

        elif dft.loc[i, 'canPowerAllElectrically'] == 1:
            dft.loc[i, 'roadwayChgKwOutAch'] = dft.loc[i, 'erAEKwOut']

        else:
            dft.loc[i, 'roadwayChgKwOutAch'] = dft.loc[i, 'erKwIfFcIsReq']

        dft.loc[i, 'minEssKw2HelpFc'] = dft.loc[i, 'mcElecKwInAch'] + dft.loc[i, 'auxInKw'] - dft.loc[i, 'curMaxFcKwOut'] - dft.loc[i, 'roadwayChgKwOutAch']

        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            dft.loc[i, 'essKwOutAch']  = 0

        elif veh.fcEffType == 4:

            if dft.loc[i, 'transKwOutAch']>=0:
                dft.loc[i, 'essKwOutAch'] = min(max(dft.loc[i, 'minEssKw2HelpFc'],dft.loc[i, 'essDesiredKw4FcEff'],dft.loc[i, 'essAccelRegenDischgKw']),dft.loc[i, 'curMaxEssKwOut'],dft.loc[i, 'mcElecKwInAch'] + dft.loc[i, 'auxInKw'] - dft.loc[i, 'roadwayChgKwOutAch'])

            else:
                dft.loc[i, 'essKwOutAch'] = dft.loc[i, 'mcElecKwInAch'] + dft.loc[i, 'auxInKw'] - dft.loc[i, 'roadwayChgKwOutAch']

        elif dft.loc[i, 'highAccFcOnTag'] == 1 or veh.noElecAux == 'TRUE':
            dft.loc[i, 'essKwOutAch'] = dft.loc[i, 'mcElecKwInAch'] - dft.loc[i, 'roadwayChgKwOutAch']

        else:
            dft.loc[i, 'essKwOutAch'] = dft.loc[i, 'mcElecKwInAch'] + dft.loc[i, 'auxInKw'] - dft.loc[i, 'roadwayChgKwOutAch']

        if veh.maxFuelConvKw == 0:
            dft.loc[i, 'fcKwOutAch'] = 0

        elif veh.fcEffType == 4:
            dft.loc[i, 'fcKwOutAch'] = min(dft.loc[i, 'curMaxFcKwOut'], max(0, dft.loc[i, 'mcElecKwInAch'] + dft.loc[i, 'auxInKw'] - dft.loc[i, 'essKwOutAch'] - dft.loc[i, 'roadwayChgKwOutAch']))

        elif veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or dft.loc[i, 'highAccFcOnTag'] == 1:
            dft.loc[i, 'fcKwOutAch'] = min(dft.loc[i, 'curMaxFcKwOut'], max(0, dft.loc[i, 'transKwInAch'] - dft.loc[i, 'mcMechKwOutAch'] + dft.loc[i, 'auxInKw']))

        else:
            dft.loc[i, 'fcKwOutAch'] = min(dft.loc[i, 'curMaxFcKwOut'], max(0, dft.loc[i, 'transKwInAch'] - dft.loc[i, 'mcMechKwOutAch']))

        if dft.loc[i, 'fcKwOutAch'] == 0:
            dft.loc[i, 'fcKwInAch'] = 0.0
            dft.loc[i, 'fcKwOutAch_pct'] = 0

        if veh.maxFuelConvKw == 0:
            dft.loc[i, 'fcKwOutAch_pct'] = 0
        else:
            dft.loc[i, 'fcKwOutAch_pct'] = dft.loc[i, 'fcKwOutAch'] / veh.maxFuelConvKw

        if dft.loc[i, 'fcKwOutAch'] == 0:
            dft.loc[i, 'fcKwInAch'] = 0
        else:
            if dft.loc[i, 'fcKwOutAch'] == veh.fcMaxOutkW:
                dft.loc[i, 'fcKwInAch'] = dft.loc[i, 'fcKwOutAch'] / veh.fcEffArray[len(veh.fcEffArray) - 1]
            else:
                dft.loc[i, 'fcKwInAch'] = dft.loc[i, 'fcKwOutAch'] / (veh.fcEffArray[max(1,np.argmax(veh.fcKwOutArray > min(dft.loc[i, 'fcKwOutAch'],veh.fcMaxOutkW - 0.001)) - 1)])

        dft.loc[i, 'fsKwOutAch'] = np.copy(dft.loc[i, 'fcKwInAch'])

        dft.loc[i, 'fsKwhOutAch'] = dft.loc[i, 'fsKwOutAch'] * secs[i] * (1 / 3600.0)


        if veh.noElecSys == 'TRUE':
            dft.loc[i, 'essCurKwh'] = 0

        elif dft.loc[i, 'essKwOutAch'] < 0:
            dft.loc[i, 'essCurKwh'] = dft.loc[i-1, 'essCurKwh'] - dft.loc[i, 'essKwOutAch'] * (secs[i] / 3600.0) * np.sqrt(veh.essRoundTripEff)

        else:
            dft.loc[i, 'essCurKwh'] = dft.loc[i-1, 'essCurKwh'] - dft.loc[i, 'essKwOutAch'] * (secs[i] / 3600.0) * (1 / np.sqrt(veh.essRoundTripEff))

        if veh.maxEssKwh == 0:
            dft.loc[i, 'soc'] = 0.0

        else:
            dft.loc[i, 'soc'] = dft.loc[i, 'essCurKwh'] / veh.maxEssKwh

        if dft.loc[i, 'canPowerAllElectrically'] == True and dft.loc[i, 'fcForcedOn'] == False and dft.loc[i, 'fcKwOutAch'] == 0:
            dft.loc[i, 'fcTimeOn'] = 0
        else:
            dft.loc[i, 'fcTimeOn'] = dft.loc[i-1, 'fcTimeOn'] + secs[i]

        ### Battery wear calcs

        if veh.noElecSys!='TRUE':

            if dft.loc[i, 'essCurKwh'] > dft.loc[i-1, 'essCurKwh']:
                dft.loc[i, 'addKwh'] = (dft.loc[i, 'essCurKwh'] - dft.loc[i-1, 'essCurKwh']) + dft.loc[i-1, 'addKwh']
            else:
                dft.loc[i, 'addKwh'] = 0

            if dft.loc[i, 'addKwh'] == 0:
                dft.loc[i, 'dodCycs'] = dft.loc[i-1, 'addKwh'] / veh.maxEssKwh
            else:
                dft.loc[i, 'dodCycs'] = 0

            if dft.loc[i, 'dodCycs']!=0:
                dft.loc[i, 'essPercDeadArray'] = np.power(veh.essLifeCoefA,1.0 / veh.essLifeCoefB) / np.power(dft.loc[i, 'dodCycs'],1.0 / veh.essLifeCoefB)
            else:
                dft.loc[i, 'essPercDeadArray'] = 0

        ### Energy Audit Calculations
        dft.loc[i, 'dragKw'] = 0.5 * airDensityKgPerM3 * veh.dragCoef * veh.frontalAreaM2 * (((dft.loc[i-1, 'mpsAch'] + dft.loc[i, 'mpsAch']) / 2.0)**3) / 1000.0
        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            dft.loc[i, 'essLossKw'] = 0
        elif dft.loc[i, 'essKwOutAch'] < 0:
            dft.loc[i, 'essLossKw'] = -dft.loc[i, 'essKwOutAch'] - (-dft.loc[i, 'essKwOutAch'] * np.sqrt(veh.essRoundTripEff))
        else:
            dft.loc[i, 'essLossKw'] = dft.loc[i, 'essKwOutAch'] * (1.0 / np.sqrt(veh.essRoundTripEff)) - dft.loc[i, 'essKwOutAch']
        dft.loc[i, 'accelKw'] = (veh.vehKg / (2.0 * (secs[i]))) * ((dft.loc[i, 'mpsAch']**2) - (dft.loc[i-1, 'mpsAch']**2)) / 1000.0
        dft.loc[i, 'ascentKw'] = gravityMPerSec2 * np.sin(np.arctan(cycGrade[i])) * veh.vehKg * ((dft.loc[i-1, 'mpsAch'] + dft.loc[i, 'mpsAch']) / 2.0) / 1000.0
        dft.loc[i, 'rrKw'] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * ((dft.loc[i-1, 'mpsAch'] + dft.loc[i, 'mpsAch']) / 2.0) / 1000.0

    ############################################
    ### Calculate Results and Assign Outputs ###
    ############################################

    output = dict()

    if sum(dft['fsKwhOutAch']) == 0:
        output['mpgge'] = 0

    else:
        output['mpgge'] = sum(dft['distMiles']) / (sum(dft['fsKwhOutAch']) * (1 / kWhPerGGE))

    roadwayChgKj = sum(dft['roadwayChgKwOutAch'] * secs)
    essDischKj = -(dft['soc'][-1] - initSoc) * veh.maxEssKwh * 3600.0
    output['battery_kWh_per_mi'] = (essDischKj / 3600.0) / sum(dft['distMiles'])
    output['electric_kWh_per_mi'] = ((roadwayChgKj + essDischKj) / 3600.0) / sum(dft['distMiles'])
    output['maxTraceMissMph'] = mphPerMps * max(abs(cycMps - dft['mpsAch']))
    fuelKj = sum(np.asarray(dft['fsKwOutAch']) * np.asarray(secs))
    roadwayChgKj = sum(np.asarray(dft['roadwayChgKwOutAch']) * np.asarray(secs))
    essDischgKj = -(dft['soc'][-1] - initSoc) * veh.maxEssKwh * 3600.0

    if (fuelKj + roadwayChgKj) == 0:
        output['ess2fuelKwh'] = 1.0

    else:
        output['ess2fuelKwh'] = essDischgKj / (fuelKj + roadwayChgKj)

    output['initial_soc'] = dft['soc'][0]
    output['final_soc'] = dft['soc'][-1]


    if output['mpgge'] == 0:
        Gallons_gas_equivalent_per_mile = output['electric_kWh_per_mi'] / 33.7

    else:
         Gallons_gas_equivalent_per_mile = 1 / output['mpgge'] + output['electric_kWh_per_mi'] / 33.7

    output['mpgge_elec'] = 1 / Gallons_gas_equivalent_per_mile
    output['soc'] = np.asarray(dft['soc'])
    output['distance_mi'] = sum(dft['distMiles'])
    duration_sec = cycSecs[-1] - cycSecs[0]
    output['avg_speed_mph'] = sum(dft['distMiles']) / (duration_sec / 3600.0)
    accel = np.diff(dft['mphAch']) / np.diff(cycSecs)
    output['avg_accel_mphps'] = np.mean(accel[accel > 0])

    if max(dft['mphAch']) > 60:
        output['ZeroToSixtyTime_secs'] = np.interp(60, dft['mphAch'], cycSecs)

    else:
        output['ZeroToSixtyTime_secs'] = 0.0

    #######################################################################
    ####  Time series information for additional analysis / debugging. ####
    ####             Add parameters of interest as needed.             ####
    #######################################################################

    output['fcKwOutAch'] = np.asarray(dft['fcKwOutAch'])
    output['fsKwhOutAch'] = np.asarray(dft['fsKwhOutAch'])
    output['fcKwInAch'] = np.asarray(dft['fcKwInAch'])
    output['time'] = np.asarray(dft['cycSecs'])

    output['localvars'] = locals()

    return output
