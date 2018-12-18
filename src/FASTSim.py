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
import warnings
import csv
warnings.simplefilter('ignore')

def get_standard_cycle(cycle_name):
    csv_path = '..//cycles//'+cycle_name+'.csv'
    data = dict()
    dkeys=[]
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(data)==0: # initialize all elements in dictionary based on header
                for ii in range(len(row)):
                    data[row[ii]] = []
                    dkeys.append( row[ii] )
            else: # append values
                for ii in range(len(row)):
                    try:
                        data[dkeys[ii]].append( float(row[ii]) )
                    except:
                        data[dkeys[ii]].append( np.nan )
    for ii in range(len(dkeys)):
        data[dkeys[ii]] = np.array(data[dkeys[ii]])
    return data

def get_veh(vnum):
    with open('..//docs//FASTSim_py_veh_db.csv','r') as csvfile:

        reader = csv.reader(csvfile)
        vd = dict()
        data = dict()
        z=0

        for i in reader:
           data[z]=i
           z=z+1

        variables = data[0]
        del data[0] # deletes the first list, which corresponds to the header w/ variable names
        vd=data

        ### selects specified vnum from vd
        veh = dict()
        variables = ['selection','name', 'vehPtType', 'dragCoef', 'frontalAreaM2', 'gliderKg', 'vehCgM', 'driveAxleWeightFrac', 'wheelBaseM', 'cargoKg', 'vehOverrideKg', 'maxFuelStorKw', 'fuelStorSecsToPeakPwr', 'fuelStorKwh', 'fuelStorKwhPerKg', 'maxFuelConvKw', 'fcEffType', 'fcAbsEffImpr', 'fuelConvSecsToPeakPwr', 'fuelConvBaseKg', 'fuelConvKwPerKg', 'maxMotorKw', 'motorPeakEff', 'motorSecsToPeakPwr', 'mcPeKgPerKw', 'mcPeBaseKg', 'maxEssKw', 'maxEssKwh', 'essKgPerKwh', 'essBaseKg', 'essRoundTripEff', 'essLifeCoefA', 'essLifeCoefB', 'wheelInertiaKgM2', 'numWheels', 'wheelRrCoef', 'wheelRadiusM', 'wheelCoefOfFric', 'minSoc', 'maxSoc', 'essDischgToFcMaxEffPerc', 'essChgToFcMaxEffPerc', 'maxAccelBufferMph', 'maxAccelBufferPercOfUseableSoc', 'percHighAccBuf', 'mphFcOn', 'kwDemandFcOn', 'altEff', 'chgEff', 'auxKw', 'forceAuxOnFC', 'transKg', 'transEff', 'compMassMultiplier', 'essToFuelOkError', 'maxRegen', 'valUddsMpgge', 'valHwyMpgge', 'valCombMpgge', 'valUddsKwhPerMile', 'valHwyKwhPerMile', 'valCombKwhPerMile', 'valCdRangeMi', 'valConst65MphKwhPerMile', 'valConst60MphKwhPerMile', 'valConst55MphKwhPerMile', 'valConst45MphKwhPerMile', 'valUnadjUddsKwhPerMile', 'valUnadjHwyKwhPerMile', 'val0To60Mph', 'valEssLifeMiles', 'valRangeMiles', 'valVehBaseCost', 'valMsrp', 'minFcTimeOn', 'idleFcKw']
        if vnum in vd:
            for i in range(len(variables)):
                vd[vnum][i]=str(vd[vnum][i])
                if vd[vnum][i].find('%') != -1:
                    vd[vnum][i]=vd[vnum][i].replace('%','')
                    vd[vnum][i]=float(vd[vnum][i])
                    vd[vnum][i]=vd[vnum][i]/100.0
                elif vd[vnum][i].find('TRUE') != -1 or vd[vnum][i].find('True') != -1 or vd[vnum][i].find('true') != -1:
                    vd[vnum][i]=1
                elif vd[vnum][i].find('FALSE') != -1 or vd[vnum][i].find('False') != -1 or vd[vnum][i].find('false') != -1:
                    vd[vnum][i]=1
                else:
                    try:
                        vd[vnum][i]=float(vd[vnum][i])
                    except:
                        pass
                veh[variables[i]]=vd[vnum][i]

    ######################################################################
    ### Append additional parameters to veh structure from calculation ###
    ######################################################################

    ### Build roadway power lookup table
    veh['MaxRoadwayChgKw_Roadway'] = range(6)
    veh['MaxRoadwayChgKw'] = [0]*len(veh['MaxRoadwayChgKw_Roadway'])
    veh['chargingOn'] = 0

     # Checking if a vehicle has any hybrid components
    if veh['maxEssKwh']==0 or veh['maxEssKw']==0 or veh['maxMotorKw']==0:
        veh['noElecSys'] = 'TRUE'

    else:
        veh['noElecSys'] = 'FALSE'

    # Checking if aux loads go through an alternator
    if veh['noElecSys']=='TRUE' or veh['maxMotorKw']<=veh['auxKw'] or veh['forceAuxOnFC']=='TRUE':
        veh['noElecAux'] = 'TRUE'

    else:
        veh['noElecAux'] = 'FALSE'

    veh['vehTypeSelection'] = np.copy( veh['vehPtType'] ) # Copying vehPtType to additional key

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


        if veh['fcEffType']==1:
            eff = np.copy( eff_si ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==2:
            eff = np.copy( eff_atk ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==3:
            eff = np.copy( eff_diesel ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==4:
            eff = np.copy( eff_fuel_cell ) + veh['fcAbsEffImpr']

        elif veh['fcEffType']==5:
            eff = np.copy( eff_hd_diesel ) + veh['fcAbsEffImpr']

        inputKwOutArray = fcPwrOutPerc * veh['maxFuelConvKw']
        fcPercOutArray = np.r_[np.arange(0,3.0,0.1),np.arange(3.0,7.0,0.5),np.arange(7.0,60.0,1.0),np.arange(60.0,105.0,5.0)] / 100
        fcKwOutArray = veh['maxFuelConvKw'] * fcPercOutArray
        fcEffArray = np.array([0.0]*len(fcPercOutArray))

        for j in range(0,len(fcPercOutArray)-1):

            low_index = np.argmax(inputKwOutArray>=fcKwOutArray[j])
            fcinterp_x_1 = inputKwOutArray[low_index-1]
            fcinterp_x_2 = inputKwOutArray[low_index]
            fcinterp_y_1 = eff[low_index-1]
            fcinterp_y_2 = eff[low_index]
            fcEffArray[j] = (fcKwOutArray[j] - fcinterp_x_1)/(fcinterp_x_2 - fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        fcEffArray[-1] = eff[-1]
        veh['fcEffArray'] = np.copy(fcEffArray)
        veh['fcKwOutArray'] = np.copy(fcKwOutArray)
        veh['maxFcEffKw'] = np.copy(veh['fcKwOutArray'][np.argmax(fcEffArray)])
        veh['fcMaxOutkW'] = np.copy(max(inputKwOutArray))
        veh['minFcTimeOn'] = 30

    else:
        veh['fcKwOutArray'] = np.array([0]*101)
        veh['maxFcEffKw'] = 0
        veh['fcMaxOutkW'] = 0
        veh['minFcTimeOn'] = 30

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

        mcKwAdjPerc = max(0.0,min((maxMotorKw - 7.5)/(75.0-7.5),1.0))
        mcEffArray = np.array([0.0]*len(mcPwrOutPerc))

        for k in range(0,len(mcPwrOutPerc)):
            mcEffArray[k] = mcKwAdjPerc*large_baseline_eff_adj[k] + (1-mcKwAdjPerc)*(small_baseline_eff[k])

        mcInputKwOutArray = mcPwrOutPerc * maxMotorKw

        mcPercOutArray = np.linspace(0,1,101)
        mcKwOutArray = np.linspace(0,1,101) * maxMotorKw

        mcFullEffArray = np.array([0.0]*len(mcPercOutArray))

        for m in range(1,len(mcPercOutArray)-1):
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
    veh['regenA'] = 500.0
    veh['regenB'] = 0.99

    ### Calculate total vehicle mass
    if veh['vehOverrideKg']==0 or veh['vehOverrideKg']=="":
        if veh['maxEssKwh']==0 or veh['maxEssKw']==0:
            ess_mass_kg = 0.0
        else:
            ess_mass_kg = ((veh['maxEssKwh']*veh['essKgPerKwh'])+veh['essBaseKg'])*veh['compMassMultiplier']
        if veh['maxMotorKw']==0:
            mc_mass_kg = 0.0
        else:
            mc_mass_kg = (veh['mcPeBaseKg']+(veh['mcPeKgPerKw']*veh['maxMotorKw']))*veh['compMassMultiplier']
        if veh['maxFuelConvKw']==0:
            fc_mass_kg = 0.0
        else:
            fc_mass_kg = (((1/veh['fuelConvKwPerKg'])*veh['maxFuelConvKw']+veh['fuelConvBaseKg']))*veh['compMassMultiplier']
        if veh['maxFuelStorKw']==0:
            fs_mass_kg = 0.0
        else:
            fs_mass_kg = ((1/veh['fuelStorKwhPerKg'])*veh['fuelStorKwh'])*veh['compMassMultiplier']
        veh['vehKg'] = veh['cargoKg'] + veh['gliderKg'] + veh['transKg']*veh['compMassMultiplier'] + ess_mass_kg + mc_mass_kg + fc_mass_kg + fs_mass_kg
    else:
        veh['vehKg'] = np.copy( veh['vehOverrideKg'] )

    return veh

def sim_drive( cyc , veh ):

    if veh['vehPtType']==1:

        # If no EV / Hybrid components, no SOC considerations.

        initSoc = 0.0
        output = sim_drive_sub( cyc , veh , initSoc )

    elif veh['vehPtType']==2:

        #####################################
        ### Charge Balancing Vehicle SOC ###
        #####################################

        # Charge balancing SOC for PHEV vehicle types. Iterating initsoc and comparing to final SOC.
        # Iterating until tolerance met or 30 attempts made.

        initSoc = (veh['maxSoc']+veh['minSoc'])/2.0
        ess2fuelKwh = 1.0
        sim_count = 0
        while ess2fuelKwh>veh['essToFuelOkError'] and sim_count<30:
            sim_count += 1
            output = sim_drive_sub( cyc , veh , initSoc )
            ess2fuelKwh = abs( output['ess2fuelKwh'] )
            initSoc = min(1.0,max(0.0,output['final_soc']))
        np.copy( veh['maxSoc'] )
        output = sim_drive_sub( cyc , veh , initSoc )

    elif veh['vehPtType']==3 or veh['vehPtType']==4:

        # If EV, initializing initial SOC to maximum SOC.

        initSoc = np.copy( veh['maxSoc'] )
        output = sim_drive_sub( cyc , veh , initSoc )

    return output

def sim_drive_sub( cyc , veh , initSoc):

    # sim_drive_sub receives second-by-second cycle information,
    # vehicle properties, and an initial state of charge and performs
    # a backward facing powertrain simulation. The function returns an 
    # output dictionary starting at approximately line 1030. Powertrain
    # variables of interest (summary or time-series) can be added to the 
    # output dictionary for reference.
    
    ############################
    ###   Define Constants   ###
    ############################

    airDensityKgPerM3 = 1.2 # Sea level air density at approximately 20C
    gravityMPerSec2 = 9.81
    mphPerMps = 2.2369
    kWhPerGGE = 33.7
    metersPerMile = 1609.00
    maxTracMps2 = ((((veh['wheelCoefOfFric']*veh['driveAxleWeightFrac']*veh['vehKg']*gravityMPerSec2)/(1+((veh['vehCgM']*veh['wheelCoefOfFric'])/veh['wheelBaseM']))))/(veh['vehKg']*gravityMPerSec2))*gravityMPerSec2
    maxRegenKwh = 0.5*veh['vehKg']*(27**2)/(3600*1000)

    #############################
    ### Initialize Variables  ###
    #############################

    ### Drive Cycle
    cycSecs = np.copy( cyc['cycSecs'] )
    cycMps = np.copy( cyc['cycMps'] )
    cycGrade = np.copy( cyc['cycGrade'] )
    cycRoadType = np.copy( cyc['cycRoadType'] )
    cycMph = [x * mphPerMps for x in cyc['cycMps']]
    secs = np.insert(np.diff(cycSecs),0,0)

    ### Component Limits
    curMaxFsKwOut = [0]*len(cycSecs)
    fcTransLimKw = [0]*len(cycSecs)
    fcFsLimKw = [0]*len(cycSecs)
    fcMaxKwIn = [0]*len(cycSecs)
    curMaxFcKwOut = [0]*len(cycSecs)
    essCapLimDischgKw = [0]*len(cycSecs)
    curMaxEssKwOut = [0]*len(cycSecs)
    curMaxAvailElecKw = [0]*len(cycSecs)
    essCapLimChgKw = [0]*len(cycSecs)
    curMaxEssChgKw = [0]*len(cycSecs)
    curMaxRoadwayChgKw = np.interp( cycRoadType, veh['MaxRoadwayChgKw_Roadway'], veh['MaxRoadwayChgKw'] )
    curMaxElecKw = [0]*len(cycSecs)
    mcElecInLimKw = [0]*len(cycSecs)
    mcTransiLimKw = [0]*len(cycSecs)
    curMaxMcKwOut = [0]*len(cycSecs)
    essLimMcRegenPercKw = [0]*len(cycSecs)
    essLimMcRegenKw = [0]*len(cycSecs)
    curMaxMechMcKwIn = [0]*len(cycSecs)
    curMaxTransKwOut = [0]*len(cycSecs)

    ### Drive Train
    cycDragKw = [0]*len(cycSecs)
    cycAccelKw = [0]*len(cycSecs)
    cycAscentKw = [0]*len(cycSecs)
    cycTracKwReq = [0]*len(cycSecs)
    curMaxTracKw = [0]*len(cycSecs)
    spareTracKw = [0]*len(cycSecs)
    cycRrKw = [0]*len(cycSecs)
    cycWheelRadPerSec = [0]*len(cycSecs)
    cycTireInertiaKw = [0]*len(cycSecs)
    cycWheelKwReq = [0]*len(cycSecs)
    regenContrLimKwPerc = [0]*len(cycSecs)
    cycRegenBrakeKw = [0]*len(cycSecs)
    cycFricBrakeKw = [0]*len(cycSecs)
    cycTransKwOutReq = [0]*len(cycSecs)
    cycMet = [0]*len(cycSecs)
    transKwOutAch = [0]*len(cycSecs)
    transKwInAch = [0]*len(cycSecs)
    curSocTarget = [0]*len(cycSecs)
    minMcKw2HelpFc = [0]*len(cycSecs)
    mcMechKwOutAch = [0]*len(cycSecs)
    mcElecKwInAch = [0]*len(cycSecs)
    auxInKw = [0]*len(cycSecs)

    #roadwayMaxEssChg = [0]*len(cycSecs)
    roadwayChgKwOutAch = [0]*len(cycSecs)
    minEssKw2HelpFc = [0]*len(cycSecs)
    essKwOutAch = [0]*len(cycSecs)
    fcKwOutAch = [0]*len(cycSecs)
    fcKwOutAch_pct = [0]*len(cycSecs)
    fcKwInAch = [0]*len(cycSecs)
    fsKwOutAch = [0]*len(cycSecs)
    fsKwhOutAch = [0]*len(cycSecs)
    essCurKwh = [0]*len(cycSecs)
    soc = [0]*len(cycSecs)

    # Vehicle Attributes, Control Variables
    regenBufferSoc = [0]*len(cycSecs)
    essRegenBufferDischgKw = [0]*len(cycSecs)
    maxEssRegenBufferChgKw = [0]*len(cycSecs)
    essAccelBufferChgKw = [0]*len(cycSecs)
    accelBufferSoc = [0]*len(cycSecs)
    maxEssAccelBufferDischgKw = [0]*len(cycSecs)
    essAccelRegenDischgKw = [0]*len(cycSecs)
    mcElectInKwForMaxFcEff = [0]*len(cycSecs)
    electKwReq4AE = [0]*len(cycSecs)
    canPowerAllElectrically = [0]*len(cycSecs)
    desiredEssKwOutForAE = [0]*len(cycSecs)
    essAEKwOut = [0]*len(cycSecs)
    erAEKwOut = [0]*len(cycSecs)
    essDesiredKw4FcEff = [0]*len(cycSecs)
    essKwIfFcIsReq = [0]*len(cycSecs)
    curMaxMcElecKwIn = [0]*len(cycSecs)
    fcKwGapFrEff = [0]*len(cycSecs)
    erKwIfFcIsReq = [0]*len(cycSecs)
    mcElecKwInIfFcIsReq = [0]*len(cycSecs)
    mcKwIfFcIsReq = [0]*len(cycSecs)
    fcForcedOn = np.full(len(cycSecs),False)
    fcForcedState = [0]*len(cycSecs)
    mcMechKw4ForcedFc = [0]*len(cycSecs)
    fcTimeOn = [0]*len(cycSecs)
    prevfcTimeOn = [0]*len(cycSecs)

    ### Additional Variables
    mpsAch = [0]*len(cycSecs)
    mphAch = [0]*len(cycSecs)
    distMeters = [0]*len(cycSecs)
    distMiles = [0]*len(cycSecs)
    highAccFcOnTag = [0]*len(cycSecs)
    reachedBuff = [0]*len(cycSecs)
    maxTracMps = [0]*len(cycSecs)
    addKwh = [0]*len(cycSecs)
    dodCycs = [0]*len(cycSecs)
    essPercDeadArray = [0]*len(cycSecs)
    dragKw = [0]*len(cycSecs)
    essLossKw = [0]*len(cycSecs)
    accelKw = [0]*len(cycSecs)
    ascentKw = [0]*len(cycSecs)
    rrKw = [0]*len(cycSecs)
    motor_index_debug = [0]*len(cycSecs)
    debug_flag = [0]*len(cycSecs)


    ############################
    ###  Assign First Value  ###
    ############################

    ### Drive Train
    cycMet[0] = 1
    curSocTarget[0] = veh['maxSoc']
    essCurKwh[0] = initSoc*veh['maxEssKwh']
    soc[0] = initSoc

    ############################
    ###   Loop Through Time  ###
    ############################

    for i in range(1,len(cycSecs)):

        ### Misc calcs
        if veh['noElecAux']=='TRUE':
            auxInKw[i] = veh['auxKw']/veh['altEff']
        else:
            auxInKw[i] = veh['auxKw']

        if soc[i-1]<(veh['minSoc']+veh['percHighAccBuf']):
            reachedBuff[i] = 0
        else:
            reachedBuff[i] = 1
        if soc[i-1]<veh['minSoc'] or (highAccFcOnTag[i-1]==1 and reachedBuff[i]==0):
            highAccFcOnTag[i] = 1
        else:
            highAccFcOnTag[i] = 0
        maxTracMps[i] = mpsAch[i-1]+(maxTracMps2*secs[i])

        ### Component Limits
        curMaxFsKwOut[i] = min( veh['maxFuelStorKw'] , fsKwOutAch[i-1] + ((veh['maxFuelStorKw']/veh['fuelStorSecsToPeakPwr'])*(secs[i])))
        fcTransLimKw[i] = fcKwOutAch[i-1] + ((veh['maxFuelConvKw']/veh['fuelConvSecsToPeakPwr'])*(secs[i]))

        fcMaxKwIn[i] = min(curMaxFsKwOut[i], veh['maxFuelStorKw'])
        fcFsLimKw[i] = veh['fcMaxOutkW']
        curMaxFcKwOut[i] = min(veh['maxFuelConvKw'],fcFsLimKw[i],fcTransLimKw[i])

        if veh['maxEssKwh']==0 or soc[i-1]<veh['minSoc']:
            essCapLimDischgKw[i] = 0.0

        else:
            essCapLimDischgKw[i] = (veh['maxEssKwh']*np.sqrt(veh['essRoundTripEff']))*3600.0*(soc[i-1]-veh['minSoc'])/(secs[i])
        curMaxEssKwOut[i] = min(veh['maxEssKw'],essCapLimDischgKw[i])

        if  veh['maxEssKwh'] == 0 or veh['maxEssKw'] == 0:
            essCapLimChgKw[i] = 0

        else:
            essCapLimChgKw[i] = max(((veh['maxSoc']-soc[i-1])*veh['maxEssKwh']*(1/np.sqrt(veh['essRoundTripEff'])))/((secs[i])*(1/3600.0)),0)

        curMaxEssChgKw[i] = min(essCapLimChgKw[i],veh['maxEssKw'])

        if veh['fcEffType']==4:
            curMaxElecKw[i] = curMaxFcKwOut[i]+curMaxRoadwayChgKw[i]+curMaxEssKwOut[i]-auxInKw[i]

        else:
            curMaxElecKw[i] = curMaxRoadwayChgKw[i]+curMaxEssKwOut[i]-auxInKw[i]

        curMaxAvailElecKw[i] = min(curMaxElecKw[i], veh['mcMaxElecInKw'])

        if curMaxElecKw[i]>0:
            if curMaxAvailElecKw[i] == max(veh['mcKwInArray']):
                mcElecInLimKw[i] = min(veh['mcKwOutArray'][len(veh['mcKwOutArray'])-1],veh['maxMotorKw'])
            else:
                mcElecInLimKw[i] = min(veh['mcKwOutArray'][np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,curMaxAvailElecKw[i]))-1],veh['maxMotorKw'])
        else:
            mcElecInLimKw[i] = 0.0

        mcTransiLimKw[i] = abs(mcMechKwOutAch[i-1])+((veh['maxMotorKw']/veh['motorSecsToPeakPwr'])*(secs[i]))
        curMaxMcKwOut[i] = max(min(mcElecInLimKw[i],mcTransiLimKw[i],veh['maxMotorKw']),-veh['maxMotorKw'])

        if curMaxMcKwOut[i] == 0:
            curMaxMcElecKwIn[i] = 0
        else:
            if curMaxMcKwOut[i] == veh['maxMotorKw']:
                curMaxMcElecKwIn[i] = curMaxMcKwOut[i]/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                curMaxMcElecKwIn[i] = curMaxMcKwOut[i]/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,curMaxMcKwOut[i]))-1)]

        if veh['maxMotorKw']==0:
            essLimMcRegenPercKw[i] = 0.0

        else:
            essLimMcRegenPercKw[i] = min((curMaxEssChgKw[i]+auxInKw[i])/veh['maxMotorKw'],1)
        if curMaxEssChgKw[i]==0:
            essLimMcRegenKw[i] = 0.0

        else:
            if veh['maxMotorKw'] == curMaxEssChgKw[i]-curMaxRoadwayChgKw[i]:
                essLimMcRegenKw[i] = min(veh['maxMotorKw'],curMaxEssChgKw[i]/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1])
            else:
                essLimMcRegenKw[i] = min(veh['maxMotorKw'],curMaxEssChgKw[i]/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,curMaxEssChgKw[i]-curMaxRoadwayChgKw[i]))-1)])

        curMaxMechMcKwIn[i] = min(essLimMcRegenKw[i],veh['maxMotorKw'])
        curMaxTracKw[i] = (((veh['wheelCoefOfFric']*veh['driveAxleWeightFrac']*veh['vehKg']*gravityMPerSec2)/(1+((veh['vehCgM']*veh['wheelCoefOfFric'])/veh['wheelBaseM'])))/1000.0)*(maxTracMps[i])

        if veh['fcEffType']==4:

            if veh['noElecSys']=='TRUE' or veh['noElecAux']=='TRUE' or highAccFcOnTag[i]==1:
                curMaxTransKwOut[i] = min((curMaxMcKwOut[i]-auxInKw[i])*veh['transEff'],curMaxTracKw[i]/veh['transEff'])
                debug_flag[i] = 1

            else:
                curMaxTransKwOut[i] = min((curMaxMcKwOut[i]-min(curMaxElecKw[i],0))*veh['transEff'],curMaxTracKw[i]/veh['transEff'])
                debug_flag[i] = 2

        else:

            if veh['noElecSys']=='TRUE' or veh['noElecAux']=='TRUE' or highAccFcOnTag[i]==1:
                curMaxTransKwOut[i] = min((curMaxMcKwOut[i]+curMaxFcKwOut[i]-auxInKw[i])*veh['transEff'],curMaxTracKw[i]/veh['transEff'])
                debug_flag[i] = 3

            else:
                curMaxTransKwOut[i] = min((curMaxMcKwOut[i]+curMaxFcKwOut[i]-min(curMaxElecKw[i],0))*veh['transEff'],curMaxTracKw[i]/veh['transEff'])
                debug_flag[i] = 4

        ### Cycle Power
        cycDragKw[i] = 0.5*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*(((mpsAch[i-1]+cycMps[i])/2.0)**3)/1000.0
        cycAccelKw[i] = (veh['vehKg']/(2.0*(secs[i])))*((cycMps[i]**2)-(mpsAch[i-1]**2))/1000.0
        cycAscentKw[i] = gravityMPerSec2*np.sin(np.arctan(cycGrade[i]))*veh['vehKg']*((mpsAch[i-1]+cycMps[i])/2.0)/1000.0
        cycTracKwReq[i] = cycDragKw[i] + cycAccelKw[i] + cycAscentKw[i]
        spareTracKw[i] = curMaxTracKw[i]-cycTracKwReq[i]
        cycRrKw[i] = gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']*((mpsAch[i-1]+cycMps[i])/2.0)/1000.0
        cycWheelRadPerSec[i] = cycMps[i]/veh['wheelRadiusM']
        cycTireInertiaKw[i] = (((0.5)*veh['wheelInertiaKgM2']*(veh['numWheels']*(cycWheelRadPerSec[i]**2.0))/secs[i])-((0.5)*veh['wheelInertiaKgM2']*(veh['numWheels']*((mpsAch[i-1]/veh['wheelRadiusM'])**2.0))/secs[i]))/1000.0

        cycWheelKwReq[i] = cycTracKwReq[i] + cycRrKw[i] + cycTireInertiaKw[i]
        regenContrLimKwPerc[i] = veh['maxRegen']/(1+veh['regenA']*np.exp(-veh['regenB']*((cycMph[i]+mpsAch[i-1]*mphPerMps)/2.0+1-0)))
        cycRegenBrakeKw[i] = max(min(curMaxMechMcKwIn[i]*veh['transEff'],regenContrLimKwPerc[i]*-cycWheelKwReq[i]),0)
        cycFricBrakeKw[i] = -min(cycRegenBrakeKw[i]+cycWheelKwReq[i],0)
        cycTransKwOutReq[i] = cycWheelKwReq[i] + cycFricBrakeKw[i]

        if cycTransKwOutReq[i]<=curMaxTransKwOut[i]:
            cycMet[i] = 1
            transKwOutAch[i] = cycTransKwOutReq[i]

        else:
            cycMet[i] = -1
            transKwOutAch[i] = curMaxTransKwOut[i]

        ################################
        ###   Speed/Distance Calcs   ###
        ################################

        #Cycle is met
        if cycMet[i]==1:
            mpsAch[i] = cycMps[i]

        #Cycle is not met
        else:
            Drag3 = (1.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']
            Accel2 = veh['vehKg']/(2.0*(secs[i]))
            Drag2 = (3.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*mpsAch[i-1]
            Wheel2 = 0.5*veh['wheelInertiaKgM2']*veh['numWheels']/(secs[i]*(veh['wheelRadiusM']**2))
            Drag1 = (3.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*((mpsAch[i-1])**2)
            Roll1 = (gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']/2.0)
            Ascent1 = (gravityMPerSec2*np.sin(np.arctan(cycGrade[i]))*veh['vehKg']/2.0)
            Accel0 = -(veh['vehKg']*((mpsAch[i-1])**2))/(2.0*(secs[i]))
            Drag0 = (1.0/16.0)*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*((mpsAch[i-1])**3)
            Roll0 = (gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']*mpsAch[i-1]/2.0)
            Ascent0 = (gravityMPerSec2*np.sin(np.arctan(cycGrade[i]))*veh['vehKg']*mpsAch[i-1]/2.0)
            Wheel0 = -((0.5*veh['wheelInertiaKgM2']*veh['numWheels']*(mpsAch[i-1]**2))/(secs[i]*(veh['wheelRadiusM']**2)))

            Total3 = Drag3/1e3
            Total2 = (Accel2+Drag2+Wheel2)/1e3
            Total1 = (Drag1+Roll1+Ascent1)/1e3
            Total0 = (Accel0+Drag0+Roll0+Ascent0+Wheel0)/1e3 - curMaxTransKwOut[i]

            Total = [Total3,Total2,Total1,Total0]
            Total_roots = np.roots(Total)
            ind = np.argmin( abs(cycMps[i] - Total_roots) )
            mpsAch[i] = Total_roots[ind]

        mphAch[i] = mpsAch[i]*mphPerMps
        distMeters[i] = mpsAch[i]*secs[i]
        distMiles[i] = distMeters[i]*(1.0/metersPerMile)

        ### Drive Train
        if transKwOutAch[i]>0:
            transKwInAch[i] = transKwOutAch[i]/veh['transEff']
        else:
            transKwInAch[i] = transKwOutAch[i]*veh['transEff']

        if cycMet[i]==1:

            if veh['fcEffType']==4:
                minMcKw2HelpFc[i] = max(transKwInAch[i], -curMaxMechMcKwIn[i])

            else:
                minMcKw2HelpFc[i] = max(transKwInAch[i] - curMaxFcKwOut[i], -curMaxMechMcKwIn[i])
        else:
            minMcKw2HelpFc[i] = max(curMaxMcKwOut[i], -curMaxMechMcKwIn[i])

        if veh['noElecSys'] == 'TRUE':
           regenBufferSoc[i] = 0

        elif veh['chargingOn']:
           regenBufferSoc[i] = max(veh['maxSoc'] - (maxRegenKwh/veh['maxEssKwh']), (veh['maxSoc']+veh['minSoc'])/2)

        else:
           regenBufferSoc[i] = max(((veh['maxEssKwh']*veh['maxSoc'])-(0.5*veh['vehKg']*(cycMps[i]**2)*(1.0/1000)*(1.0/3600)*veh['motorPeakEff']*veh['maxRegen']))/veh['maxEssKwh'],veh['minSoc'])

        essRegenBufferDischgKw[i] = min(curMaxEssKwOut[i], max(0,(soc[i-1]-regenBufferSoc[i])*veh['maxEssKwh']*3600/secs[i]))

        maxEssRegenBufferChgKw[i] = min(max(0,(regenBufferSoc[i]-soc[i-1])*veh['maxEssKwh']*3600.0/secs[i]),curMaxEssChgKw[i])

        if veh['noElecSys']=='TRUE':
           accelBufferSoc[i] = 0

        else:
           accelBufferSoc[i] = min(max((((((((veh['maxAccelBufferMph']*(1/mphPerMps))**2))-((cycMps[i]**2)))/(((veh['maxAccelBufferMph']*(1/mphPerMps))**2)))*(min(veh['maxAccelBufferPercOfUseableSoc']*(veh['maxSoc']-veh['minSoc']),maxRegenKwh/veh['maxEssKwh'])*veh['maxEssKwh']))/veh['maxEssKwh'])+veh['minSoc'],veh['minSoc']), veh['maxSoc'])

        essAccelBufferChgKw[i] = max(0,(accelBufferSoc[i] - soc[i-1])*veh['maxEssKwh']*3600.0/secs[i])
        maxEssAccelBufferDischgKw[i] = min(max(0, (soc[i-1]-accelBufferSoc[i])*veh['maxEssKwh']*3600/secs[i]),curMaxEssKwOut[i])

        if regenBufferSoc[i] < accelBufferSoc[i]:
            essAccelRegenDischgKw[i] = max(min(((soc[i-1]-(regenBufferSoc[i]+accelBufferSoc[i])/2)*veh['maxEssKwh']*3600.0)/secs[i],curMaxEssKwOut[i]),-curMaxEssChgKw[i])

        elif soc[i-1]>regenBufferSoc[i]:
            essAccelRegenDischgKw[i] = max(min(essRegenBufferDischgKw[i],curMaxEssKwOut[i]),-curMaxEssChgKw[i])

        elif soc[i-1]<accelBufferSoc[i]:
            essAccelRegenDischgKw[i] = max(min(-1.0*essAccelBufferChgKw[i],curMaxEssKwOut[i]),-curMaxEssChgKw[i])

        else:
            essAccelRegenDischgKw[i] = max(min(0,curMaxEssKwOut[i]),-curMaxEssChgKw[i])

        fcKwGapFrEff[i] = abs(transKwOutAch[i]-veh['maxFcEffKw'])

        if veh['noElecSys']=='TRUE':
            mcElectInKwForMaxFcEff[i] = 0

        elif transKwOutAch[i]<veh['maxFcEffKw']:

            if fcKwGapFrEff[i] == veh['maxMotorKw']:
                mcElectInKwForMaxFcEff[i] = fcKwGapFrEff[i]/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]*-1
            else:
                mcElectInKwForMaxFcEff[i] = fcKwGapFrEff[i]/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,fcKwGapFrEff[i]))-1)]*-1

        else:

            if fcKwGapFrEff[i] == veh['maxMotorKw']:
                mcElectInKwForMaxFcEff[i] = veh['mcKwInArray'][len(veh['mcKwInArray'])-1]
            else:
                mcElectInKwForMaxFcEff[i] = veh['mcKwInArray'][np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,fcKwGapFrEff[i]))-1]

        if veh['noElecSys']=='TRUE':
            electKwReq4AE[i] = 0

        elif transKwInAch[i] > 0:
            if transKwInAch[i] == veh['maxMotorKw']:
        
                electKwReq4AE[i] = transKwInAch[i]/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]+auxInKw[i]
            else:
                electKwReq4AE[i] = transKwInAch[i]/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,transKwInAch[i]))-1)]+auxInKw[i]

        else:
           electKwReq4AE[i] = 0

        prevfcTimeOn[i] = fcTimeOn[i-1]

        if veh['maxFuelConvKw']==0:
            canPowerAllElectrically[i] = accelBufferSoc[i]<soc[i-1] and transKwInAch[i]<=curMaxMcKwOut[i] and (electKwReq4AE[i]<curMaxElecKw[i] or veh['maxFuelConvKw']==0)

        else:
            canPowerAllElectrically[i] = accelBufferSoc[i]<soc[i-1] and transKwInAch[i]<=curMaxMcKwOut[i] and (electKwReq4AE[i]<curMaxElecKw[i] or veh['maxFuelConvKw']==0) and (cycMph[i]-0.00001<=veh['mphFcOn'] or veh['chargingOn']) and electKwReq4AE[i]<=veh['kwDemandFcOn']

        if canPowerAllElectrically[i]:

            if transKwInAch[i]<+auxInKw[i]:
                desiredEssKwOutForAE[i] = auxInKw[i]+transKwInAch[i]

            elif regenBufferSoc[i]<accelBufferSoc[i]:
                desiredEssKwOutForAE[i] = essAccelRegenDischgKw[i]

            elif soc[i-1]>regenBufferSoc[i]:
                desiredEssKwOutForAE[i] = essRegenBufferDischgKw[i]

            elif soc[i-1]<accelBufferSoc[i]:
                desiredEssKwOutForAE[i] = -essAccelBufferChgKw[i]

            else:
                desiredEssKwOutForAE[i] = transKwInAch[i]+auxInKw[i]-curMaxRoadwayChgKw[i]

        else:
            desiredEssKwOutForAE[i] = 0

        if canPowerAllElectrically[i]:
            essAEKwOut[i] = max(-curMaxEssChgKw[i],-maxEssRegenBufferChgKw[i],min(0,curMaxRoadwayChgKw[i]-(transKwInAch[i]+auxInKw[i])),min(curMaxEssKwOut[i],desiredEssKwOutForAE[i]))

        else:
            essAEKwOut[i] = 0

        erAEKwOut[i] = min(max(0,transKwInAch[i]+auxInKw[i]-essAEKwOut[i]),curMaxRoadwayChgKw[i])

        if prevfcTimeOn[i]>0 and prevfcTimeOn[i]<veh['minFcTimeOn']-secs[i]:
            fcForcedOn[i] = True
        else:
            fcForcedOn[i] = False

        if fcForcedOn[i]==False or canPowerAllElectrically[i]==False:
            fcForcedState[i] = 1
            mcMechKw4ForcedFc[i] = 0

        elif transKwInAch[i]<0:
            fcForcedState[i] = 2
            mcMechKw4ForcedFc[i] = transKwInAch[i]

        elif veh['maxFcEffKw']==transKwInAch[i]:
            fcForcedState[i] = 3
            mcMechKw4ForcedFc[i] = 0

        elif veh['idleFcKw'] > transKwInAch[i] and cycAccelKw[i] >=0:
            fcForcedState[i] = 4
            mcMechKw4ForcedFc[i] = transKwInAch[i] - veh['idleFcKw']

        elif veh['maxFcEffKw']>transKwInAch[i]:
            fcForcedState[i] = 5
            mcMechKw4ForcedFc[i] = 0

        else:
            fcForcedState[i] = 6
            mcMechKw4ForcedFc[i] = transKwInAch[i] - veh['maxFcEffKw']

        if (-mcElectInKwForMaxFcEff[i]-curMaxRoadwayChgKw[i])>0:
            essDesiredKw4FcEff[i] = (-mcElectInKwForMaxFcEff[i]-curMaxRoadwayChgKw[i]) * veh['essDischgToFcMaxEffPerc']

        else:
            essDesiredKw4FcEff[i] = (-mcElectInKwForMaxFcEff[i]-curMaxRoadwayChgKw[i]) * veh['essChgToFcMaxEffPerc']

        if accelBufferSoc[i]>regenBufferSoc[i]:
            essKwIfFcIsReq[i] = min(curMaxEssKwOut[i],veh['mcMaxElecInKw']+auxInKw[i],curMaxMcElecKwIn[i]+auxInKw[i], max(-curMaxEssChgKw[i], essAccelRegenDischgKw[i]))

        elif essRegenBufferDischgKw[i]>0:
            essKwIfFcIsReq[i] = min(curMaxEssKwOut[i],veh['mcMaxElecInKw']+auxInKw[i],curMaxMcElecKwIn[i]+auxInKw[i], max(-curMaxEssChgKw[i], min(essAccelRegenDischgKw[i],mcElecInLimKw[i]+auxInKw[i], max(essRegenBufferDischgKw[i],essDesiredKw4FcEff[i]))))

        elif essAccelBufferChgKw[i]>0:
            essKwIfFcIsReq[i] = min(curMaxEssKwOut[i],veh['mcMaxElecInKw']+auxInKw[i],curMaxMcElecKwIn[i]+auxInKw[i], max(-curMaxEssChgKw[i], max(-1*maxEssRegenBufferChgKw[i], min(-essAccelBufferChgKw[i],essDesiredKw4FcEff[i]))))


        elif essDesiredKw4FcEff[i]>0:
            essKwIfFcIsReq[i] = min(curMaxEssKwOut[i],veh['mcMaxElecInKw']+auxInKw[i],curMaxMcElecKwIn[i]+auxInKw[i], max(-curMaxEssChgKw[i], min(essDesiredKw4FcEff[i],maxEssAccelBufferDischgKw[i])))

        else:
            essKwIfFcIsReq[i] = min(curMaxEssKwOut[i],veh['mcMaxElecInKw']+auxInKw[i],curMaxMcElecKwIn[i]+auxInKw[i], max(-curMaxEssChgKw[i], max(essDesiredKw4FcEff[i],-maxEssRegenBufferChgKw[i])))

        erKwIfFcIsReq[i] = max(0,min(curMaxRoadwayChgKw[i],curMaxMechMcKwIn[i],essKwIfFcIsReq[i]-mcElecInLimKw[i]+auxInKw[i]))

        mcElecKwInIfFcIsReq[i] = essKwIfFcIsReq[i]+erKwIfFcIsReq[i]-auxInKw[i]

        if veh['noElecSys']=='TRUE':
            mcKwIfFcIsReq[i] = 0

        elif  mcElecKwInIfFcIsReq[i] == 0:
            mcKwIfFcIsReq[i] = 0

        elif mcElecKwInIfFcIsReq[i] > 0:

            if mcElecKwInIfFcIsReq[i] == max(veh['mcKwInArray']):
                 mcKwIfFcIsReq[i] = mcElecKwInIfFcIsReq[i]*veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                 mcKwIfFcIsReq[i] = mcElecKwInIfFcIsReq[i]*veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,mcElecKwInIfFcIsReq[i]))-1)]

        else:
            if mcElecKwInIfFcIsReq[i]*-1 == max(veh['mcKwInArray']):
                mcKwIfFcIsReq[i] = mcElecKwInIfFcIsReq[i]/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                mcKwIfFcIsReq[i] = mcElecKwInIfFcIsReq[i]/(veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,mcElecKwInIfFcIsReq[i]*-1))-1)])

        if veh['maxMotorKw']==0:
            mcMechKwOutAch[i] = 0

        elif fcForcedOn[i]==True and canPowerAllElectrically[i]==True and (veh['vehPtType']==2.0 or veh['vehPtType']==3.0) and veh['fcEffType']!=4:
           mcMechKwOutAch[i] =  mcMechKw4ForcedFc[i]

        elif transKwInAch[i]<=0:

            if veh['fcEffType']!=4 and veh['maxFuelConvKw']> 0:
                if canPowerAllElectrically[i] == 1:
                    mcMechKwOutAch[i] = -min(curMaxMechMcKwIn[i],-transKwInAch[i])
                else:
                    mcMechKwOutAch[i] = min(-min(curMaxMechMcKwIn[i], -transKwInAch[i]),max(-curMaxFcKwOut[i], mcKwIfFcIsReq[i]))
            else:
                mcMechKwOutAch[i] = min(-min(curMaxMechMcKwIn[i],-transKwInAch[i]),-transKwInAch[i])

        elif canPowerAllElectrically[i] == 1:
            mcMechKwOutAch[i] = transKwInAch[i]

        else:
            mcMechKwOutAch[i] = max(minMcKw2HelpFc[i],mcKwIfFcIsReq[i])

        if mcMechKwOutAch[i]==0:
            mcElecKwInAch[i] = 0.0
            motor_index_debug[i] = 0

        elif mcMechKwOutAch[i]<0:

            if mcMechKwOutAch[i]*-1 == max(veh['mcKwInArray']):
                mcElecKwInAch[i] = mcMechKwOutAch[i]*veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                mcElecKwInAch[i] = mcMechKwOutAch[i]*veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwInArray']>min(max(veh['mcKwInArray'])-0.01,mcMechKwOutAch[i]*-1))-1)]

        else:
            if veh['maxMotorKw'] == mcMechKwOutAch[i]:
                mcElecKwInAch[i] = mcMechKwOutAch[i]/veh['mcFullEffArray'][len(veh['mcFullEffArray'])-1]
            else:
                mcElecKwInAch[i] = mcMechKwOutAch[i]/veh['mcFullEffArray'][max(1,np.argmax(veh['mcKwOutArray']>min(veh['maxMotorKw']-0.01,mcMechKwOutAch[i]))-1)]

        if curMaxRoadwayChgKw[i] == 0:
            roadwayChgKwOutAch[i] = 0

        elif veh['fcEffType']==4:
            roadwayChgKwOutAch[i] = max(0, mcElecKwInAch[i], maxEssRegenBufferChgKw[i], essRegenBufferDischgKw[i], curMaxRoadwayChgKw[i])

        elif canPowerAllElectrically[i] == 1:
            roadwayChgKwOutAch[i] = erAEKwOut[i]

        else:
            roadwayChgKwOutAch[i] = erKwIfFcIsReq[i]

        minEssKw2HelpFc[i] = mcElecKwInAch[i] + auxInKw[i] - curMaxFcKwOut[i] - roadwayChgKwOutAch[i]

        if veh['maxEssKw'] == 0 or veh['maxEssKwh'] == 0:
            essKwOutAch[i]  = 0

        elif veh['fcEffType']==4:

            if transKwOutAch[i]>=0:
                essKwOutAch[i] = min(max(minEssKw2HelpFc[i],essDesiredKw4FcEff[i],essAccelRegenDischgKw[i]),curMaxEssKwOut[i],mcElecKwInAch[i]+auxInKw[i]-roadwayChgKwOutAch[i])

            else:
                essKwOutAch[i] = mcElecKwInAch[i]+auxInKw[i]-roadwayChgKwOutAch[i]

        elif highAccFcOnTag[i]==1 or veh['noElecAux']=='TRUE':
            essKwOutAch[i] = mcElecKwInAch[i]-roadwayChgKwOutAch[i]

        else:
            essKwOutAch[i] = mcElecKwInAch[i]+auxInKw[i]-roadwayChgKwOutAch[i]

        if veh['maxFuelConvKw']==0:
            fcKwOutAch[i] = 0

        elif veh['fcEffType']==4:
            fcKwOutAch[i] = min(curMaxFcKwOut[i], max(0, mcElecKwInAch[i]+auxInKw[i]-essKwOutAch[i]-roadwayChgKwOutAch[i]))

        elif veh['noElecSys']=='TRUE' or veh['noElecAux']=='TRUE' or highAccFcOnTag[i]==1:
            fcKwOutAch[i] = min(curMaxFcKwOut[i], max(0, transKwInAch[i]-mcMechKwOutAch[i]+auxInKw[i]))

        else:
            fcKwOutAch[i] = min(curMaxFcKwOut[i], max(0, transKwInAch[i]-mcMechKwOutAch[i]))

        if fcKwOutAch[i]==0:
            fcKwInAch[i] = 0.0
            fcKwOutAch_pct[i] = 0

        if veh['maxFuelConvKw'] == 0:
            fcKwOutAch_pct[i] = 0
        else:
            fcKwOutAch_pct[i] = fcKwOutAch[i] / veh['maxFuelConvKw']

        if fcKwOutAch[i] == 0:
            fcKwInAch[i] = 0
        else:
            if fcKwOutAch[i] == veh['fcMaxOutkW']:
                fcKwInAch[i] = fcKwOutAch[i]/veh['fcEffArray'][len(veh['fcEffArray'])-1]
            else:
                fcKwInAch[i] = fcKwOutAch[i]/(veh['fcEffArray'][max(1,np.argmax(veh['fcKwOutArray']>min(fcKwOutAch[i],veh['fcMaxOutkW']-0.001))-1)])

        fsKwOutAch[i] = np.copy( fcKwInAch[i] )

        fsKwhOutAch[i] = fsKwOutAch[i]*secs[i]*(1/3600.0)


        if veh['noElecSys']=='TRUE':
            essCurKwh[i] = 0

        elif essKwOutAch[i]<0:
            essCurKwh[i] = essCurKwh[i-1]-essKwOutAch[i]*(secs[i]/3600.0)*np.sqrt(veh['essRoundTripEff'])

        else:
            essCurKwh[i] = essCurKwh[i-1]-essKwOutAch[i]*(secs[i]/3600.0)*(1/np.sqrt(veh['essRoundTripEff']))

        if veh['maxEssKwh']==0:
            soc[i] = 0.0

        else:
            soc[i] = essCurKwh[i]/veh['maxEssKwh']

        if canPowerAllElectrically[i]==True and fcForcedOn[i]==False and fcKwOutAch[i]==0:
            fcTimeOn[i] = 0
        else:
            fcTimeOn[i] = fcTimeOn[i-1] + secs[i]

        ### Battery wear calcs

        if veh['noElecSys']!='TRUE':

            if essCurKwh[i]>essCurKwh[i-1]:
                addKwh[i] = (essCurKwh[i]-essCurKwh[i-1]) + addKwh[i-1]
            else:
                addKwh[i] = 0

            if addKwh[i]==0:
                dodCycs[i] = addKwh[i-1]/veh['maxEssKwh']
            else:
                dodCycs[i] = 0

            if dodCycs[i]!=0:
                essPercDeadArray[i] = np.power(veh['essLifeCoefA'],1.0/veh['essLifeCoefB']) / np.power(dodCycs[i],1.0/veh['essLifeCoefB'])
            else:
                essPercDeadArray[i] = 0

        ### Energy Audit Calculations
        dragKw[i] = 0.5*airDensityKgPerM3*veh['dragCoef']*veh['frontalAreaM2']*(((mpsAch[i-1]+mpsAch[i])/2.0)**3)/1000.0
        if veh['maxEssKw'] == 0 or veh['maxEssKwh']==0:
            essLossKw[i] = 0
        elif essKwOutAch[i]<0:
            essLossKw[i] = -essKwOutAch[i] - (-essKwOutAch[i]*np.sqrt(veh['essRoundTripEff']))
        else:
            essLossKw[i] = essKwOutAch[i]*(1.0/np.sqrt(veh['essRoundTripEff']))-essKwOutAch[i]
        accelKw[i] = (veh['vehKg']/(2.0*(secs[i])))*((mpsAch[i]**2)-(mpsAch[i-1]**2))/1000.0
        ascentKw[i] = gravityMPerSec2*np.sin(np.arctan(cycGrade[i]))*veh['vehKg']*((mpsAch[i-1]+mpsAch[i])/2.0)/1000.0
        rrKw[i] = gravityMPerSec2*veh['wheelRrCoef']*veh['vehKg']*((mpsAch[i-1]+mpsAch[i])/2.0)/1000.0

    ############################################
    ### Calculate Results and Assign Outputs ###
    ############################################

    output = dict()

    if sum(fsKwhOutAch) == 0:
        output['mpgge'] = 0

    else:
        output['mpgge'] = sum(distMiles)/(sum(fsKwhOutAch)*(1/kWhPerGGE))

    roadwayChgKj = sum(roadwayChgKwOutAch*secs)
    essDischKj = -(soc[-1]-initSoc)*veh['maxEssKwh']*3600.0
    output['battery_kWh_per_mi'] = (essDischKj/3600.0) / sum(distMiles)
    output['electric_kWh_per_mi'] = ((roadwayChgKj+essDischKj)/3600.0) / sum(distMiles)
    output['maxTraceMissMph'] = mphPerMps*max(abs(cycMps-mpsAch))
    fuelKj = sum(np.asarray(fsKwOutAch)*np.asarray(secs))
    roadwayChgKj = sum(np.asarray(roadwayChgKwOutAch)*np.asarray(secs))
    essDischgKj = -(soc[-1]-initSoc)*veh['maxEssKwh']*3600.0

    if (fuelKj+roadwayChgKj)==0:
        output['ess2fuelKwh'] = 1.0

    else:
        output['ess2fuelKwh'] = essDischgKj/(fuelKj+roadwayChgKj)

    output['initial_soc'] = soc[0]
    output['final_soc'] = soc[-1]


    if output['mpgge'] == 0:
        Gallons_gas_equivalent_per_mile = output['electric_kWh_per_mi']/33.7

    else:
         Gallons_gas_equivalent_per_mile = 1/output['mpgge'] + output['electric_kWh_per_mi']/33.7

    output['mpgge_elec'] = 1/Gallons_gas_equivalent_per_mile
    output['soc'] = np.asarray(soc)
    output['distance_mi'] = sum(distMiles)
    duration_sec = cycSecs[-1]-cycSecs[0]
    output['avg_speed_mph'] = sum(distMiles) / (duration_sec/3600.0)
    accel = np.diff(mphAch) / np.diff(cycSecs)
    output['avg_accel_mphps'] = np.mean(accel[accel>0])

    if max(mphAch)>60:
        output['ZeroToSixtyTime_secs'] = np.interp(60,mphAch,cycSecs)

    else:
        output['ZeroToSixtyTime_secs'] = 0.0

    #######################################################################
    ####  Time series information for additional analysis / debugging. ####
    ####             Add parameters of interest as needed.             ####
    #######################################################################

    output['fcKwOutAch'] = np.asarray(fcKwOutAch)
    output['fsKwhOutAch'] = np.asarray(fsKwhOutAch)
    output['fcKwInAch'] = np.asarray(fcKwInAch)
    output['time'] = np.asarray(cycSecs)

    return output
