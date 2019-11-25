import pandas as pd
from collections import namedtuple
import numpy as np
import re

def get_standard_cycle(cycle_name):
    """Load time trace of speed, grade, and road type."""
    csv_path = '..//cycles//'+cycle_name+'.csv'
    data = pd.read_csv(csv_path)
    return data

def get_veh(vnum):
    """Load vehicle parameters and assign to namedtuple 'veh'."""

    vehdf = pd.read_csv('..//docs//FASTSim_py_veh_db.csv')
    vehdf.set_index('Selection', inplace=True, drop=False)

    ### selects specified vnum from vehdf
    veh = dict()
    for col in vehdf.columns:
        # convert all data to string types
        vehdf.loc[vnum, col] = str(vehdf.loc[vnum, col])
        # remove percent signs if any are found
        if vehdf.loc[vnum, col].find('%') != -1:
            vehdf.loc[vnum, col] = vehdf.loc[vnum, col].replace('%', '')
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
    if veh['noElecSys'] == 'TRUE' or veh['maxMotorKw'] <= veh['auxKw'] or veh['forceAuxOnFC'] == 'TRUE':
        veh['noElecAux'] = 'TRUE'

    else:
        veh['noElecAux'] = 'FALSE'

    # Copying vehPtType to additional key
    veh['vehTypeSelection'] = np.copy(veh['vehPtType'])
    # to be consistent with Excel version but not used in Python version

    ### Defining Fuel Converter efficiency curve as lookup table for %power_in vs power_out
    ### see "FC Model" tab in FASTSim for Excel

    if veh['maxFuelConvKw'] > 0:

        # Discrete power out percentages for assigning FC efficiencies
        fcPwrOutPerc = np.array(
            [0, 0.005, 0.015, 0.04, 0.06, 0.10, 0.14, 0.20, 0.40, 0.60, 0.80, 1.00])

        # Efficiencies at different power out percentages by FC type
        eff_si = np.array([0.10, 0.12, 0.16, 0.22, 0.28, 0.33,
                           0.35, 0.36, 0.35, 0.34, 0.32, 0.30])
        eff_atk = np.array([0.10, 0.12, 0.28, 0.35, 0.375,
                            0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35])
        eff_diesel = np.array(
            [0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])
        eff_fuel_cell = np.array(
            [0.10, 0.30, 0.36, 0.45, 0.50, 0.56, 0.58, 0.60, 0.58, 0.57, 0.55, 0.54])
        eff_hd_diesel = np.array(
            [0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34])

        if veh['fcEffType'] == 1:  # SI engine
            eff = np.copy(eff_si) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 2:  # Atkinson cycle SI engine -- greater expansion
            eff = np.copy(eff_atk) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 3:  # Diesel (compression ignition) engine
            eff = np.copy(eff_diesel) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 4:  # H2 fuel cell
            eff = np.copy(eff_fuel_cell) + veh['fcAbsEffImpr']

        elif veh['fcEffType'] == 5:  # heavy duty Diesel engine
            eff = np.copy(eff_hd_diesel) + veh['fcAbsEffImpr']

        # discrete array of possible engine power outputs
        inputKwOutArray = fcPwrOutPerc * veh['maxFuelConvKw']
        # Relatively continuous power out percentages for assigning FC efficiencies
        fcPercOutArray = np.r_[np.arange(0, 3.0, 0.1), np.arange(
            3.0, 7.0, 0.5), np.arange(7.0, 60.0, 1.0), np.arange(60.0, 105.0, 5.0)] / 100
        # Relatively continuous array of possible engine power outputs
        fcKwOutArray = veh['maxFuelConvKw'] * fcPercOutArray
        # Initializes relatively continuous array for fcEFF
        fcEffArray = np.array([0.0] * len(fcPercOutArray))

        # the following for loop populates fcEffArray
        for j in range(0, len(fcPercOutArray) - 1):
            low_index = np.argmax(inputKwOutArray >= fcKwOutArray[j])
            fcinterp_x_1 = inputKwOutArray[low_index-1]
            fcinterp_x_2 = inputKwOutArray[low_index]
            fcinterp_y_1 = eff[low_index-1]
            fcinterp_y_2 = eff[low_index]
            fcEffArray[j] = (fcKwOutArray[j] - fcinterp_x_1)/(fcinterp_x_2 -
                                                              fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

        # populate final value
        fcEffArray[-1] = eff[-1]

        # assign corresponding values in veh dict
        veh['fcEffArray'] = np.copy(fcEffArray)
        veh['fcKwOutArray'] = np.copy(fcKwOutArray)
        veh['maxFcEffKw'] = np.copy(veh['fcKwOutArray'][np.argmax(fcEffArray)])
        veh['fcMaxOutkW'] = np.copy(max(inputKwOutArray))
        veh['minFcTimeOn'] = 30  # hardcoded

    else:
        # these things are all zero for BEV powertrains
        # not sure why `veh['fcEffArray']` is not being assigned.
        # Maybe it's not used anywhere in this condition.  *** delete this comment before public release
        veh['fcKwOutArray'] = np.array([0] * 101)
        veh['maxFcEffKw'] = 0
        veh['fcMaxOutkW'] = 0
        veh['minFcTimeOn'] = 30  # hardcoded

    ### Defining MC efficiency curve as lookup table for %power_in vs power_out
    ### see "Motor" tab in FASTSim for Excel
    if veh['maxMotorKw'] > 0:

        maxMotorKw = veh['maxMotorKw']

        mcPwrOutPerc = np.array(
            [0.00, 0.02, 0.04, 0.06, 0.08,	0.10,	0.20,	0.40,	0.60,	0.80,	1.00])
        large_baseline_eff = np.array(
            [0.83, 0.85,	0.87,	0.89,	0.90,	0.91,	0.93,	0.94,	0.94,	0.93,	0.92])
        small_baseline_eff = np.array(
            [0.12,	0.16,	 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93,	0.93,	0.92])

        modern_max = 0.95
        modern_diff = modern_max - max(large_baseline_eff)

        large_baseline_eff_adj = large_baseline_eff + modern_diff

        mcKwAdjPerc = max(0.0, min((maxMotorKw - 7.5)/(75.0 - 7.5), 1.0))
        mcEffArray = np.array([0.0] * len(mcPwrOutPerc))

        for k in range(0, len(mcPwrOutPerc)):
            mcEffArray[k] = mcKwAdjPerc * large_baseline_eff_adj[k] + \
                (1 - mcKwAdjPerc)*(small_baseline_eff[k])

        mcInputKwOutArray = mcPwrOutPerc * maxMotorKw

        mcPercOutArray = np.linspace(0, 1, 101)
        mcKwOutArray = np.linspace(0, 1, 101) * maxMotorKw

        mcFullEffArray = np.array([0.0] * len(mcPercOutArray))

        for m in range(1, len(mcPercOutArray) - 1):
            low_index = np.argmax(mcInputKwOutArray >= mcKwOutArray[m])

            fcinterp_x_1 = mcInputKwOutArray[low_index-1]
            fcinterp_x_2 = mcInputKwOutArray[low_index]
            fcinterp_y_1 = mcEffArray[low_index-1]
            fcinterp_y_2 = mcEffArray[low_index]

            mcFullEffArray[m] = (mcKwOutArray[m] - fcinterp_x_1)/(
                fcinterp_x_2 - fcinterp_x_1)*(fcinterp_y_2 - fcinterp_y_1) + fcinterp_y_1

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
        veh['mcKwOutArray'] = np.array([0.0] * 101)
        veh['mcMaxElecInKw'] = 0

    veh['mcMaxElecInKw'] = max(veh['mcKwInArray'])

    ### Specify shape of mc regen efficiency curve
    ### see "Regen" tab in FASTSim for Excel
    veh['regenA'] = 500.0  # hardcoded
    veh['regenB'] = 0.99  # hardcoded

    ### Calculate total vehicle mass
    # sum up component masses if positive real number is not specified for vehOverrideKg
    if not(veh['vehOverrideKg'] > 0):
        if veh['maxEssKwh'] == 0 or veh['maxEssKw'] == 0:
            ess_mass_kg = 0.0
        else:
            ess_mass_kg = ((veh['maxEssKwh'] * veh['essKgPerKwh']) +
                           veh['essBaseKg']) * veh['compMassMultiplier']
        if veh['maxMotorKw'] == 0:
            mc_mass_kg = 0.0
        else:
            mc_mass_kg = (veh['mcPeBaseKg']+(veh['mcPeKgPerKw']
                                             * veh['maxMotorKw'])) * veh['compMassMultiplier']
        if veh['maxFuelConvKw'] == 0:
            fc_mass_kg = 0.0
        else:
            fc_mass_kg = (((1 / veh['fuelConvKwPerKg']) * veh['maxFuelConvKw'] +
                           veh['fuelConvBaseKg'])) * veh['compMassMultiplier']
        if veh['maxFuelStorKw'] == 0:
            fs_mass_kg = 0.0
        else:
            fs_mass_kg = ((1 / veh['fuelStorKwhPerKg']) *
                          veh['fuelStorKwh']) * veh['compMassMultiplier']
        veh['vehKg'] = veh['cargoKg'] + veh['gliderKg'] + veh['transKg'] * \
            veh['compMassMultiplier'] + ess_mass_kg + \
            mc_mass_kg + fc_mass_kg + fs_mass_kg
    # if positive real number is specified for vehOverrideKg, use that
    else:
        veh['vehKg'] = np.copy(veh['vehOverrideKg'])

    # replace any spaces with underscores
    veh = dict(list(zip([key.replace(' ', '_')
                         for key in veh.keys()], veh.values())))

    # convert veh dict to namedtuple
    Veh = namedtuple('Veh', list(veh.keys()))
    veh = Veh(**veh)
    return veh
