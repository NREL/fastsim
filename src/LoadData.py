import pandas as pd
from collections import namedtuple
import numpy as np
import re

def get_standard_cycle(cycle_name):
    """Load time trace of speed, grade, and road type."""
    csv_path = '..//cycles//'+cycle_name+'.csv'
    data = pd.read_csv(csv_path)
    return data

class Vehicle(object):
    """Class for loading and contaning vehicle """
    def __init__(self, vnum=None):
        super().__init__()
        if vnum:
            self.load_vnum(vnum)
        
    def load_vnum(self, vnum):
        """Load vehicle parameters and assign to self."""

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
            col1 = col.replace(' ', '_')
            
            # assign dataframe columns 
            self.__setattr__(col1, vehdf.loc[vnum, col])
        
        self.set_init_calcs()
        self.set_veh_mass()

    def set_init_calcs(self):
        """Set parameters that can be calculated after loading vehicle data"""
        ### Build roadway power lookup table
        self.MaxRoadwayChgKw_Roadway = range(6)
        self.MaxRoadwayChgKw = [0] * len(self.MaxRoadwayChgKw_Roadway)
        self.chargingOn = 0

        # Checking if a vehicle has any hybrid components
        if self.maxEssKwh == 0 or self.maxEssKw == 0 or self.maxMotorKw == 0:
            self.noElecSys = 'TRUE'

        else:
            self.noElecSys = 'FALSE'

        # Checking if aux loads go through an alternator
        if self.noElecSys == 'TRUE' or self.maxMotorKw <= self.auxKw or self.forceAuxOnFC == 'TRUE':
            self.noElecAux = 'TRUE'

        else:
            self.noElecAux = 'FALSE'

        # Copying vehPtType to additional key
        self.vehTypeSelection = np.copy(self.vehPtType)
        # to be consistent with Excel version but not used in Python version

        ### Defining Fuel Converter efficiency curve as lookup table for %power_in vs power_out
        ### see "FC Model" tab in FASTSim for Excel

        if self.maxFuelConvKw > 0:

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

            if self.fcEffType == 1:  # SI engine
                eff = np.copy(eff_si) + self.fcAbsEffImpr

            elif self.fcEffType == 2:  # Atkinson cycle SI engine -- greater expansion
                eff = np.copy(eff_atk) + self.fcAbsEffImpr

            elif self.fcEffType == 3:  # Diesel (compression ignition) engine
                eff = np.copy(eff_diesel) + self.fcAbsEffImpr

            elif self.fcEffType == 4:  # H2 fuel cell
                eff = np.copy(eff_fuel_cell) + self.fcAbsEffImpr

            elif self.fcEffType == 5:  # heavy duty Diesel engine
                eff = np.copy(eff_hd_diesel) + self.fcAbsEffImpr

            # discrete array of possible engine power outputs
            inputKwOutArray = fcPwrOutPerc * self.maxFuelConvKw
            # Relatively continuous power out percentages for assigning FC efficiencies
            fcPercOutArray = np.r_[np.arange(0, 3.0, 0.1), np.arange(
                3.0, 7.0, 0.5), np.arange(7.0, 60.0, 1.0), np.arange(60.0, 105.0, 5.0)] / 100
            # Relatively continuous array of possible engine power outputs
            fcKwOutArray = self.maxFuelConvKw * fcPercOutArray
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
            self.fcEffArray = np.copy(fcEffArray)
            self.fcKwOutArray = np.copy(fcKwOutArray)
            self.maxFcEffKw = np.copy(self.fcKwOutArray[np.argmax(fcEffArray)])
            self.fcMaxOutkW = np.copy(max(inputKwOutArray))
            self.minFcTimeOn = 30  # hardcoded

        else:
            # these things are all zero for BEV powertrains
            # not sure why `self.fcEffArray` is not being assigned.
            # Maybe it's not used anywhere in this condition.  *** delete this comment before public release
            self.fcKwOutArray = np.array([0] * 101)
            self.maxFcEffKw = 0
            self.fcMaxOutkW = 0
            self.minFcTimeOn = 30  # hardcoded

        ### Defining MC efficiency curve as lookup table for %power_in vs power_out
        ### see "Motor" tab in FASTSim for Excel
        if self.maxMotorKw > 0:

            maxMotorKw = self.maxMotorKw

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

            self.mcKwInArray = np.copy(mcKwInArray)
            self.mcKwOutArray = np.copy(mcKwOutArray)
            self.mcMaxElecInKw = np.copy(max(mcKwInArray))
            self.mcFullEffArray = np.copy(mcFullEffArray)
            self.mcEffArray = np.copy(mcEffArray)

        else:
            self.mcKwInArray = np.array([0.0] * 101)
            self.mcKwOutArray = np.array([0.0] * 101)
            self.mcMaxElecInKw = 0

        self.mcMaxElecInKw = max(self.mcKwInArray)

        ### Specify shape of mc regen efficiency curve
        ### see "Regen" tab in FASTSim for Excel
        self.regenA = 500.0  # hardcoded
        self.regenB = 0.99  # hardcoded

    def set_veh_mass(self):
        """Calculate total vehicle mass.  Sum up component masses if 
        positive real number is not specified for vehOverrideKg"""
        if not(self.vehOverrideKg > 0):
            if self.maxEssKwh == 0 or self.maxEssKw == 0:
                ess_mass_kg = 0.0
            else:
                ess_mass_kg = ((self.maxEssKwh * self.essKgPerKwh) +
                            self.essBaseKg) * self.compMassMultiplier
            if self.maxMotorKw == 0:
                mc_mass_kg = 0.0
            else:
                mc_mass_kg = (self.mcPeBaseKg+(self.mcPeKgPerKw
                                                * self.maxMotorKw)) * self.compMassMultiplier
            if self.maxFuelConvKw == 0:
                fc_mass_kg = 0.0
            else:
                fc_mass_kg = (((1 / self.fuelConvKwPerKg) * self.maxFuelConvKw +
                            self.fuelConvBaseKg)) * self.compMassMultiplier
            if self.maxFuelStorKw == 0:
                fs_mass_kg = 0.0
            else:
                fs_mass_kg = ((1 / self.fuelStorKwhPerKg) *
                            self.fuelStorKwh) * self.compMassMultiplier
            self.vehKg = self.cargoKg + self.gliderKg + self.transKg * \
                self.compMassMultiplier + ess_mass_kg + \
                mc_mass_kg + fc_mass_kg + fs_mass_kg
        # if positive real number is specified for vehOverrideKg, use that
        else:
            self.vehKg = np.copy(self.vehOverrideKg)
