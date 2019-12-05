"""Module containing class and functions for simulating vehicle drive cycle."""

### Import necessary python modules
import numpy as np
import pandas as pd
from collections import namedtuple
import warnings
warnings.simplefilter('ignore')

airDensityKgPerM3 = 1.2 # Sea level air density at approximately 20C
gravityMPerSec2 = 9.81
mphPerMps = 2.2369
kWhPerGGE = 33.7
metersPerMile = 1609.00

class SimDrive(object):
    """Class for contaning time series data used by sim_drive_sub"""
    def __init__(self):
        super().__init__()
    
    def sim_drive(self, cyc, veh, initSoc=None):
        """Initialize and run sim_drive_sub as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        cyc: pandas dataframe of time traces of cycle data from LoadData.get_standard_cycle()
        veh: instance of LoadData.Vehicle() class
        initSoc: initial SOC for electrified vehicles

        Outputs
        -----------
        output: dict of key output variables
        tarr: object contaning arrays of all temporally dynamic variables

        Example Usage (from within ../docs/):
        >>> # local modules
        >>> import SimDrive
        >>> import LoadData
        >>> cyc = LoadData.get_standard_cycle("UDDS")
        >>> output, tarr = SimDrive.sim_drive(cyc, veh)
        """

        if initSoc != None:
            if initSoc > 1.0 or initSoc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                initSoc = None
    
        if veh.vehPtType == 1: # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            initSoc = 0.0
            
            self.sim_drive_sub(cyc, veh, initSoc)

        elif veh.vehPtType == 2 and initSoc == None:  # HEV 

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
                output = self.get_output(veh)
                ess2fuelKwh = abs(output['ess2fuelKwh'])
                initSoc = min(1.0, max(0.0, output['final_soc']))
                        
            self.sim_drive_sub(cyc, veh, initSoc)

        elif (veh.vehPtType == 3 and initSoc == None) or (veh.vehPtType == 4 and initSoc == None): # PHEV and BEV

            # If EV, initializing initial SOC to maximum SOC.

            initSoc = np.copy(veh.maxSoc)
            
            self.sim_drive_sub(cyc, veh, initSoc)

        else:
            
            self.sim_drive_sub(cyc, veh, initSoc)

    def sim_drive_sub(self, cyc, veh, initSoc):
        """  
        Receives second-by-second cycle information, vehicle properties, 
        and an initial state of charge and performs a backward facing 
        powertrain simulation. The function returns an output dictionary 
        starting at approximately line 1030. Powertrain variables of 
        interest (summary or time-series) can be added to the output 
        dictionary for reference. Function 'sim_drive' runs this to 
        iterate through the time steps of 'cyc'.

        Arguments
        ------------
        cyc: pandas dataframe of time traces of cycle data from LoadData.get_standard_cycle()
        veh: instance of LoadData.Vehicle() class
        initSoc: initial battery state-of-charge (SOC) for electrified vehicles
        
        Outputs
        -----------
        output: dict of key output variables
        tarr: object contaning arrays of all temporally dynamic variables"""
        
        ############################
        ###   Define Constants   ###
        ############################

        veh.maxTracMps2 = ((((veh.wheelCoefOfFric * veh.driveAxleWeightFrac * veh.vehKg * gravityMPerSec2) /\
            (1+((veh.vehCgM * veh.wheelCoefOfFric) / veh.wheelBaseM))))/(veh.vehKg * gravityMPerSec2)) * gravityMPerSec2
        veh.maxRegenKwh = 0.5 * veh.vehKg * (27**2) / (3600 * 1000)

        #############################
        ### Initialize Variables  ###
        #############################

        ### Drive Cycle copied as numpy array for computational speed
        self.cycSecs = cyc['cycSecs'].copy().to_numpy()
        self.cycMps = cyc['cycMps'].copy().to_numpy()
        self.cycGrade = cyc['cycGrade'].copy().to_numpy()
        self.cycRoadType = cyc['cycRoadType'].copy().to_numpy()
        self.cycMph = np.copy(self.cycMps * mphPerMps)
        self.secs = np.insert(np.diff(self.cycSecs), 0, 0)

        ############################
        ###   Loop Through Time  ###
        ############################

        self.init_arrays(veh, initSoc)

        for i in range(1, len(self.cycSecs)):
            ### Misc calcs
            # If noElecAux, then the HV electrical system is not used to power aux loads 
            # and it must all come from the alternator.  This apparently assumes no belt-driven aux 
            # loads
            # *** 

            self.get_misc_calcs(i, veh)
            self.get_comp_lims(i, veh)
            self.get_power_calcs(i, veh)
            self.get_speed_dist_calcs(i, veh)

            if self.transKwOutAch[i] > 0:
                self.transKwInAch[i] = self.transKwOutAch[i] / veh.transEff
            else:
                self.transKwInAch[i] = self.transKwOutAch[i] * veh.transEff

            if self.cycMet[i] == 1:

                if veh.fcEffType == 4:
                    self.minMcKw2HelpFc[i] = max(self.transKwInAch[i], -self.curMaxMechMcKwIn[i])

                else:
                    self.minMcKw2HelpFc[i] = max(self.transKwInAch[i] - self.curMaxFcKwOut[i], -self.curMaxMechMcKwIn[i])
            else:
                self.minMcKw2HelpFc[i] = max(self.curMaxMcKwOut[i], -self.curMaxMechMcKwIn[i])

            if veh.noElecSys == 'TRUE':
                self.regenBufferSoc[i] = 0

            elif veh.chargingOn:
                self.regenBufferSoc[i] = max(veh.maxSoc - (veh.maxRegenKwh / veh.maxEssKwh), (veh.maxSoc + veh.minSoc) / 2)

            else:
                self.regenBufferSoc[i] = max(((veh.maxEssKwh * veh.maxSoc) - (0.5 * veh.vehKg * (self.cycMps[i]**2) * (1.0 / 1000) \
                    * (1.0 / 3600) * veh.motorPeakEff * veh.maxRegen)) / veh.maxEssKwh, veh.minSoc)

                self.essRegenBufferDischgKw[i] = min(self.curMaxEssKwOut[i], max(0, (self.soc[i-1] - self.regenBufferSoc[i]) * veh.maxEssKwh * 3600 / self.secs[i]))

                self.maxEssRegenBufferChgKw[i] = min(max(0, (self.regenBufferSoc[i] - self.soc[i-1]) * veh.maxEssKwh * 3600.0 / self.secs[i]), self.curMaxEssChgKw[i])

            if veh.noElecSys == 'TRUE':
                self.accelBufferSoc[i] = 0

            else:
                self.accelBufferSoc[i] = min(max((((((((veh.maxAccelBufferMph * (1 / mphPerMps))**2)) - ((self.cycMps[i]**2))) / \
                    (((veh.maxAccelBufferMph * (1 / mphPerMps))**2))) * (min(veh.maxAccelBufferPercOfUseableSoc * \
                        (veh.maxSoc - veh.minSoc), veh.maxRegenKwh / veh.maxEssKwh) * veh.maxEssKwh)) / veh.maxEssKwh) + \
                            veh.minSoc, veh.minSoc), veh.maxSoc)

                self.essAccelBufferChgKw[i] = max(0, (self.accelBufferSoc[i] - self.soc[i-1]) * veh.maxEssKwh * 3600.0 / self.secs[i])
                self.maxEssAccelBufferDischgKw[i] = min(max(0, (self.soc[i-1] - self.accelBufferSoc[i]) * veh.maxEssKwh * 3600 / self.secs[i]), self.curMaxEssKwOut[i])

            if self.regenBufferSoc[i] < self.accelBufferSoc[i]:
                self.essAccelRegenDischgKw[i] = max(min(((self.soc[i-1] - (self.regenBufferSoc[i] + self.accelBufferSoc[i]) / 2) * veh.maxEssKwh * 3600.0) /\
                    self.secs[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

            elif self.soc[i-1] > self.regenBufferSoc[i]:
                self.essAccelRegenDischgKw[i] = max(min(self.essRegenBufferDischgKw[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

            elif self.soc[i-1] < self.accelBufferSoc[i]:
                self.essAccelRegenDischgKw[i] = max(min(-1.0 * self.essAccelBufferChgKw[i], self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

            else:
                self.essAccelRegenDischgKw[i] = max(min(0, self.curMaxEssKwOut[i]), -self.curMaxEssChgKw[i])

            self.fcKwGapFrEff[i] = abs(self.transKwOutAch[i] - veh.maxFcEffKw)

            if veh.noElecSys == 'TRUE':
                self.mcElectInKwForMaxFcEff[i] = 0

            elif self.transKwOutAch[i] < veh.maxFcEffKw:

                if self.fcKwGapFrEff[i] == veh.maxMotorKw:
                    self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1] * -1
                else:
                    self.mcElectInKwForMaxFcEff[i] = self.fcKwGapFrEff[i] / veh.mcFullEffArray[max(1, np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1)] * -1

            else:

                if self.fcKwGapFrEff[i] == veh.maxMotorKw:
                    self.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[len(veh.mcKwInArray) - 1]
                else:
                    self.mcElectInKwForMaxFcEff[i] = veh.mcKwInArray[np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01, self.fcKwGapFrEff[i])) - 1]

            if veh.noElecSys == 'TRUE':
                self.electKwReq4AE[i] = 0

            elif self.transKwInAch[i] > 0:
                if self.transKwInAch[i] == veh.maxMotorKw:
            
                    self.electKwReq4AE[i] = self.transKwInAch[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1] + self.auxInKw[i]
                else:
                    self.electKwReq4AE[i] = self.transKwInAch[i] / veh.mcFullEffArray[max(1, np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01, self.transKwInAch[i])) - 1)] + self.auxInKw[i]

            else:
                self.electKwReq4AE[i] = 0

            self.prevfcTimeOn[i] = self.fcTimeOn[i-1]

            if veh.maxFuelConvKw == 0:
                self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and self.transKwInAch[i]<=self.curMaxMcKwOut[i] and (self.electKwReq4AE[i] < self.curMaxElecKw[i] or veh.maxFuelConvKw == 0)

            else:
                self.canPowerAllElectrically[i] = self.accelBufferSoc[i] < self.soc[i-1] and self.transKwInAch[i]<=self.curMaxMcKwOut[i] and (self.electKwReq4AE[i] < self.curMaxElecKw[i] \
                    or veh.maxFuelConvKw == 0) and (self.cycMph[i] - 0.00001<=veh.mphFcOn or veh.chargingOn) and self.electKwReq4AE[i]<=veh.kwDemandFcOn

            if self.canPowerAllElectrically[i]:

                if self.transKwInAch[i]<+self.auxInKw[i]:
                    self.desiredEssKwOutForAE[i] = self.auxInKw[i] + self.transKwInAch[i]

                elif self.regenBufferSoc[i] < self.accelBufferSoc[i]:
                    self.desiredEssKwOutForAE[i] = self.essAccelRegenDischgKw[i]

                elif self.soc[i-1] > self.regenBufferSoc[i]:
                    self.desiredEssKwOutForAE[i] = self.essRegenBufferDischgKw[i]

                elif self.soc[i-1] < self.accelBufferSoc[i]:
                    self.desiredEssKwOutForAE[i] = -self.essAccelBufferChgKw[i]

                else:
                    self.desiredEssKwOutForAE[i] = self.transKwInAch[i] + self.auxInKw[i] - self.curMaxRoadwayChgKw[i]

            else:
                self.desiredEssKwOutForAE[i] = 0

            if self.canPowerAllElectrically[i]:
                self.essAEKwOut[i] = max(-self.curMaxEssChgKw[i], -self.maxEssRegenBufferChgKw[i], min(0, self.curMaxRoadwayChgKw[i] - (self.transKwInAch[i] + self.auxInKw[i])), min(self.curMaxEssKwOut[i], self.desiredEssKwOutForAE[i]))

            else:
                self.essAEKwOut[i] = 0

            self.erAEKwOut[i] = min(max(0, self.transKwInAch[i] + self.auxInKw[i] - self.essAEKwOut[i]), self.curMaxRoadwayChgKw[i])

            self.get_fc_forced_state(i, veh)

            if (-self.mcElectInKwForMaxFcEff[i] - self.curMaxRoadwayChgKw[i]) > 0:
                self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] - self.curMaxRoadwayChgKw[i]) * veh.essDischgToFcMaxEffPerc

            else:
                self.essDesiredKw4FcEff[i] = (-self.mcElectInKwForMaxFcEff[i] - self.curMaxRoadwayChgKw[i]) * veh.essChgToFcMaxEffPerc

            if self.accelBufferSoc[i] > self.regenBufferSoc[i]:
                self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i], \
                    max(-self.curMaxEssChgKw[i], self.essAccelRegenDischgKw[i]))

            elif self.essRegenBufferDischgKw[i] > 0:
                self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i], \
                    max(-self.curMaxEssChgKw[i], min(self.essAccelRegenDischgKw[i], self.mcElecInLimKw[i] + self.auxInKw[i], max(self.essRegenBufferDischgKw[i], self.essDesiredKw4FcEff[i]))))

            elif self.essAccelBufferChgKw[i] > 0:
                self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i], \
                    max(-self.curMaxEssChgKw[i], max(-1 * self.maxEssRegenBufferChgKw[i], min(-self.essAccelBufferChgKw[i], self.essDesiredKw4FcEff[i]))))


            elif self.essDesiredKw4FcEff[i] > 0:
                self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i], \
                    max(-self.curMaxEssChgKw[i], min(self.essDesiredKw4FcEff[i], self.maxEssAccelBufferDischgKw[i])))

            else:
                self.essKwIfFcIsReq[i] = min(self.curMaxEssKwOut[i], veh.mcMaxElecInKw + self.auxInKw[i], self.curMaxMcElecKwIn[i] + self.auxInKw[i], \
                    max(-self.curMaxEssChgKw[i], max(self.essDesiredKw4FcEff[i], -self.maxEssRegenBufferChgKw[i])))

            self.erKwIfFcIsReq[i] = max(0, min(self.curMaxRoadwayChgKw[i], self.curMaxMechMcKwIn[i], self.essKwIfFcIsReq[i] - self.mcElecInLimKw[i] + self.auxInKw[i]))

            self.mcElecKwInIfFcIsReq[i] = self.essKwIfFcIsReq[i] + self.erKwIfFcIsReq[i] - self.auxInKw[i]

            if veh.noElecSys == 'TRUE':
                self.mcKwIfFcIsReq[i] = 0

            elif  self.mcElecKwInIfFcIsReq[i] == 0:
                self.mcKwIfFcIsReq[i] = 0

            elif self.mcElecKwInIfFcIsReq[i] > 0:

                if self.mcElecKwInIfFcIsReq[i] == max(veh.mcKwInArray):
                    self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
                else:
                    self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] * veh.mcFullEffArray[max(1, np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i])) - 1)]

            else:
                if self.mcElecKwInIfFcIsReq[i] * -1 == max(veh.mcKwInArray):
                    self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
                else:
                    self.mcKwIfFcIsReq[i] = self.mcElecKwInIfFcIsReq[i] / (veh.mcFullEffArray[max(1, np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01, self.mcElecKwInIfFcIsReq[i] * -1)) - 1)])

            if veh.maxMotorKw == 0:
                self.mcMechKwOutAch[i] = 0

            elif self.fcForcedOn[i] == True and self.canPowerAllElectrically[i] == True and (veh.vehPtType == 2.0 or veh.vehPtType == 3.0) and veh.fcEffType!=4:
                self.mcMechKwOutAch[i] =  self.mcMechKw4ForcedFc[i]

            elif self.transKwInAch[i]<=0:

                if veh.fcEffType!=4 and veh.maxFuelConvKw> 0:
                    if self.canPowerAllElectrically[i] == 1:
                        self.mcMechKwOutAch[i] = -min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i])
                    else:
                        self.mcMechKwOutAch[i] = min(-min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]), max(-self.curMaxFcKwOut[i], self.mcKwIfFcIsReq[i]))
                else:
                    self.mcMechKwOutAch[i] = min(-min(self.curMaxMechMcKwIn[i], -self.transKwInAch[i]), -self.transKwInAch[i])

            elif self.canPowerAllElectrically[i] == 1:
                self.mcMechKwOutAch[i] = self.transKwInAch[i]

            else:
                self.mcMechKwOutAch[i] = max(self.minMcKw2HelpFc[i], self.mcKwIfFcIsReq[i])

            if self.mcMechKwOutAch[i] == 0:
                self.mcElecKwInAch[i] = 0.0
                self.motor_index_debug[i] = 0

            elif self.mcMechKwOutAch[i] < 0:

                if self.mcMechKwOutAch[i] * -1 == max(veh.mcKwInArray):
                    self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
                else:
                    self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] * veh.mcFullEffArray[max(1, np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) - 0.01, self.mcMechKwOutAch[i] * -1)) - 1)]

            else:
                if veh.maxMotorKw == self.mcMechKwOutAch[i]:
                    self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
                else:
                    self.mcElecKwInAch[i] = self.mcMechKwOutAch[i] / veh.mcFullEffArray[max(1, np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01, self.mcMechKwOutAch[i])) - 1)]

            if self.curMaxRoadwayChgKw[i] == 0:
                self.roadwayChgKwOutAch[i] = 0

            elif veh.fcEffType == 4:
                self.roadwayChgKwOutAch[i] = max(0, self.mcElecKwInAch[i], self.maxEssRegenBufferChgKw[i], self.essRegenBufferDischgKw[i], self.curMaxRoadwayChgKw[i])

            elif self.canPowerAllElectrically[i] == 1:
                self.roadwayChgKwOutAch[i] = self.erAEKwOut[i]

            else:
                self.roadwayChgKwOutAch[i] = self.erKwIfFcIsReq[i]

            self.minEssKw2HelpFc[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - self.curMaxFcKwOut[i] - self.roadwayChgKwOutAch[i]

            if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
                self.essKwOutAch[i]  = 0

            elif veh.fcEffType == 4:

                if self.transKwOutAch[i]>=0:
                    self.essKwOutAch[i] = min(max(self.minEssKw2HelpFc[i], self.essDesiredKw4FcEff[i], self.essAccelRegenDischgKw[i]), self.curMaxEssKwOut[i], self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i])

                else:
                    self.essKwOutAch[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i]

            elif self.highAccFcOnTag[i] == 1 or veh.noElecAux == 'TRUE':
                self.essKwOutAch[i] = self.mcElecKwInAch[i] - self.roadwayChgKwOutAch[i]

            else:
                self.essKwOutAch[i] = self.mcElecKwInAch[i] + self.auxInKw[i] - self.roadwayChgKwOutAch[i]

            if veh.maxFuelConvKw == 0:
                self.fcKwOutAch[i] = 0

            elif veh.fcEffType == 4:
                self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(0, self.mcElecKwInAch[i] + self.auxInKw[i] - self.essKwOutAch[i] - self.roadwayChgKwOutAch[i]))

            elif veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or self.highAccFcOnTag[i] == 1:
                self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(0, self.transKwInAch[i] - self.mcMechKwOutAch[i] + self.auxInKw[i]))

            else:
                self.fcKwOutAch[i] = min(self.curMaxFcKwOut[i], max(0, self.transKwInAch[i] - self.mcMechKwOutAch[i]))

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
                    self.fcKwInAch[i] = self.fcKwOutAch[i] / veh.fcEffArray[len(veh.fcEffArray) - 1]
                else:
                    self.fcKwInAch[i] = self.fcKwOutAch[i] / (veh.fcEffArray[max(1, np.argmax(veh.fcKwOutArray > min(self.fcKwOutAch[i], veh.fcMaxOutkW - 0.001)) - 1)])

            self.fsKwOutAch[i] = np.copy(self.fcKwInAch[i])

            self.fsKwhOutAch[i] = self.fsKwOutAch[i] * self.secs[i] * (1 / 3600.0)


            if veh.noElecSys == 'TRUE':
                self.essCurKwh[i] = 0

            elif self.essKwOutAch[i] < 0:
                self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * (self.secs[i] / 3600.0) * np.sqrt(veh.essRoundTripEff)

            else:
                self.essCurKwh[i] = self.essCurKwh[i-1] - self.essKwOutAch[i] * (self.secs[i] / 3600.0) * (1 / np.sqrt(veh.essRoundTripEff))

            if veh.maxEssKwh == 0:
                self.soc[i] = 0.0

            else:
                self.soc[i] = self.essCurKwh[i] / veh.maxEssKwh

            if self.canPowerAllElectrically[i] == True and self.fcForcedOn[i] == False and self.fcKwOutAch[i] == 0:
                self.fcTimeOn[i] = 0
            else:
                self.fcTimeOn[i] = self.fcTimeOn[i-1] + self.secs[i]

            self.get_battery_wear(i, veh)
            self.get_energy_audit(i, veh)

    def init_arrays(self, veh, initSoc):
        """Initializes arrays of time dependent variables as attributes of self.
        Arguments
        ------------
        veh: instance of LoadData.Vehicle() class
        initSoc: initial SOC for electrified vehicles
        """

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
        control_list = ['regenBufferSoc', 'essRegenBufferDischgKw', 'maxEssRegenBufferChgKw',
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
        attributes = comp_lim_list + drivetrain_list + control_list + misc_list

        # assign numpy.zeros of the same length as cycSecs to self attributes
        for attribute in attributes:
            self.__setattr__(attribute, np.zeros(len(self.cycSecs)))

        self.fcForcedOn = np.array([False] * len(self.cycSecs))
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

    # Function definitions for functions to be run at each time step
    def get_misc_calcs(self, i, veh):
        """Performs misc. calculations.
        Arguments
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step
        
        Output: tarr"""

        if veh.noElecAux == 'TRUE':
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
        self.maxTracMps[i] = self.mpsAch[i-1] + (veh.maxTracMps2 * self.secs[i])

    def get_comp_lims(self, i, veh):
        """Return time array (tarr) with component limits set for time step 'i'
        Arguments
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step

        Output: tarr"""

        # max fuel storage power output
        self.curMaxFsKwOut[i] = min(veh.maxFuelStorKw, self.fsKwOutAch[i-1] + (
            (veh.maxFuelStorKw/veh.fuelStorSecsToPeakPwr) * (self.secs[i])))
        # maximum fuel storage power output rate of change
        self.fcTransLimKw[i] = self.fcKwOutAch[i-1] + \
            ((veh.maxFuelConvKw / veh.fuelConvSecsToPeakPwr) * (self.secs[i]))

        # *** this min seems redundant with line 518
        self.fcMaxKwIn[i] = min(self.curMaxFsKwOut[i], veh.maxFuelStorKw)
        self.fcFsLimKw[i] = veh.fcMaxOutkW
        self.curMaxFcKwOut[i] = min(
            veh.maxFuelConvKw, self.fcFsLimKw[i], self.fcTransLimKw[i])

        # Does ESS discharge need to be limited? *** I think veh.maxEssKw should also be in the following
        # boolean condition
        if veh.maxEssKwh == 0 or self.soc[i-1] < veh.minSoc:
            self.essCapLimDischgKw[i] = 0.0

        else:
            self.essCapLimDischgKw[i] = (
                veh.maxEssKwh * np.sqrt(veh.essRoundTripEff)) * 3600.0 * (self.soc[i-1] - veh.minSoc) / (self.secs[i])
        self.curMaxEssKwOut[i] = min(
            veh.maxEssKw, self.essCapLimDischgKw[i])

        if veh.maxEssKwh == 0 or veh.maxEssKw == 0:
            self.essCapLimChgKw[i] = 0

        else:
            self.essCapLimChgKw[i] = max(((veh.maxSoc - self.soc[i-1]) * veh.maxEssKwh * (1 /
                                                                                            np.sqrt(veh.essRoundTripEff))) / ((self.secs[i]) * (1 / 3600.0)), 0)

        self.curMaxEssChgKw[i] = min(self.essCapLimChgKw[i], veh.maxEssKw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if veh.fcEffType == 4:
            self.curMaxElecKw[i] = self.curMaxFcKwOut[i] + self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        else:
            self.curMaxElecKw[i] = self.curMaxRoadwayChgKw[i] + \
                self.curMaxEssKwOut[i] - self.auxInKw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        self.curMaxAvailElecKw[i] = min(
            self.curMaxElecKw[i], veh.mcMaxElecInKw)

        if self.curMaxElecKw[i] > 0:
            # limit power going into e-machine controller to
            if self.curMaxAvailElecKw[i] == max(veh.mcKwInArray):
                self.mcElecInLimKw[i] = min(
                    veh.mcKwOutArray[len(veh.mcKwOutArray) - 1], veh.maxMotorKw)
            else:
                self.mcElecInLimKw[i] = min(veh.mcKwOutArray[np.argmax(veh.mcKwInArray > min(max(veh.mcKwInArray) -
                                                                                                0.01, self.curMaxAvailElecKw[i])) - 1], veh.maxMotorKw)
        else:
            self.mcElecInLimKw[i] = 0.0

        # Motor transient power limit
        self.mcTransiLimKw[i] = abs(
            self.mcMechKwOutAch[i-1]) + ((veh.maxMotorKw / veh.motorSecsToPeakPwr) * (self.secs[i]))

        self.curMaxMcKwOut[i] = max(min(
            self.mcElecInLimKw[i], self.mcTransiLimKw[i], veh.maxMotorKw), -veh.maxMotorKw)

        if self.curMaxMcKwOut[i] == 0:
            self.curMaxMcElecKwIn[i] = 0
        else:
            if self.curMaxMcKwOut[i] == veh.maxMotorKw:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / \
                    veh.mcFullEffArray[len(veh.mcFullEffArray) - 1]
            else:
                self.curMaxMcElecKwIn[i] = self.curMaxMcKwOut[i] / veh.mcFullEffArray[max(1, np.argmax(veh.mcKwOutArray
                                                                                                        > min(veh.maxMotorKw - 0.01, self.curMaxMcKwOut[i])) - 1)]

        if veh.maxMotorKw == 0:
            self.essLimMcRegenPercKw[i] = 0.0

        else:
            self.essLimMcRegenPercKw[i] = min(
                (self.curMaxEssChgKw[i] + self.auxInKw[i]) / veh.maxMotorKw, 1)
        if self.curMaxEssChgKw[i] == 0:
            self.essLimMcRegenKw[i] = 0.0

        else:
            if veh.maxMotorKw == self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i]:
                self.essLimMcRegenKw[i] = min(
                    veh.maxMotorKw, self.curMaxEssChgKw[i] / veh.mcFullEffArray[len(veh.mcFullEffArray) - 1])
            else:
                self.essLimMcRegenKw[i] = min(veh.maxMotorKw, self.curMaxEssChgKw[i] / veh.mcFullEffArray
                                                [max(1, np.argmax(veh.mcKwOutArray > min(veh.maxMotorKw - 0.01, self.curMaxEssChgKw[i] - self.curMaxRoadwayChgKw[i])) - 1)])

        self.curMaxMechMcKwIn[i] = min(
            self.essLimMcRegenKw[i], veh.maxMotorKw)
        self.curMaxTracKw[i] = (((veh.wheelCoefOfFric * veh.driveAxleWeightFrac * veh.vehKg * gravityMPerSec2)
                                    / (1 + ((veh.vehCgM * veh.wheelCoefOfFric) / veh.wheelBaseM))) / 1000.0) * (self.maxTracMps[i])

        if veh.fcEffType == 4:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = min(
                    (self.curMaxMcKwOut[i] - self.auxInKw[i]) * veh.transEff, self.curMaxTracKw[i] / veh.transEff)
                self.debug_flag[i] = 1

            else:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] - min(
                    self.curMaxElecKw[i], 0)) * veh.transEff, self.curMaxTracKw[i] / veh.transEff)
                self.debug_flag[i] = 2

        else:

            if veh.noElecSys == 'TRUE' or veh.noElecAux == 'TRUE' or self.highAccFcOnTag[i] == 1:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                self.auxInKw[i]) * veh.transEff, self.curMaxTracKw[i] / veh.transEff)
                self.debug_flag[i] = 3

            else:
                self.curMaxTransKwOut[i] = min((self.curMaxMcKwOut[i] + self.curMaxFcKwOut[i] -
                                                min(self.curMaxElecKw[i], 0)) * veh.transEff, self.curMaxTracKw[i] / veh.transEff)
                self.debug_flag[i] = 4
        
    def get_power_calcs(self, i, veh):
        """Calculate and return power variables.
        Arguments
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step
        
        Output: tarr"""

        self.cycDragKw[i] = 0.5 * airDensityKgPerM3 * veh.dragCoef * \
            veh.frontalAreaM2 * \
            (((self.mpsAch[i-1] + self.cycMps[i]) / 2.0)**3) / 1000.0
        self.cycAccelKw[i] = (veh.vehKg / (2.0 * (self.secs[i]))) * \
            ((self.cycMps[i]**2) - (self.mpsAch[i-1]**2)) / 1000.0
        self.cycAscentKw[i] = gravityMPerSec2 * np.sin(np.arctan(
            self.cycGrade[i])) * veh.vehKg * ((self.mpsAch[i-1] + self.cycMps[i]) / 2.0) / 1000.0
        self.cycTracKwReq[i] = self.cycDragKw[i] + \
            self.cycAccelKw[i] + self.cycAscentKw[i]
        self.spareTracKw[i] = self.curMaxTracKw[i] - self.cycTracKwReq[i]
        self.cycRrKw[i] = gravityMPerSec2 * veh.wheelRrCoef * \
            veh.vehKg * ((self.mpsAch[i-1] + self.cycMps[i]) / 2.0) / 1000.0
        self.cycWheelRadPerSec[i] = self.cycMps[i] / veh.wheelRadiusM
        self.cycTireInertiaKw[i] = (((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * (self.cycWheelRadPerSec[i]**2.0)) / self.secs[i]) -
                                    ((0.5) * veh.wheelInertiaKgM2 * (veh.numWheels * ((self.mpsAch[i-1] / veh.wheelRadiusM)**2.0)) / self.secs[i])) / 1000.0

        self.cycWheelKwReq[i] = self.cycTracKwReq[i] + \
            self.cycRrKw[i] + self.cycTireInertiaKw[i]
        self.regenContrLimKwPerc[i] = veh.maxRegen / (1 + veh.regenA * np.exp(-veh.regenB * (
            (self.cycMph[i] + self.mpsAch[i-1] * mphPerMps) / 2.0 + 1 - 0)))
        self.cycRegenBrakeKw[i] = max(min(
            self.curMaxMechMcKwIn[i] * veh.transEff, self.regenContrLimKwPerc[i] * -self.cycWheelKwReq[i]), 0)
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
        
    def get_speed_dist_calcs(self, i, veh):
        """Calculate variables dependent on speed
        Arguments
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step

        Output: tarr"""

        # Cycle is met
        if self.cycMet[i] == 1:
            self.mpsAch[i] = self.cycMps[i]

        #Cycle is not met
        else:
            Drag3 = (1.0 / 16.0) * airDensityKgPerM3 * \
                veh.dragCoef * veh.frontalAreaM2
            Accel2 = veh.vehKg / (2.0 * (self.secs[i]))
            Drag2 = (3.0 / 16.0) * airDensityKgPerM3 * \
                veh.dragCoef * veh.frontalAreaM2 * self.mpsAch[i-1]
            Wheel2 = 0.5 * veh.wheelInertiaKgM2 * \
                veh.numWheels / (self.secs[i] * (veh.wheelRadiusM**2))
            Drag1 = (3.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * \
                veh.frontalAreaM2 * ((self.mpsAch[i-1])**2)
            Roll1 = (gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg / 2.0)
            Ascent1 = (gravityMPerSec2 *
                        np.sin(np.arctan(self.cycGrade[i])) * veh.vehKg / 2.0)
            Accel0 = - \
                (veh.vehKg * ((self.mpsAch[i-1])**2)) / (2.0 * (self.secs[i]))
            Drag0 = (1.0 / 16.0) * airDensityKgPerM3 * veh.dragCoef * \
                veh.frontalAreaM2 * ((self.mpsAch[i-1])**3)
            Roll0 = (gravityMPerSec2 * veh.wheelRrCoef *
                        veh.vehKg * self.mpsAch[i-1] / 2.0)
            Ascent0 = (
                gravityMPerSec2 * np.sin(np.arctan(self.cycGrade[i])) * veh.vehKg * self.mpsAch[i-1] / 2.0)
            Wheel0 = -((0.5 * veh.wheelInertiaKgM2 * veh.numWheels *
                        (self.mpsAch[i-1]**2)) / (self.secs[i] * (veh.wheelRadiusM**2)))

            Total3 = Drag3 / 1e3
            Total2 = (Accel2 + Drag2 + Wheel2) / 1e3
            Total1 = (Drag1 + Roll1 + Ascent1) / 1e3
            Total0 = (Accel0 + Drag0 + Roll0 + Ascent0 + Wheel0) / \
                1e3 - self.curMaxTransKwOut[i]

            Total = [Total3, Total2, Total1, Total0]
            Total_roots = np.roots(Total)
            ind = np.argmin(abs(self.cycMps[i] - Total_roots))
            self.mpsAch[i] = Total_roots[ind]

        self.mphAch[i] = self.mpsAch[i] * mphPerMps
        self.distMeters[i] = self.mpsAch[i] * self.secs[i]
        self.distMiles[i] = self.distMeters[i] * (1.0 / metersPerMile)
        
    def get_fc_forced_state(self, i, veh):
        """Calculate variables dependent on speed
        Arguments
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step

        Output: tarr, with fcForcedOn and fcForcedState set for timestep i"""

        # force fuel converter on if it was on in the previous time step, but only if fc
        # has not been on longer than minFcTimeOn
        if self.prevfcTimeOn[i] > 0 and self.prevfcTimeOn[i] < veh.minFcTimeOn - self.secs[i]:
            self.fcForcedOn[i] = True
        else:
            self.fcForcedOn[i] = False

        #
        if self.fcForcedOn[i] == False or self.canPowerAllElectrically[i] == False:
            self.fcForcedState[i] = 1
            self.mcMechKw4ForcedFc[i] = 0

        elif self.transKwInAch[i] < 0:
            self.fcForcedState[i] = 2
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i]

        elif veh.maxFcEffKw == self.transKwInAch[i]:
            self.fcForcedState[i] = 3
            self.mcMechKw4ForcedFc[i] = 0

        elif veh.idleFcKw > self.transKwInAch[i] and self.cycAccelKw[i] >= 0:
            self.fcForcedState[i] = 4
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - veh.idleFcKw

        elif veh.maxFcEffKw > self.transKwInAch[i]:
            self.fcForcedState[i] = 5
            self.mcMechKw4ForcedFc[i] = 0

        else:
            self.fcForcedState[i] = 6
            self.mcMechKw4ForcedFc[i] = self.transKwInAch[i] - \
                veh.maxFcEffKw
        
    def get_battery_wear(self, i, veh):
        """Battery wear calcs
        Arguments:
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step
        
        Output: tarr"""

        if veh.noElecSys != 'TRUE':

            if self.essCurKwh[i] > self.essCurKwh[i-1]:
                self.addKwh[i] = (self.essCurKwh[i] -
                                    self.essCurKwh[i-1]) + self.addKwh[i-1]
            else:
                self.addKwh[i] = 0

            if self.addKwh[i] == 0:
                self.dodCycs[i] = self.addKwh[i-1] / veh.maxEssKwh
            else:
                self.dodCycs[i] = 0

            if self.dodCycs[i] != 0:
                self.essPercDeadArray[i] = np.power(
                    veh.essLifeCoefA, 1.0 / veh.essLifeCoefB) / np.power(self.dodCycs[i], 1.0 / veh.essLifeCoefB)
            else:
                self.essPercDeadArray[i] = 0
        
    def get_energy_audit(self, i, veh):
        """Energy Audit Calculations
        Arguments
        ------------
        tarr: instance of SimDrive.TimeArrays()
        i: integer representing index of current time step
        
        Output: tarr"""

        self.dragKw[i] = 0.5 * airDensityKgPerM3 * veh.dragCoef * \
            veh.frontalAreaM2 * \
            (((self.mpsAch[i-1] + self.mpsAch[i]) / 2.0)**3) / 1000.0
        if veh.maxEssKw == 0 or veh.maxEssKwh == 0:
            self.essLossKw[i] = 0
        elif self.essKwOutAch[i] < 0:
            self.essLossKw[i] = -self.essKwOutAch[i] - \
                (-self.essKwOutAch[i] * np.sqrt(veh.essRoundTripEff))
        else:
            self.essLossKw[i] = self.essKwOutAch[i] * \
                (1.0 / np.sqrt(veh.essRoundTripEff)) - self.essKwOutAch[i]
        self.accelKw[i] = (veh.vehKg / (2.0 * (self.secs[i]))) * \
            ((self.mpsAch[i]**2) - (self.mpsAch[i-1]**2)) / 1000.0
        self.ascentKw[i] = gravityMPerSec2 * np.sin(np.arctan(self.cycGrade[i])) * veh.vehKg * (
            (self.mpsAch[i-1] + self.mpsAch[i]) / 2.0) / 1000.0
        self.rrKw[i] = gravityMPerSec2 * veh.wheelRrCoef * veh.vehKg * \
            ((self.mpsAch[i-1] + self.mpsAch[i]) / 2.0) / 1000.0

    # post-processing
    def get_output(self, veh):
        "Calculate Results and Assign Outputs after running"

        output = {}

        if sum(self.fsKwhOutAch) == 0:
            output['mpgge'] = 0

        else:
            output['mpgge'] = sum(self.distMiles) / \
                (sum(self.fsKwhOutAch) * (1 / kWhPerGGE))

        self.roadwayChgKj = sum(self.roadwayChgKwOutAch * self.secs)
        self.essDischKj = - \
            (self.soc[-1] - self.soc[0]) * veh.maxEssKwh * 3600.0
        output['battery_kWh_per_mi'] = (
            self.essDischKj / 3600.0) / sum(self.distMiles)
        self.battery_kWh_per_mi = output['battery_kWh_per_mi']
        output['electric_kWh_per_mi'] = (
            (self.roadwayChgKj + self.essDischKj) / 3600.0) / sum(self.distMiles)
        self.electric_kWh_per_mi = output['electric_kWh_per_mi']
        output['maxTraceMissMph'] = mphPerMps * \
            max(abs(self.cycMps - self.mpsAch))
        self.maxTraceMissMph = output['maxTraceMissMph']
        self.fuelKj = sum(np.asarray(self.fsKwOutAch) * np.asarray(self.secs))
        self.roadwayChgKj = sum(np.asarray(
            self.roadwayChgKwOutAch) * np.asarray(self.secs))
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
        duration_sec = self.cycSecs[-1] - self.cycSecs[0]
        output['avg_speed_mph'] = sum(
            self.distMiles) / (duration_sec / 3600.0)
        self.avg_speed_mph = output['avg_speed_mph']
        self.accel = np.diff(self.mphAch) / np.diff(self.cycSecs)
        output['avg_accel_mphps'] = np.mean(self.accel[self.accel > 0])
        self.avg_accel_mphps = output['avg_accel_mphps']

        if max(self.mphAch) > 60:
            output['ZeroToSixtyTime_secs'] = np.interp(60, self.mphAch, self.cycSecs)

        else:
            output['ZeroToSixtyTime_secs'] = 0.0

        #######################################################################
        ####  Time series information for additional analysis / debugging. ####
        ####             Add parameters of interest as needed.             ####
        #######################################################################

        output['fcKwOutAch'] = np.asarray(self.fcKwOutAch)
        output['fsKwhOutAch'] = np.asarray(self.fsKwhOutAch)
        output['fcKwInAch'] = np.asarray(self.fcKwInAch)
        output['time'] = np.asarray(self.cycSecs)

        return output
