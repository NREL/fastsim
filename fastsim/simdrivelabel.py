"""Module containing classes and methods for calculating label fuel economy.
For example usage, see ../README.md"""

import sys
import numpy as np

from fastsim import simdrive, cycle, vehicle, params

def get_label_fe(veh, full_detail=False, verbose=False):
    """Generates label fuel economy (FE) values for a provided vehicle.
    
    Arguments:
    ----------
    veh : vehicle.Vehicle() or vehicle.TypedVehicle() instance.  
        If TypedVehicle instance is provided, simdrive.SimDriveJit() instance will be used.  
        Otherwise, simdrive.SimDriveClassic() instance will be used.
    full_detail : boolean, default False
        If True, sim_drive objects for each cycle are also returned.  
    verbose : boolean, default false
        If true, print out key results
        
    Returns label fuel economy values as a dict and (optionally)
    simdrive.SimDriveJit objects."""

    cyc = {}
    sd = {}
    out = {}

    # load the cycles and intstantiate simdrive objects
    if 'TypedVehicle' in str(type(veh)):
        cyc['udds'] = cycle.Cycle('udds').get_numba_cyc()
        cyc['hwfet'] = cycle.Cycle('hwfet').get_numba_cyc()

        sd['udds'] = simdrive.SimDriveJit(cyc['udds'], veh)
        sd['hwfet'] = simdrive.SimDriveJit(cyc['hwfet'], veh)

    else:
        cyc['udds'] = cycle.Cycle('udds')
        cyc['hwfet'] = cycle.Cycle('hwfet')

        sd['udds'] = simdrive.SimDriveClassic(cyc['udds'], veh)
        sd['hwfet'] = simdrive.SimDriveClassic(cyc['hwfet'], veh)

    # run simdrive for non-phev powertrains
    sd['udds'].sim_drive()
    sd['hwfet'].sim_drive()

    # run calculations for non-PHEV powertrains
    if params.PT_TYPES[veh.vehPtType] != 'PHEV':
        # efficiency-related calculations
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
        out['labUddsMpgge'] = sd['udds'].mpgge
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
        out['labHwyMpgge'] = sd['hwfet'].mpgge
        try:
            out['labCombMpgge'] = 1 / \
                (0.55 / sd['udds'].mpgge + 0.45 / sd['hwfet'].mpgge)
        except:
            out['labCombMpgge'] = 0

        out['labUddsKwhPerMile'] = sd['udds'].battery_kWh_per_mi
        out['labHwyKwhPerMile'] = sd['hwfet'].battery_kWh_per_mi
        out['labCombKwhPerMile'] = 0.55 * sd['udds'].battery_kWh_per_mi + \
            0.45 * sd['hwfet'].battery_kWh_per_mi

        # adjusted combined city/highway mpg
        out['adjCombMpgge'] = 666
        # range for combined city/highway
        out['rangeMiles'] = 666
        # utility factor (percent driving in charge depletion mode)
        out['UF'] = 666
        # adjusted combined city/highway kW-hr/mi
        out['adjCombKwhPerMile'] = 666

    else:
        # do PHEV soc iteration
        # This runs 1 cycle starting at max SOC then runs 1 cycle starting at min SOC.
        # By assuming that the battery SOC depletion per mile is constant across cycles,
        # the first cycle can be extrapolated until charge sustaining kicks in.

        for key in sd.keys():
            sd[key].sim_drive()
        
        # charge depletion battery kW-hr
        cdBattKwh = sd[key].essDischgKj / 3600.0
        # charge depletion fuel gallons
        cdFsGal = sd[key].fsKwhOutAch.sum() / params.kWhPerGGE

        # # from excel where prevMiles `C298:`
        # udds_mpg_thing = max(
        #     1 / (cityIntercept + (citySlope / (sd[key].distMiles.sum() / (phevUddsGasKwh / params.kWhPerGGE)))),
        #     sd[key].distMiles.sum() / (phevUddsGasKwh / params.kWhPerGGE) * (1 - params.maxEpaAdj)
        #     )

        # udds_kwHr__mi_thing = =IF(ISNUMBER(+@phevUddsElecKwh),IF(+@phevUddsElecKwh=0,0,(1/MAX(1/(cityIntercept+(citySlope/((1/+@phevUddsLabKwhPerMile)*'Veh Model'!kWhPerGGE))),(1-'Veh Model'!maxEpaAdj)*((+@phevUddsMiles-C301)/(+@phevUddsElecKwh*(1/'Veh Model'!kWhPerGGE)))))*'Veh Model'!kWhPerGGE),"")

        # SOC change during 1 cycle
        deltaSoc = (sd[key].veh.maxSoc - sd[key].veh.minSoc)
        # total number of miles in charge depletion mode, assuming constant kWh_per_mi
        totalCdMiles = deltaSoc * \
            sd[key].veh.maxEssKwh / sd[key].battery_kWh_per_mi
        # float64 number of cycles in charge depletion mode, up to transition
        cdCycs = totalCdMiles / sd[key].distMiles.sum()
        # fraction of transition cycle spent in charge depletion
        cdFracInTrans = cdCycs % np.floor(cdCycs)
        totalMiles = sd[key].distMiles.sum() * (cdCycs + (1 - cdFracInTrans))

        # first cycle that ends in charge sustaining behavior
        initSoc = sd[key].veh.minSoc + 0.01  # the 0.01 is here to be consistent with Excel
        sd[key].sim_drive(initSoc)
        # charge depletion battery kW-hr
        csBattKwh = sd[key].essDischgKj / 3600.0
        # charge depletion fuel gallons
        csFsGal = sd[key].fsKwhOutAch.sum() / params.kWhPerGGE

        # note that all values set by `sd[key].set_post_scalars` are relevant only to
        # the final charge sustaining cycle

        # harmonic average of charged depletion, transition, and charge sustaining phases
        sd[key].mpgge = totalMiles / (cdFsGal * cdCycs + csFsGal)
        sd[key].battery_kWh_per_mi = (
            cdBattKwh * cdCycs + csBattKwh) / totalMiles

        # efficiency-related calculations
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
        out['labUddsMpgge'] = 666
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
        out['labHwyMpgge'] = 666
        out['labCombMpgge'] = 666

        out['labUddsKwhPerMile'] = 666
        out['labHwyKwhPerMile'] = 666
        out['labCombKwhPerMile'] = 666

        # adjusted combined city/highway mpg
        out['adjCombMpgge'] = 666
        # range for combined city/highway
        out['rangeMiles'] = 666
        # utility factor (percent driving in charge depletion mode)
        out['UF'] = 666
        # adjusted combined city/highway kW-hr/mi
        out['adjCombKwhPerMile'] = 666

    # non-efficiency-related calculations
    # zero-to-sixty time
    out['accelSecs'] = 666
    
    # success Boolea -- did all of the tests work(e.g. met trace within ~2 mph)?
    out['resFound'] = True # this may need fancier logic

    if full_detail and verbose:
        for key in out.keys():
            print(key + f': {out[key]:.5g}')
        return out, sd
    elif full_detail:
        return out, sd
    elif verbose:
        for key in out.keys():
            print(key + f': {out[key]:.5g}')
        return outtn
    else:
        return out

if __name__ == '__main__':
    if len(sys.argv) > 1:
        veh = vehicle.Vehicle(sys.argv[1])
    else:
        veh = vehicle.Vehicle(1) # load default vehicle

    out = get_label_fe(veh)
    for key in out.keys():
        print(key + f': {out[key]:.5g}')
    
