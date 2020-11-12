"""Module containing classes and methods for calculating label fuel economy.
For example usage, see ../README.md"""

import sys
import numpy as np
import re

from fastsim import simdrive, cycle, vehicle, params

def get_label_fe(veh, full_detail=False, verbose=False, chgEff=None):
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
    chgEff : float between 0 and 1
        Override for chgEff -- currently not functional
        
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
    
    # find year-based adjustment parameters
    # re is for vehicle model year if Scenario_name contains a 4 digit string
    if re.match('\d{4}', veh.Scenario_name):
        vehYear = np.float32(
            re.match('\d{4}', veh.Scenario_name).group()
        )
        if vehYear < 2017:
            adjParams = params.param_dict['LD_FE_Adj_Coef']['2008']
        else:
            adjParams = params.param_dict['LD_FE_Adj_Coef']['2017']
    else:
        adjParams = params.param_dict['LD_FE_Adj_Coef']['2017']
    out['adjParams'] = adjParams

    # run calculations for non-PHEV powertrains
    if params.PT_TYPES[veh.vehPtType] != 'PHEV':
        # lab values

        if params.PT_TYPES[veh.vehPtType] != 'EV':
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
            out['labUddsMpgge'] = sd['udds'].mpgge
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
            out['labHwyMpgge'] = sd['hwfet'].mpgge
            out['labCombMpgge'] = 1 / \
                (0.55 / sd['udds'].mpgge + 0.45 / sd['hwfet'].mpgge)
        else:
            out['labUddsMpgge'] = 0
            out['labHwyMpgge'] = 0
            out['labCombMpgge'] = 0

        if params.PT_TYPES[veh.vehPtType] == 'EV':
            out['labUddsKwhPerMile'] = sd['udds'].battery_kWh_per_mi
            out['labHwyKwhPerMile'] = sd['hwfet'].battery_kWh_per_mi
            out['labCombKwhPerMile'] = 0.55 * sd['udds'].battery_kWh_per_mi + \
                0.45 * sd['hwfet'].battery_kWh_per_mi
        else:
            out['labUddsKwhPerMile'] = 0
            out['labHwyKwhPerMile'] = 0
            out['labCombKwhPerMile'] = 0

        # adjusted values for mpg
        if params.PT_TYPES[veh.vehPtType] != 'EV': # non-EV case
            # CV or HEV case (not PHEV)
            # HEV SOC iteration is handled in simdrive.SimDriveClassic
            out['adjUddsMpgge'] = 1 / (
                adjParams['City Intercept'] + adjParams['City Slope'] / sd['udds'].mpgge)
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjHwyMpgge
            out['adjHwyMpgge'] = 1 / (
                adjParams['Highway Intercept'] + adjParams['Highway Slope'] / sd['hwfet'].mpgge)
            out['adjCombMpgge'] = 1 / \
                (0.55 / out['adjUddsMpgge'] + 0.45 / out['adjHwyMpgge'])
        else: # EV case
            # Mpgge is all zero for EV
            zero_keys = ['labUddsMpgge', 'labHwyMpgge', 'labCombMpgge',
                         'adjUddsMpgge', 'adjHwyMpgge', 'adjCombMpgge']
            for key in zero_keys:
                out[key] = 0
            
        # adjusted kW-hr/mi
        if params.PT_TYPES[veh.vehPtType] == "EV": # EV Case
            out['adjUddsKwhPerMile'] = (1 / max(
                (1 / (adjParams['City Intercept'] + (adjParams['City Slope'] / ((1 / out['labUddsKwhPerMile']) * params.kWhPerGGE)))),
                (1 / out['labUddsKwhPerMile']) * params.kWhPerGGE * (1 - params.maxEpaAdj))
                ) * params.kWhPerGGE / params.chgEff 
            out['adjHwyKwhPerMile'] = (1 / max(
                (1 / (adjParams['Highway Intercept'] + (adjParams['Highway Slope'] / ((1 / out['labHwyKwhPerMile']) * params.kWhPerGGE)))),
                (1 / out['labHwyKwhPerMile']) * params.kWhPerGGE * (1 - params.maxEpaAdj))
                ) * params.kWhPerGGE / params.chgEff 
            out['adjCombKwhPerMile'] = 0.55 * out['adjUddsKwhPerMile'] + \
                0.45 * out['adjHwyKwhPerMile']

            out['adjUddsEssKwhPerMile'] = out['adjUddsKwhPerMile'] * params.chgEff
            out['adjHwyEssKwhPerMile'] = out['adjHwyKwhPerMile'] * params.chgEff
            out['adjCombEssKwhPerMile'] = out['adjCombKwhPerMile'] * params.chgEff

            # range for combined city/highway
            out['rangeMiles'] = veh.maxEssKwh / out['adjCombEssKwhPerMile']

        else: # non-PEV cases
            zero_keys = ['adjUddsKwhPerMile', 'adjHwyKwhPerMile', 'adjCombKwhPerMile',
                         'adjUddsEssKwhPerMile', 'adjHwyEssKwhPerMile', 'adjCombEssKwhPerMile'
                         'rangeMiles',]
            for key in zero_keys:
                out[key] = 0

        # utility factor (percent driving in PHEV charge depletion mode)
        out['UF'] = 0

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
        # lab
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
        out['labUddsMpgge'] = 666
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
        out['labHwyMpgge'] = 666
        out['labCombMpgge'] = 666

        out['labUddsKwhPerMile'] = 666
        out['labHwyKwhPerMile'] = 666
        out['labCombKwhPerMile'] = 666

        # adjusted
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjUddsMpgge
        out['adjUddsMpgge'] = 666
        # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjHwyMpgge
        out['adjHwyMpgge'] = 666
        out['adjCombMpgge'] = 666

        # out['adjUddsKwhPerMile'] =  sum(if(isnumber(phevUddsAdjUfKwhPerMile), phevUddsAdjUfKwhPerMile)) / max(phevUddsUf) / chgEff
        out['adjUddsKwhPerMile'] = 1 / (max(
            (1 / (adjParams['City Intercept'] + (adjParams['City Slope'] /
                                                 (1 / out['labUddsKwhPerMile'] * params.kWhPerGGE)))), 666))
        out['adjHwyKwhPerMile'] = 666
        out['adjCombKwhPerMile'] = 666

        out['adjUddsEssKwhPerMile'] = out['adjUddsKwhPerMile'] * params.chgEff
        out['adjHwyEssKwhPerMile'] = out['adjHwyKwhPerMile'] * params.chgEff
        out['adjCombEssKwhPerMile'] = out['adjCombKwhPerMile'] * params.chgEff

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
    
    # success Boolean -- did all of the tests work(e.g. met trace within ~2 mph)?
    out['resFound'] = True # this may need fancier logic than just always being true

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
    
