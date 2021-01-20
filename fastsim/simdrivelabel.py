"""Module containing classes and methods for calculating label fuel economy.
For example usage, see ../README.md"""

from numba.cuda.simulator import kernel
from fastsim.parameters import PT_TYPES
import sys
import numpy as np
from scipy.interpolate import interp1d
import re

from fastsim import simdrive, cycle, vehicle, params

cyc_udds = cycle.Cycle('udds')
cyc_hwfet = cycle.Cycle('hwfet')

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
    out['veh'] = veh

    # load the cycles and intstantiate simdrive objects
    if 'VehicleJit' in str(type(veh)):
        cyc['udds'] = cyc_udds.get_numba_cyc()
        cyc['hwy'] = cyc_hwfet.get_numba_cyc()

        sd['udds'] = simdrive.SimDriveJit(cyc['udds'], veh)
        sd['hwy'] = simdrive.SimDriveJit(cyc['hwy'], veh)
        
        out['numba_used'] = True

    else:
        cyc['udds'] = cyc_udds.copy()
        cyc['hwy'] = cyc_hwfet.copy()

        sd['udds'] = simdrive.SimDriveClassic(cyc['udds'], veh)
        sd['hwy'] = simdrive.SimDriveClassic(cyc['hwy'], veh)

        out['numba_used'] = False

    # run simdrive for non-phev powertrains
    sd['udds'].sim_drive()
    sd['hwy'].sim_drive()
    
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
        # assume 2017 coefficients are valid
        adjParams = params.param_dict['LD_FE_Adj_Coef']['2017']
    out['adjParams'] = adjParams

    # this is used only for PHEV
    interpf = interp1d(x=params.param_dict['rechgFreqMiles'],
        y=params.param_dict['ufArray'], 
        kind='previous')

    # run calculations for non-PHEV powertrains
    if params.PT_TYPES[veh.vehPtType] != 'PHEV':
        # lab values

        if params.PT_TYPES[veh.vehPtType] != 'EV':
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
            out['labUddsMpgge'] = sd['udds'].mpgge
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
            out['labHwyMpgge'] = sd['hwy'].mpgge
            out['labCombMpgge'] = 1 / \
                (0.55 / sd['udds'].mpgge + 0.45 / sd['hwy'].mpgge)
        else:
            out['labUddsMpgge'] = 0
            out['labHwyMpgge'] = 0
            out['labCombMpgge'] = 0

        if params.PT_TYPES[veh.vehPtType] == 'EV':
            out['labUddsKwhPerMile'] = sd['udds'].battery_kWh_per_mi
            out['labHwyKwhPerMile'] = sd['hwy'].battery_kWh_per_mi
            out['labCombKwhPerMile'] = 0.55 * sd['udds'].battery_kWh_per_mi + \
                0.45 * sd['hwy'].battery_kWh_per_mi
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
                adjParams['Highway Intercept'] + adjParams['Highway Slope'] / sd['hwy'].mpgge)
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
        def get_label_fe_phev():
            """PHEV-specific function for label fe.  Requires same args as get_label_fe,
            which will be available to it because it is called only inside that namespace."""

            # do PHEV soc iteration
            # This runs 1 cycle starting at max SOC then runs 1 cycle starting at min SOC.
            # By assuming that the battery SOC depletion per mile is constant across cycles,
            # the first cycle can be extrapolated until charge sustaining kicks in.
            
            sd['udds'].sim_drive(veh.maxSoc)
            sd['hwy'].sim_drive(veh.maxSoc)

            phev_calcs = {} # init dict for phev calcs
            phev_calcs['regenSocBuffer'] = min(
                ((0.5 * veh.vehKg * ((60 * (1 / params.mphPerMps)) ** 2)) * (1 / 3600) * (1 / 1000)
                * veh.maxRegen * veh.motorPeakEff) / veh.maxEssKwh,
                (veh.maxSoc - veh.minSoc) / 2
            )

            # charge sustaining behavior
            for key in sd.keys():
                phev_calcs[key] = {} 
                phev_calc = phev_calcs[key] # assigned for for easier syntax
                # charge depletion cycle has already been simulated
                # charge depletion battery kW-hr
                phev_calc['cdBattKwh'] = (
                    veh.maxSoc - veh.minSoc) * veh.maxEssKwh
                # charge depletion fuel gallons
                phev_calc['cdFsGal'] = 0 # sd[key].fsKwhOutAch.sum() / params.kWhPerGGE

                # SOC change during 1 cycle
                phev_calc['deltaSoc'] = (sd[key].soc[0] - sd[key].soc[-1])
                # total number of miles in charge depletion mode, assuming constant kWh_per_mi
                phev_calc['totalCdMiles'] = (veh.maxSoc - veh.minSoc) * \
                    sd[key].veh.maxEssKwh / sd[key].battery_kWh_per_mi
                # float64 number of cycles in charge depletion mode, up to transition
                phev_calc['cdCycs'] = phev_calc['totalCdMiles'] / sd[key].distMiles.sum()
                # fraction of transition cycle spent in charge depletion
                phev_calc['cdFracInTrans'] = phev_calc['cdCycs'] % np.floor(phev_calc['cdCycs'])
                # phev_calc['totalMiles'] = sd[key].distMiles.sum() * (phev_calc['cdCycs'] + (1 - phev_calc['cdFracInTrans']))

                phev_calc['cdFsGal'] = sd[key].fsKwhOutAch.sum() / params.kWhPerGGE
                phev_calc['cdFsKwh'] = sd[key].fsKwhOutAch.sum()
                phev_calc['cd_batt_kWh__mi'] = sd[key].battery_kWh_per_mi
                phev_calc['cd_mpg'] = sd[key].mpgge

                # utility factor calculation for last charge depletion iteration and transition iteration
                # ported from excel
                phev_calc['labIterUf'] = interpf(
                    # might need a plus 1 here
                    np.arange(
                        np.ceil(phev_calc['cdCycs']) + 1) * sd[key].distMiles.sum(),
                )

                # transition cycle
                phev_calc['transInitSoc'] = veh.maxSoc - np.floor(phev_calc['cdCycs']) * phev_calc['deltaSoc']
                
                # run the transition cycle
                sd[key].sim_drive(phev_calc['transInitSoc'])
                # charge depletion battery kW-hr
                phev_calc['transBattKwh'] = (phev_calc['cd_batt_kWh__mi'] * sd[key].distMiles.sum() * 
                    phev_calc['cdFracInTrans'])
                    # (sd[key].soc[0] - sd[key].soc[-1]) * veh.maxEssKwh # not how excel does it
                # charge depletion fuel gallons
                    # sd[key].fsKwhOutAch.sum() / \
                    #     params.kWhPerGGE # not how excel does it
                phev_calc['trans_batt_kWh__mi'] = (phev_calc['cd_batt_kWh__mi'] * 
                    phev_calc['cdFracInTrans'])
                    # sd[key].battery_kWh_per_mi # not how excel does it

                # =IF(AND(ISNUMBER(+@phevLabUddsUf), ISNUMBER(+@phevUddsLabMpg)), (1/+@phevUddsLabMpg)*(+@phevLabUddsUf-R301), IF(AND(ISNUMBER(R301), +@phevUddsUf=""), (1-R301)*(1/+@phevUddsLabMpg), ""))

                # phev_calc['transFsKwh'] = sd[key].fsKwhOutAch.sum()  # not how excel does it

                # charge sustaining
                # the 0.01 is here to be consistent with Excel
                initSoc = sd[key].veh.minSoc + 0.01
                sd[key].sim_drive(initSoc)
                # charge sustaining battery kW-hr
                phev_calc['csBattKwh'] = 0 # (sd[key].soc[0] - sd[key].soc[-1]) * veh.maxEssKwh
                # charge sustaining fuel gallons
                phev_calc['csFsGal'] = sd[key].fsKwhOutAch.sum() / params.kWhPerGGE
                # charge depletion fuel gallons, dependent on phev_calc['transFsGal']
                phev_calc['transFsGal'] = (
                    phev_calc['csFsGal'] * (1 - phev_calc['cdFracInTrans']))
                phev_calc['csFsKwh'] = sd[key].fsKwhOutAch.sum() 
                phev_calc['transFsKwh'] = (
                    phev_calc['csFsKwh'] * (1 - phev_calc['cdFracInTrans']))
                phev_calc['csBattKwh'] = sd[key].essDischgKj
                phev_calc['cs_batt_kWh__mi'] = sd[key].battery_kWh_per_mi

                phev_calc['labUfGpm'] = np.array(
                        [phev_calc['transFsGal'] * np.diff(phev_calc['labIterUf'])[-1],
                        phev_calc['csFsGal'] * (1 - phev_calc['labIterUf'][-1])
                    ]) / sd[key].distMiles.sum()

                phev_calc['cd_mpg'] = sd[key].mpgge

                # note that all values set by `sd[key].set_post_scalars` are relevant only to
                # the final charge sustaining cycle

                # city and highway cycle ranges
                if (veh.maxSoc - phev_calcs['regenSocBuffer'] - sd[key].soc.min()) < 0.01:
                    phev_calc['cdMiles'] = 1000
                else:
                    phev_calc['cdMiles'] = np.ceil(phev_calc['cdCycs']) * sd[key].distMiles.sum()
            
                # city and highway mpg
                # charge depleting
                # phevUddsLabUfGpm = (1/+@phevUddsLabMpg)*(+@phevLabUddsUf-R302)
                # cdLabUddsMpg = 1 / (SUM(IF(ISNUMBER(phevLabUddsUf), phevUddsLabUfGpm)) / MAX(phevLabUddsUf))
                # =IF(AND(ISNUMBER(+@phevLabUddsUf), ISNUMBER(+@phevUddsLabMpg)), 
                #   (1/+@phevUddsLabMpg)*(+@phevLabUddsUf-R302), 
                #       IF(AND(ISNUMBER(R302), +@phevUddsUf=""), 
                #           (1-R302)*(1/+@phevUddsLabMpg), 
                #               ""))
                phev_calc['cdLabMpg'] = phev_calc['labIterUf'][-1] / (
                    phev_calc['transFsGal'] / sd[key].distMiles.sum())

                # charge sustaining
                phev_calc['csMpg'] = sd[key].distMiles.sum() / phev_calc['csFsGal']

                phev_calc['labUf'] = np.float(interpf(phev_calc['cdMiles']))
                                
                # labCombMpgge
                phev_calc['cdMpg'] = max(
                    phev_calc['labIterUf']) / phev_calc['labUfGpm'][-2]

                phev_calc['labMpgge'] = 1 / (phev_calc['labUf'] / phev_calc['cdMpg'] +
                    (1 - phev_calc['labUf']) / phev_calc['csMpg'])

                phev_calc['labIterKwhPerMile'] = np.concatenate(([0], 
                    [phev_calc['cd_batt_kWh__mi']] * int(np.floor(phev_calc['cdCycs'])), 
                    [phev_calc['trans_batt_kWh__mi']], [0]))

                phev_calc['labIterUfKwhPerMile'] = np.concatenate(([0], 
                    phev_calc['labIterKwhPerMile'][1:-1] * np.diff(phev_calc['labIterUf']), 
                    [0]))

                phev_calc['labKwhPerMile'] = phev_calc['labIterUfKwhPerMile'].sum() / \
                    max(phev_calc['labIterUf'])
                
                if key == 'udds':
                    phev_calc['adjIterMpgge'] = np.concatenate((np.zeros(int(np.floor(phev_calc['cdCycs']))), 
                        [max(1 / (adjParams['City Intercept'] + (adjParams['City Slope'] / (sd[key].distMiles.sum() / (phev_calc['transFsKwh'] / params.kWhPerGGE)))), 
                            sd[key].distMiles.sum() / (phev_calc['transFsKwh'] / params.kWhPerGGE) * (1 - params.maxEpaAdj))],
                        [max(1 / (adjParams['City Intercept'] + (adjParams['City Slope'] / (sd[key].distMiles.sum() / (phev_calc['csFsKwh'] / params.kWhPerGGE)))),
                            sd[key].distMiles.sum() / (phev_calc['csFsKwh'] / params.kWhPerGGE) * (1 - params.maxEpaAdj))],
                    ))

                    phev_calc['adjIterKwhPerMile'] = np.zeros(len(phev_calc['labIterKwhPerMile']))
                    for c, _ in enumerate(phev_calc['labIterKwhPerMile']):
                        if phev_calc['labIterKwhPerMile'][c] == 0:
                            phev_calc['adjIterKwhPerMile'][c] = 0
                        else:
                            phev_calc['adjIterKwhPerMile'][c] = (
                                1 / max(1 / (adjParams['City Intercept'] + (adjParams['City Slope'] / (
                                    (1 / phev_calc['labIterKwhPerMile'][c]) * params.kWhPerGGE))), 
                                (1 - params.maxEpaAdj) * ((1 / phev_calc['labIterKwhPerMile'][c]) * params.kWhPerGGE)
                                )) * params.kWhPerGGE

                else:
                    phev_calc['adjIterMpgge'] = np.concatenate((np.zeros(int(np.floor(phev_calc['cdCycs']))), 
                        [max(1 / (adjParams['Highway Intercept'] + (adjParams['Highway Slope'] / (sd[key].distMiles.sum() / (phev_calc['transFsKwh'] / params.kWhPerGGE)))), 
                            sd[key].distMiles.sum() / (phev_calc['transFsKwh'] / params.kWhPerGGE) * (1 - params.maxEpaAdj))],
                        [max(1 / (adjParams['Highway Intercept'] + (adjParams['Highway Slope'] / (sd[key].distMiles.sum() / (phev_calc['csFsKwh'] / params.kWhPerGGE)))),
                            sd[key].distMiles.sum() / (phev_calc['csFsKwh'] / params.kWhPerGGE) * (1 - params.maxEpaAdj))],
                    ))

                    phev_calc['adjIterKwhPerMile']=np.zeros(
                        len(phev_calc['labIterKwhPerMile']))
                    for c, _ in enumerate(phev_calc['labIterKwhPerMile']):
                        if phev_calc['labIterKwhPerMile'][c] == 0:
                            phev_calc['adjIterKwhPerMile'][c]=0
                        else:
                            phev_calc['adjIterKwhPerMile'][c]=(
                                1 / max(1 / (adjParams['Highway Intercept'] + (adjParams['Highway Slope'] / (
                                    (1 / phev_calc['labIterKwhPerMile'][c]) * params.kWhPerGGE))),
                                    (1 - params.maxEpaAdj) * ((1 / phev_calc['labIterKwhPerMile'][c]) * params.kWhPerGGE)
                                )) * params.kWhPerGGE

                phev_calc['adjIterCdMiles'] = np.zeros(
                    int(np.ceil(phev_calc['cdCycs'])) + 2)
                for c, _ in enumerate(phev_calc['adjIterCdMiles']):
                    if c == 0:
                        phev_calc['adjIterCdMiles'][c] = 0
                    elif c <= np.floor(phev_calc['cdCycs']):
                        phev_calc['adjIterCdMiles'][c] = (
                            phev_calc['adjIterCdMiles'][c-1] + phev_calc['cd_batt_kWh__mi'] * 
                            sd[key].distMiles.sum() / phev_calc['adjIterKwhPerMile'][c])
                    elif c == np.floor(phev_calc['cdCycs']) + 1:
                        phev_calc['adjIterCdMiles'][c] = (
                            phev_calc['adjIterCdMiles'][c-1] + phev_calc['trans_batt_kWh__mi'] *
                            sd[key].distMiles.sum() / phev_calc['adjIterKwhPerMile'][c])
                    else:
                        phev_calc['adjIterCdMiles'][c] = 0

                if (veh.maxSoc - phev_calcs['regenSocBuffer'] - sd[key].soc.min() < 0.01):
                    phev_calc['adjCdMiles'] = 1000
                else:
                    phev_calc['adjCdMiles'] = phev_calc['adjIterCdMiles'].max()

                # utility factor calculation for last charge depletion iteration and transition iteration
                # ported from excel
                phev_calc['adjIterUf'] = interpf(
                    # might need a plus 1 here
                    phev_calc['adjIterCdMiles']
                )

                phev_calc['adjIterUfGpm'] = np.concatenate(
                    (np.zeros(int(np.floor(phev_calc['cdCycs']))),
                    [(1 / phev_calc['adjIterMpgge'][-2]) * np.diff(phev_calc['adjIterUf'])[-2]],
                    [(1 / phev_calc['adjIterMpgge'][-1]) * (1 - phev_calc['adjIterUf'][-2])]
                    ))

                phev_calc['adjIterUfKwhPerMile'] = phev_calc['adjIterKwhPerMile'] * \
                    np.concatenate(([0], np.diff(phev_calc['adjIterUf'])))

                phev_calc['adjCdMpgge'] = 1 / \
                    phev_calc['adjIterUfGpm'][-2] * max(phev_calc['adjIterUf'])
                phev_calc['adjCsMpgge'] = 1 / \
                    phev_calc['adjIterUfGpm'][-1] * (1 - max(phev_calc['adjIterUf']))

                phev_calc['adjUf'] = np.float(interpf(phev_calc['adjCdMiles']))

                phev_calc['adjMpgge'] = 1 / (phev_calc['adjUf'] / phev_calc['adjCdMpgge'] +
                    (1 - phev_calc['adjUf']) / phev_calc['adjCsMpgge'])
                
                # adjCdUddsMpg
                # adjCdHwyMpg
                # adjCsUddsMpg
                # adjCsHwyMpg

                # labUddsUf
                # labHwyUf
                # labCombMpgge
                # labCombKwhPerMile
                # labUddsEssKwhPerMile
                # labHwyEssKwhPerMile

            out['phev_calcs'] = phev_calcs

            # efficiency-related calculations
            # lab
            out['labUddsMpgge'] = phev_calcs['udds']['labMpgge']
            out['labHwyMpgge'] = phev_calcs['hwy']['labMpgge']
            out['labCombMpgge'] = 1 / (
                0.55 / phev_calcs['udds']['labMpgge'] + 0.45 / phev_calcs['hwy']['labMpgge'])

            out['labUddsKwhPerMile'] = phev_calcs['udds']['labKwhPerMile']
            out['labHwyKwhPerMile'] = phev_calcs['hwy']['labKwhPerMile']
            out['labCombKwhPerMile'] =  0.55 * phev_calcs['udds']['labKwhPerMile'] + \
                0.45 * phev_calcs['hwy']['labKwhPerMile']

            # adjusted
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjUddsMpgge
            out['adjUddsMpgge'] = 666
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjHwyMpgge
            out['adjHwyMpgge'] = 666
            out['adjCombMpgge'] = 666

            # out['adjUddsKwhPerMile'] =  sum(if(isnumber(phevUddsAdjUfKwhPerMile), phevUddsAdjUfKwhPerMile)) / max(phevUddsUf) / chgEff
            out['adjUddsKwhPerMile'] = 666
            out['adjHwyKwhPerMile'] = 666
            out['adjCombKwhPerMile'] = 666

            out['adjUddsEssKwhPerMile'] = out['adjUddsKwhPerMile'] * params.chgEff
            out['adjHwyEssKwhPerMile'] = out['adjHwyKwhPerMile'] * params.chgEff
            out['adjCombEssKwhPerMile'] = out['adjCombKwhPerMile'] * params.chgEff

            # range for combined city/highway
            out['rangeMiles'] = 666
            # utility factor (percent driving in charge depletion mode)
            out['UF'] = 666

        get_label_fe_phev()

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
        return out
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
    
