"""Module containing classes and methods for calculating label fuel economy."""

import numpy as np
from scipy.interpolate import interp1d

from fastsim import simdrive, cycle, vehicle
from fastsim import parameters as params
from fastsim.auxiliaries import set_nested_values

cyc_udds = cycle.Cycle.from_file('udds')
cyc_hwfet = cycle.Cycle.from_file('hwfet')

sim_params = simdrive.SimDriveParams()
props = params.PhysicalProperties()

def get_label_fe(veh:vehicle.Vehicle, full_detail:bool=False, verbose:bool=False, chg_eff:float=None, use_rust=False):
    """Generates label fuel economy (FE) values for a provided vehicle.
    
    Arguments:
    ----------
    veh : vehicle.Vehicle()
    full_detail : boolean, default False
        If True, sim_drive objects for each cycle are also returned.  
    verbose : boolean, default false
        If true, print out key results
    chg_eff : float between 0 and 1
        Override for chg_eff -- currently not functional
    use_rust: bool, if True, use rust version of classes, else Python
        
    Returns label fuel economy values as a dict and (optionally)
    simdrive.SimDriveClassic objects."""

    cyc = {}
    sd = {}
    out = {}
    def maybe_to_rust(obj):
        if use_rust:
            return obj.to_rust()
        return obj
    
    def make_simdrive(*args, **kwargs):
        if use_rust:
            return simdrive.RustSimDrive(*args, **kwargs)
        return simdrive.SimDrive(*args, **kwargs)

    out['veh'] = maybe_to_rust(veh) if use_rust else veh

    # load the cycles and intstantiate simdrive objects
    accel_cyc_secs = np.arange(0, 300, 0.1)
    cyc_dict = {'time_s': accel_cyc_secs,
                'mps': np.append([0],
                np.ones(len(accel_cyc_secs) - 1) * 90 / params.MPH_PER_MPS)}

    cyc['accel'] = maybe_to_rust(cycle.Cycle.from_dict(cyc_dict))
    cyc['udds'] = maybe_to_rust(cycle.copy_cycle(cyc_udds))
    cyc['hwy'] = maybe_to_rust(cycle.copy_cycle(cyc_hwfet))

    sd['udds'] = make_simdrive(cyc['udds'], veh)
    sd['hwy'] = make_simdrive(cyc['hwy'], veh)

    # run simdrive for non-phev powertrains
    sd['udds'].sim_params = set_nested_values(sd['udds'].sim_params)
    sd['udds'].sim_drive()
    sd['hwy'].sim_params = set_nested_values(sd['hwy'].sim_params)
    sd['hwy'].sim_drive()
    
    # find year-based adjustment parameters
    if veh.veh_year < 2017:
        adj_params = params.param_dict['LD_FE_Adj_Coef']['2008']
    else:
        # assume 2017 coefficients are valid
        adj_params = params.param_dict['LD_FE_Adj_Coef']['2017']
    out['adjParams'] = adj_params

    # this is used only for PHEV
    interpf = interp1d(x=params.param_dict['rechgFreqMiles'],
        y=params.param_dict['ufArray'], 
        kind='previous')

    # run calculations for non-PHEV powertrains
    if veh.veh_pt_type != vehicle.PHEV:
        # lab values

        if veh.veh_pt_type != vehicle.BEV:
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

        if veh.veh_pt_type == vehicle.BEV:
            out['labUddsKwhPerMile'] = sd['udds'].battery_kwh_per_mi
            out['labHwyKwhPerMile'] = sd['hwy'].battery_kwh_per_mi
            out['labCombKwhPerMile'] = 0.55 * sd['udds'].battery_kwh_per_mi + \
                0.45 * sd['hwy'].battery_kwh_per_mi
        else:
            out['labUddsKwhPerMile'] = 0
            out['labHwyKwhPerMile'] = 0
            out['labCombKwhPerMile'] = 0

        # adjusted values for mpg
        if veh.veh_pt_type != vehicle.BEV: # non-EV case
            # CV or HEV case (not PHEV)
            # HEV SOC iteration is handled in simdrive.SimDriveClassic
            out['adjUddsMpgge'] = 1 / (
                adj_params['City Intercept'] + adj_params['City Slope'] / sd['udds'].mpgge)
            # compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjHwyMpgge
            out['adjHwyMpgge'] = 1 / (
                adj_params['Highway Intercept'] + adj_params['Highway Slope'] / sd['hwy'].mpgge)
            out['adjCombMpgge'] = 1 / \
                (0.55 / out['adjUddsMpgge'] + 0.45 / out['adjHwyMpgge'])
        else: # EV case 
            # Mpgge is all zero for EV
            zero_keys = ['labUddsMpgge', 'labHwyMpgge', 'labCombMpgge',
                         'adjUddsMpgge', 'adjHwyMpgge', 'adjCombMpgge']
            for key in zero_keys:
                out[key] = 0
            
        # adjusted kW-hr/mi
        if veh.veh_pt_type == vehicle.BEV: # EV Case
            out['adjUddsKwhPerMile'] = (1 / max(
                (1 / (adj_params['City Intercept'] + (adj_params['City Slope'] / ((1 / out['labUddsKwhPerMile']) * props.kwh_per_gge)))),
                (1 / out['labUddsKwhPerMile']) * props.kwh_per_gge * (1 - sim_params.max_epa_adj))
                ) * props.kwh_per_gge / params.chg_eff 
            out['adjHwyKwhPerMile'] = (1 / max(
                (1 / (adj_params['Highway Intercept'] + (adj_params['Highway Slope'] / ((1 / out['labHwyKwhPerMile']) * props.kwh_per_gge)))),
                (1 / out['labHwyKwhPerMile']) * props.kwh_per_gge * (1 - sim_params.max_epa_adj))
                ) * props.kwh_per_gge / params.chg_eff 
            out['adjCombKwhPerMile'] = 0.55 * out['adjUddsKwhPerMile'] + \
                0.45 * out['adjHwyKwhPerMile']

            out['adjUddsEssKwhPerMile'] = out['adjUddsKwhPerMile'] * params.chg_eff
            out['adjHwyEssKwhPerMile'] = out['adjHwyKwhPerMile'] * params.chg_eff
            out['adjCombEssKwhPerMile'] = out['adjCombKwhPerMile'] * params.chg_eff

            # range for combined city/highway
            out['netRangeMiles'] = veh.ess_max_kwh / out['adjCombEssKwhPerMile']

        else: # non-PEV cases
            zero_keys = ['adjUddsKwhPerMile', 'adjHwyKwhPerMile', 'adjCombKwhPerMile',
                         'adjUddsEssKwhPerMile', 'adjHwyEssKwhPerMile', 'adjCombEssKwhPerMile'
                         'netRangeMiles',]
            for key in zero_keys:
                out[key] = 0

        # utility factor (percent driving in PHEV charge depletion mode)
        out['UF'] = 0

    else:
        # PHEV
        def get_label_fe_phev():
            """PHEV-specific function for label fe.  Requires same args as get_label_fe,
            which will be available to it because it is called only inside that namespace."""

            # do PHEV soc iteration
            # This runs 1 cycle starting at max SOC then runs 1 cycle starting at min SOC.
            # By assuming that the battery SOC depletion per mile is constant across cycles,
            # the first cycle can be extrapolated until charge sustaining kicks in.
            sd['udds'].sim_params = set_nested_values(sd['udds'].sim_params) 
            sd['udds'].sim_drive(veh.max_soc)
            sd['hwy'].sim_params = set_nested_values(sd['hwy'].sim_params)
            sd['hwy'].sim_drive(veh.max_soc)

            phev_calcs = {} # init dict for phev calcs
            phev_calcs['regenSocBuffer'] = min(
                ((0.5 * veh.veh_kg * ((60 * (1 / params.MPH_PER_MPS)) ** 2)) * (1 / 3600) * (1 / 1000)
                * veh.max_regen * veh.mc_peak_eff) / veh.ess_max_kwh,
                (veh.max_soc - veh.min_soc) / 2
            )

            # charge sustaining behavior
            for key in sd.keys():
                phev_calcs[key] = {} 
                phev_calc = phev_calcs[key] # assigned for for easier syntax
                # charge depletion cycle has already been simulated
                # charge depletion battery kW-hr
                phev_calc['cdEssKwh'] = (
                    veh.max_soc - veh.min_soc) * veh.ess_max_kwh
                # charge depletion fuel gallons
                phev_calc['cdFsGal'] = 0 # sd[key].fsKwhOutAch.sum() / props.kWhPerGGE

                # SOC change during 1 cycle
                phev_calc['deltaSoc'] = (sd[key].soc[0] - sd[key].soc[-1])
                # total number of miles in charge depletion mode, assuming constant kWh_per_mi
                phev_calc['totalCdMiles'] = (veh.max_soc - veh.min_soc) * \
                    sd[key].veh.ess_max_kwh / sd[key].battery_kwh_per_mi
                # float64 number of cycles in charge depletion mode, up to transition
                phev_calc['cdCycs'] = np.array(phev_calc['totalCdMiles']) / np.array(sd[key].dist_mi).sum()
                # fraction of transition cycle spent in charge depletion
                phev_calc['cdFracInTrans'] = np.array(phev_calc['cdCycs']) % np.floor(phev_calc['cdCycs'])
                # phev_calc['totalMiles'] = sd[key].distMiles.sum() * (phev_calc['cdCycs'] + (1 - phev_calc['cdFracInTrans']))

                phev_calc['cdFsGal'] = np.array(sd[key].fs_kwh_out_ach).sum() / props.kwh_per_gge
                phev_calc['cdFsKwh'] = np.array(sd[key].fs_kwh_out_ach).sum()
                phev_calc['cd_ess_kWh__mi'] = sd[key].battery_kwh_per_mi
                phev_calc['cd_mpg'] = sd[key].mpgge

                # utility factor calculation for last charge depletion iteration and transition iteration
                # ported from excel
                phev_calc['labIterUf'] = interpf(
                    # might need a plus 1 here
                    np.arange(
                        np.ceil(phev_calc['cdCycs']) + 1) * np.array(sd[key].dist_mi).sum(),
                )

                # transition cycle
                phev_calc['transInitSoc'] = veh.max_soc - np.floor(phev_calc['cdCycs']) * phev_calc['deltaSoc']
                
                # run the transition cycle
                sd[key].sim_params = set_nested_values(sd[key].sim_params) 
                sd[key].sim_drive(phev_calc['transInitSoc'])
                # charge depletion battery kW-hr
                phev_calc['transEssKwh'] = (phev_calc['cd_ess_kWh__mi'] * np.array(sd[key].dist_mi).sum() * 
                    phev_calc['cdFracInTrans'])
                    # (sd[key].soc[0] - sd[key].soc[-1]) * veh.ess_max_kwh # not how excel does it
                # charge depletion fuel gallons
                    # sd[key].fsKwhOutAch.sum() / \
                    #     props.kWhPerGGE # not how excel does it
                phev_calc['trans_ess_kWh__mi'] = (phev_calc['cd_ess_kWh__mi'] * 
                    phev_calc['cdFracInTrans'])
                    # sd[key].battery_kWh_per_mi # not how excel does it

                # =IF(AND(ISNUMBER(+@phevLabUddsUf), ISNUMBER(+@phevUddsLabMpg)), (1/+@phevUddsLabMpg)*(+@phevLabUddsUf-R301), IF(AND(ISNUMBER(R301), +@phevUddsUf=""), (1-R301)*(1/+@phevUddsLabMpg), ""))

                # phev_calc['transFsKwh'] = sd[key].fsKwhOutAch.sum()  # not how excel does it

                # charge sustaining
                # the 0.01 is here to be consistent with Excel
                initSoc = sd[key].veh.min_soc + 0.01
                sd[key].sim_params = set_nested_values(sd[key].sim_params) 
                sd[key].sim_drive(initSoc)
                # charge sustaining battery kW-hr
                phev_calc['csEssKwh'] = 0 # (sd[key].soc[0] - sd[key].soc[-1]) * veh.ess_max_kwh
                # charge sustaining fuel gallons
                phev_calc['csFsGal'] = np.array(sd[key].fs_kwh_out_ach).sum() / props.kwh_per_gge
                # charge depletion fuel gallons, dependent on phev_calc['transFsGal']
                phev_calc['transFsGal'] = (
                    phev_calc['csFsGal'] * (1 - phev_calc['cdFracInTrans']))
                phev_calc['csFsKwh'] = np.array(sd[key].fs_kwh_out_ach).sum() 
                phev_calc['transFsKwh'] = (
                    phev_calc['csFsKwh'] * (1 - phev_calc['cdFracInTrans']))
                phev_calc['csEssKwh'] = sd[key].ess_dischg_kj
                phev_calc['cs_ess_kWh__mi'] = sd[key].battery_kwh_per_mi

                phev_calc['labUfGpm'] = np.array(
                        [phev_calc['transFsGal'] * np.diff(phev_calc['labIterUf'])[-1],
                        phev_calc['csFsGal'] * (1 - phev_calc['labIterUf'][-1])
                    ]) / np.array(sd[key].dist_mi).sum()

                phev_calc['cd_mpg'] = sd[key].mpgge

                # note that all values set by `sd[key].set_post_scalars` are relevant only to
                # the final charge sustaining cycle

                # city and highway cycle ranges
                if (veh.max_soc - phev_calcs['regenSocBuffer'] - np.array(sd[key].soc).min()) < 0.01:
                    phev_calc['cdMiles'] = 1000
                else:
                    phev_calc['cdMiles'] = np.ceil(phev_calc['cdCycs']) * np.array(sd[key].dist_mi).sum()
            
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
                    phev_calc['transFsGal'] / np.array(sd[key].dist_mi).sum())

                # charge sustaining
                phev_calc['csMpg'] = np.array(sd[key].dist_mi).sum() / phev_calc['csFsGal']

                phev_calc['labUf'] = float(interpf(phev_calc['cdMiles']))
                                
                # labCombMpgge
                phev_calc['cdMpg'] = max(
                    phev_calc['labIterUf']) / phev_calc['labUfGpm'][-2]

                phev_calc['labMpgge'] = 1 / (phev_calc['labUf'] / phev_calc['cdMpg'] +
                    (1 - phev_calc['labUf']) / phev_calc['csMpg'])

                phev_calc['labIterKwhPerMile'] = np.concatenate(([0], 
                    [phev_calc['cd_ess_kWh__mi']] * int(np.floor(phev_calc['cdCycs'])), 
                    [phev_calc['trans_ess_kWh__mi']], [0]))

                phev_calc['labIterUfKwhPerMile'] = np.concatenate(([0], 
                    phev_calc['labIterKwhPerMile'][1:-1] * np.diff(phev_calc['labIterUf']), 
                    [0]))

                phev_calc['labKwhPerMile'] = phev_calc['labIterUfKwhPerMile'].sum() / \
                    max(phev_calc['labIterUf'])
                
                if key == 'udds':
                    phev_calc['adjIterMpgge'] = np.concatenate((np.zeros(int(np.floor(phev_calc['cdCycs']))), 
                        [max(1 / (adj_params['City Intercept'] + (adj_params['City Slope'] / (np.array(sd[key].dist_mi).sum() / (phev_calc['transFsKwh'] / props.kwh_per_gge)))), 
                            np.array(sd[key].dist_mi).sum() / (phev_calc['transFsKwh'] / props.kwh_per_gge) * (1 - sim_params.max_epa_adj))],
                        [max(1 / (adj_params['City Intercept'] + (adj_params['City Slope'] / (np.array(sd[key].dist_mi).sum() / (phev_calc['csFsKwh'] / props.kwh_per_gge)))),
                            np.array(sd[key].dist_mi).sum() / (phev_calc['csFsKwh'] / props.kwh_per_gge) * (1 - sim_params.max_epa_adj))],
                    ))

                    phev_calc['adjIterKwhPerMile'] = np.zeros(len(phev_calc['labIterKwhPerMile']))
                    for c, _ in enumerate(phev_calc['labIterKwhPerMile']):
                        if phev_calc['labIterKwhPerMile'][c] == 0:
                            phev_calc['adjIterKwhPerMile'][c] = 0
                        else:
                            phev_calc['adjIterKwhPerMile'][c] = (
                                1 / max(1 / (adj_params['City Intercept'] + (adj_params['City Slope'] / (
                                    (1 / phev_calc['labIterKwhPerMile'][c]) * props.kwh_per_gge))), 
                                (1 - sim_params.max_epa_adj) * ((1 / phev_calc['labIterKwhPerMile'][c]) * props.kwh_per_gge)
                                )) * props.kwh_per_gge

                else:
                    phev_calc['adjIterMpgge'] = np.concatenate((np.zeros(int(np.floor(phev_calc['cdCycs']))), 
                        [max(1 / (adj_params['Highway Intercept'] + (adj_params['Highway Slope'] / (np.array(sd[key].dist_mi).sum() / (phev_calc['transFsKwh'] / props.kwh_per_gge)))), 
                            np.array(sd[key].dist_mi).sum() / (phev_calc['transFsKwh'] / props.kwh_per_gge) * (1 - sim_params.max_epa_adj))],
                        [max(1 / (adj_params['Highway Intercept'] + (adj_params['Highway Slope'] / (np.array(sd[key].dist_mi).sum() / (phev_calc['csFsKwh'] / props.kwh_per_gge)))),
                            np.array(sd[key].dist_mi).sum() / (phev_calc['csFsKwh'] / props.kwh_per_gge) * (1 - sim_params.max_epa_adj))],
                    ))

                    phev_calc['adjIterKwhPerMile']=np.zeros(
                        len(phev_calc['labIterKwhPerMile']))
                    for c, _ in enumerate(phev_calc['labIterKwhPerMile']):
                        if phev_calc['labIterKwhPerMile'][c] == 0:
                            phev_calc['adjIterKwhPerMile'][c]=0
                        else:
                            phev_calc['adjIterKwhPerMile'][c]=(
                                1 / max(1 / (adj_params['Highway Intercept'] + (adj_params['Highway Slope'] / (
                                    (1 / phev_calc['labIterKwhPerMile'][c]) * props.kwh_per_gge))),
                                    (1 - sim_params.max_epa_adj) * ((1 / phev_calc['labIterKwhPerMile'][c]) * props.kwh_per_gge)
                                )) * props.kwh_per_gge

                phev_calc['adjIterCdMiles'] = np.zeros(
                    int(np.ceil(phev_calc['cdCycs'])) + 2)
                for c, _ in enumerate(phev_calc['adjIterCdMiles']):
                    if c == 0:
                        phev_calc['adjIterCdMiles'][c] = 0
                    elif c <= np.floor(phev_calc['cdCycs']):
                        phev_calc['adjIterCdMiles'][c] = (
                            phev_calc['adjIterCdMiles'][c-1] + phev_calc['cd_ess_kWh__mi'] * 
                            np.array(sd[key].dist_mi).sum() / phev_calc['adjIterKwhPerMile'][c])
                    elif c == np.floor(phev_calc['cdCycs']) + 1:
                        phev_calc['adjIterCdMiles'][c] = (
                            phev_calc['adjIterCdMiles'][c-1] + phev_calc['trans_ess_kWh__mi'] *
                            np.array(sd[key].dist_mi).sum() / phev_calc['adjIterKwhPerMile'][c])
                    else:
                        phev_calc['adjIterCdMiles'][c] = 0

                if (veh.max_soc - phev_calcs['regenSocBuffer'] - np.array(sd[key].soc).min() < 0.01):
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

                phev_calc['adjUf'] = float(interpf(phev_calc['adjCdMiles']))

                phev_calc['adjMpgge'] = 1 / (phev_calc['adjUf'] / phev_calc['adjCdMpgge'] +
                    (1 - phev_calc['adjUf']) / phev_calc['adjCsMpgge'])
                
                phev_calc['adjKwhPerMile'] = phev_calc['adjIterUfKwhPerMile'].sum(
                    ) / phev_calc['adjIterUf'].max() / veh.chg_eff

                phev_calc['adjEssKwhPerMile'] = phev_calc['adjIterUfKwhPerMile'].sum(
                    ) / phev_calc['adjIterUf'].max() 

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
            out['adjUddsMpgge'] = phev_calcs['udds']['adjMpgge']
            out['adjHwyMpgge'] = phev_calcs['hwy']['adjMpgge']
            out['adjCombMpgge'] = 1 / (
                0.55 / phev_calcs['udds']['adjMpgge'] + 0.45 / phev_calcs['hwy']['adjMpgge'])

            out['adjCsCombMpgge'] = 1 / (
                0.55 / phev_calcs['udds']['adjCsMpgge'] + 0.45 / phev_calcs['hwy']['adjCsMpgge'])
            out['adjCdCombMpgge'] = 1 / (
                0.55 / phev_calcs['udds']['adjCdMpgge'] + 0.45 / phev_calcs['hwy']['adjCdMpgge'])

            out['adjUddsKwhPerMile'] = phev_calcs['udds']['adjKwhPerMile']
            out['adjHwyKwhPerMile'] = phev_calcs['hwy']['adjKwhPerMile']
            out['adjCombKwhPerMile'] = 0.55 * phev_calcs['udds']['adjKwhPerMile'] + \
                0.45 * phev_calcs['hwy']['adjKwhPerMile']

            out['adjUddsEssKwhPerMile'] = phev_calcs['udds']['adjEssKwhPerMile']
            out['adjHwyEssKwhPerMile'] = phev_calcs['hwy']['adjEssKwhPerMile']
            out['adjCombEssKwhPerMile'] = 0.55 * phev_calcs['udds']['adjEssKwhPerMile'] + \
                0.45 * phev_calcs['hwy']['adjEssKwhPerMile']

            # range for combined city/highway
            # utility factor (percent driving in charge depletion mode)
            out['UF'] = interpf(0.55 * phev_calcs['udds']['adjCdMiles'] +
                                0.45 * phev_calcs['hwy']['adjCdMiles'])

            out['netPhevCDMiles'] = 0.55 * phev_calcs['udds']['adjCdMiles'] + \
                                0.45 * phev_calcs['hwy']['adjCdMiles']
           
            out['netRangeMiles'] = (veh.fs_kwh / props.kwh_per_gge - 
                out['netPhevCDMiles'] / out['adjCdCombMpgge']
                ) * out['adjCsCombMpgge'] + out['netPhevCDMiles']
 
        get_label_fe_phev()
        
    # run accelerating sim_drive
    if use_rust:
        sd['accel'] = simdrive.RustSimDrive(cyc['accel'], veh)
    else:
        sd['accel'] = simdrive.SimDrive(cyc['accel'], veh)
    sd['accel'].sim_params = set_nested_values(sd['accel'].sim_params)
    simdrive.run_simdrive_for_accel_test(sd['accel'])
    if (np.array(sd['accel'].mph_ach) >= 60).any():
        out['netAccel'] = np.interp(
            x=60, xp=np.array(sd['accel'].mph_ach), fp=np.array(cyc['accel'].time_s))
    else:
        # in case vehicle never exceeds 60 mph, penalize it a lot with a high number
        print(veh.scenario_name + ' never achieves 60 mph.')
        out['netAccel'] = 1e3

    # success Boolean -- did all of the tests work(e.g. met trace within ~2 mph)?
    out['resFound'] = "model needs to be implemented for this" # this may need fancier logic than just always being true

    if full_detail and verbose:
        for key in out.keys():
            try:
                print(key + f': {out[key]:.5g}')
            except:
                print(key + f': {out[key]}')
    
        return out, sd
    elif full_detail:
        return out, sd
    elif verbose:
        for key in out.keys():
            try:
                print(key + f': {out[key]:.5g}')
            except:
                print(key + f': {out[key]}')
        return out
    else:
        return out


if __name__ == '__main__':
    veh = vehicle.Vehicle.from_vehdb(5).to_rust() # load default vehicle

    out, sds = get_label_fe(veh, use_rust=True, full_detail=True)
    for key in out.keys():
        try:
            print(key + f': {out[key]:.5g}')
        except:
            print(key + f': {out[key]}')
    
