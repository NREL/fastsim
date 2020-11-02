"""Module containing classes and methods for calculating label fuel economy.
For example usage, see ../README.md"""

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
        cyc['hwy'] = cycle.Cycle('hwy').get_numba_cyc()

        sd['udds'] = simdrive.SimDriveJit(cyc['udds'], veh)
        sd['hwy'] = simdrive.SimDriveJit(cyc['hwy'], veh)

    else:
        cyc['udds'] = cycle.Cycle('udds')
        cyc['hwy'] = cycle.Cycle('hwy')

        sd['udds'] = simdrive.SimDriveClassic(cyc['udds'], veh)
        sd['hwy'] = simdrive.SimDriveClassic(cyc['hwy'], veh)

    if params.PT_TYPES[veh.vehPtType] != 'PHEV':
        # run simdrive for non-phev powertrains    
        sd['udds'].sim_drive()
        sd['hwy'].sim_drive()
        # efficiency-related calculations
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
        cdBattKwh = self.essDischgKj / 3600.0
        # charge depletion fuel gallons
        cdFsGal = self.fsKwhOutAch.sum() / params.kWhPerGGE

        # # from excel where prevMiles `C298:`
        # udds_mpg_thing = max(
        #     1 / (cityIntercept + (citySlope / (self.distMiles.sum() / (phevUddsGasKwh / params.kWhPerGGE)))),
        #     self.distMiles.sum() / (phevUddsGasKwh / params.kWhPerGGE) * (1 - params.maxEpaAdj)
        #     )

        # udds_kwHr__mi_thing = =IF(ISNUMBER(+@phevUddsElecKwh),IF(+@phevUddsElecKwh=0,0,(1/MAX(1/(cityIntercept+(citySlope/((1/+@phevUddsLabKwhPerMile)*'Veh Model'!kWhPerGGE))),(1-'Veh Model'!maxEpaAdj)*((+@phevUddsMiles-C301)/(+@phevUddsElecKwh*(1/'Veh Model'!kWhPerGGE)))))*'Veh Model'!kWhPerGGE),"")

        # SOC change during 1 cycle
        deltaSoc = (self.veh.maxSoc - self.veh.minSoc)
        # total number of miles in charge depletion mode, assuming constant kWh_per_mi
        totalCdMiles = deltaSoc * \
            self.veh.maxEssKwh / self.battery_kWh_per_mi
        # float64 number of cycles in charge depletion mode, up to transition
        cdCycs = totalCdMiles / self.distMiles.sum()
        # fraction of transition cycle spent in charge depletion
        cdFracInTrans = cdCycs % np.floor(cdCycs)
        totalMiles = self.distMiles.sum() * (cdCycs + (1 - cdFracInTrans))

        # first cycle that ends in charge sustaining behavior
        initSoc = self.veh.minSoc + 0.01  # the 0.01 is here to be consistent with Excel
        self.sim_drive_walk(initSoc, auxInKwOverride)
        self.set_post_scalars()
        # charge depletion battery kW-hr
        csBattKwh = self.essDischgKj / 3600.0
        # charge depletion fuel gallons
        csFsGal = self.fsKwhOutAch.sum() / params.kWhPerGGE

        # note that all values set by `self.set_post_scalars` are relevant only to
        # the final charge sustaining cycle

        # harmonic average of charged depletion, transition, and charge sustaining phases
        self.mpgge = totalMiles / (cdFsGal * cdCycs + csFsGal)
        self.battery_kWh_per_mi = (
            cdBattKwh * cdCycs + csBattKwh) / totalMiles

        # efficiency-related calculations
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

    if full_detail:
        return out, sd
    else:
        return out