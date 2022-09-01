import sys
import os
from pathlib import Path
import numpy as np
import time
import fastsim as fsim
import fastsim.parameters as params
import fastsim.utilities as utils
import matplotlib.pyplot as plt


def convention_eu_veh_wltp_fe_test():
    veh_2020_golf = fsim.vehicle.Vehicle.from_file('2020_EU_VW_Golf_2.0TDI.csv').to_rust()

    wltp_low_cyc_3 = fsim.cycle.Cycle.from_file('wltc_class3_low3.csv').to_rust()
    wltp_med_cyc_3b = fsim.cycle.Cycle.from_file('wltc_class3_med3b.csv').to_rust()
    wltp_high_cyc_3b = fsim.cycle.Cycle.from_file('wltc_class3_high3b.csv').to_rust()
    wltp_extrahigh_cyc_3 = fsim.cycle.Cycle.from_file('wltc_class3_extra_high3.csv').to_rust()
    cyc_wltp_combo = fsim.cycle.concat([wltp_low_cyc_3.get_cyc_dict(), 
                                        wltp_med_cyc_3b.get_cyc_dict(),
                                        wltp_high_cyc_3b.get_cyc_dict(),
                                        wltp_extrahigh_cyc_3.get_cyc_dict()]
                                      )
    cyc_wltp_combo = fsim.cycle.Cycle.from_dict(cyc_wltp_combo).to_rust()


    def simdrive_get_mpg(cur_simdrive):
        cur_simdrive.sim_drive()
        return cur_simdrive.mpgge

    simdrive_lst = [
                    fsim.simdrive.RustSimDrive(wltp_low_cyc_3, veh_2020_golf),
                    fsim.simdrive.RustSimDrive(wltp_med_cyc_3b, veh_2020_golf),
                    fsim.simdrive.RustSimDrive(wltp_high_cyc_3b, veh_2020_golf),
                    fsim.simdrive.RustSimDrive(wltp_extrahigh_cyc_3, veh_2020_golf),
                    fsim.simdrive.RustSimDrive(cyc_wltp_combo, veh_2020_golf),
                    ] 

    mpg_list = list(map(simdrive_get_mpg,simdrive_lst))
    print(mpg_list)

def hybrid_eu_veh_wltp_fe_test():
    WILLANS_FACTOR_gram_CO2__MJ = 724  # gCO2/MJ
    E10_HEAT_VALUE_kWh__liter = 8.64  # kWh/L
    veh_2022_yaris = fsim.vehicle.Vehicle.from_file('2022_TOYOTA_Yaris_Hybrid_Mid.csv')
    veh_2022_yaris_rust = fsim.vehicle.Vehicle.from_file('2022_TOYOTA_Yaris_Hybrid_Mid.csv', to_rust=True).to_rust()
    
    # Measured FE (L/100km)
    meas_fe_combined_liter__100km = 3.8
    meas_fe_low_liter__100km = 2.9
    meas_fe_med_liter__100km = 3
    meas_fe_high_liter__100km = 3.5
    meas_fe_extrahigh_liter__100km = 5
    
    wltp_low_cyc_3 = fsim.cycle.Cycle.from_file('wltc_class3_low3.csv')
    wltp_low_cyc_3_rust = wltp_low_cyc_3.to_rust()
    wltp_med_cyc_3b = fsim.cycle.Cycle.from_file('wltc_class3_med3b.csv')
    wltp_med_cyc_3b_rust = wltp_med_cyc_3b.to_rust()
    wltp_high_cyc_3b = fsim.cycle.Cycle.from_file('wltc_class3_high3b.csv')
    wltp_high_cyc_3b_rust = wltp_high_cyc_3b.to_rust()
    wltp_extrahigh_cyc_3 = fsim.cycle.Cycle.from_file('wltc_class3_extra_high3.csv')
    wltp_extrahigh_cyc_3_rust = wltp_extrahigh_cyc_3.to_rust()
    cyc_wltp_combined = fsim.cycle.concat([wltp_low_cyc_3.get_cyc_dict(), 
                                        wltp_med_cyc_3b.get_cyc_dict(),
                                        wltp_high_cyc_3b.get_cyc_dict(),
                                        wltp_extrahigh_cyc_3.get_cyc_dict()
                                        ]
                                       )
    cyc_wltp_combined = fsim.cycle.Cycle.from_dict(cyc_wltp_combined)
    cyc_wltp_combined_rust = cyc_wltp_combined.to_rust()

    sim = fsim.simdrive.SimDrive(cyc_wltp_combined, veh_2022_yaris)
    sim_rust = fsim.simdrive.RustSimDrive(cyc_wltp_combined_rust,veh_2022_yaris_rust)
    sim.sim_drive()
    sim_rust.sim_drive()
    
    dist_miles_combined = sim.dist_mi.sum()
    dist_miles_combined_rust = sum(sim_rust.dist_mi)
    print(f"Distance modelled in miles: Pure Python FastSim:\t{dist_miles_combined:.2f}\t Rust backend FastSim:\t{dist_miles_combined_rust:.2f}")
    energy_combined = sim.fs_kwh_out_ach.sum()
    energy_combined_rust = sum(sim_rust.fs_kwh_out_ach)
    print(f"Fuel Supply achieved in kilowatts-hours: Pure Python FastSim:\t{energy_combined:.2f}\t Rust backend FastSim:\t{energy_combined_rust:.2f}")
    fe_mpgge_combined = sim.mpgge
    fe_mpgge_combined_rust = sim_rust.mpgge
    fe_l__100km_combined = utils.mpg_to_l__100km(fe_mpgge_combined)
    fe_l__100km_combined_rust = utils.mpg_to_l__100km(fe_mpgge_combined_rust)
    print(f"Fuel Consumption achieved in L/100km: Pure Python FastSim:\t{fe_l__100km_combined:.2f}\t Rust backend FastSim:\t{fe_l__100km_combined_rust:.2f}")

    i0 = len(wltp_low_cyc_3.time_s)
    i1 = i0 + len(wltp_med_cyc_3b.time_s)-1
    i2 = i1 + len(wltp_high_cyc_3b.time_s)-1

    low = slice(None, i0)
    medium = slice(i0-1, i1)
    high = slice(i1-1, i2)
    extrahigh = slice(i2-1, None)

    wltp_3b_phase_slice_list = [low, medium, high, extrahigh]

    def hybrid_veh_fe_soc_correction(cur_veh, raw_simdrive, phase_slice_list):
        '''
        phase_slice: [low phase slice, medium phase slice, high phase slice, extra high clice]
        '''
        fe_liter__100km_list = []
        for cur_phase_slice in phase_slice_list:
            cur_dist_miles = sum(np.array(raw_simdrive.dist_mi)[cur_phase_slice])
            cur_dist_km = cur_dist_miles * params.M_PER_MI / 1000 
            cur_energy_consumption_kwh = sum(np.array(sim.fs_kwh_out_ach)[cur_phase_slice])
            cur_fe_mpgge = cur_dist_miles / (cur_energy_consumption_kwh/sim.props.kwh_per_gge)
            cur_fe_liter__100km = utils.mpg_to_l__100km(cur_fe_mpgge)
            cur_dSOC = sim.soc[cur_phase_slice][-1] - sim.soc[cur_phase_slice][0]
            cur_dE_wh = -cur_dSOC * cur_veh.ess_max_kwh * 1000
            cur_dM_CO2_gram__100km  = 0.0036 * cur_dE_wh * 1/cur_veh.alt_eff * WILLANS_FACTOR_gram_CO2__MJ * 1/cur_dist_km
            cur_dfe_liter__100km = cur_dE_wh/1000 * 1/cur_veh.alt_eff * 1/E10_HEAT_VALUE_kWh__liter * 100/cur_dist_km
            cur_fe_adj_liter__100km = cur_fe_liter__100km + cur_dfe_liter__100km
            fe_liter__100km_list.append(cur_fe_adj_liter__100km)
        return fe_liter__100km_list
    
    fe_low3_l__100km, fe_med3b_l__100km, fe_high3b_l__100km, fe_extrahigh3_l__100km = hybrid_veh_fe_soc_correction(veh_2022_yaris, sim, wltp_3b_phase_slice_list)
    fe_low3_l__100km_rust, fe_med3b_l__100km_rust, fe_high3b_l__100km_rust, fe_extrahigh3_l__100km_rust = hybrid_veh_fe_soc_correction(veh_2022_yaris_rust, sim_rust, wltp_3b_phase_slice_list)
    
    print("LOW")
    print(f"  Target: {meas_fe_low_liter__100km} L/100km")
    print(f"  Simulation:  {fe_low3_l__100km:.2f} L/100km ({(fe_low3_l__100km - meas_fe_low_liter__100km)/meas_fe_low_liter__100km * 100:+.2f}%)")
    print(f"  Rust Simulation:  {fe_low3_l__100km_rust:.2f} L/100km ({(fe_low3_l__100km_rust - meas_fe_low_liter__100km)/meas_fe_low_liter__100km * 100:+.2f}%)")

    print("MEDIUM")
    print(f"  Target: {meas_fe_med_liter__100km} L/100km")
    print(f"  Simulation:  {fe_med3b_l__100km:.2f} L/100km ({(fe_med3b_l__100km - meas_fe_med_liter__100km)/meas_fe_med_liter__100km * 100:+.2f}%)")
    print(f"  Rust Simulation:  {fe_med3b_l__100km_rust:.2f} L/100km ({(fe_med3b_l__100km_rust - meas_fe_med_liter__100km)/meas_fe_med_liter__100km * 100:+.2f}%)")

    print("HIGH")
    print(f"  Target: {meas_fe_high_liter__100km} L/100km")
    print(f"  Simulation:  {fe_high3b_l__100km:.2f} L/100km ({(fe_high3b_l__100km - meas_fe_high_liter__100km)/meas_fe_high_liter__100km * 100:+.2f}%)")
    print(f"  Rust Simulation:  {fe_high3b_l__100km_rust:.2f} L/100km ({(fe_high3b_l__100km_rust - meas_fe_high_liter__100km)/meas_fe_high_liter__100km * 100:+.2f}%)")


    print("EXTRA-HIGH")
    print(f"  Target: {meas_fe_extrahigh_liter__100km} L/100km")
    print(f"  Simulation:  {fe_extrahigh3_l__100km:.6f} L/100km ({(fe_extrahigh3_l__100km - meas_fe_extrahigh_liter__100km)/meas_fe_extrahigh_liter__100km * 100:+.6f}%)")
    print(f"  Rust Simulation:  {fe_extrahigh3_l__100km_rust:.6f} L/100km ({(fe_extrahigh3_l__100km_rust - meas_fe_extrahigh_liter__100km)/meas_fe_extrahigh_liter__100km * 100:+.6f}%)")

    print("COMBINED")
    print(f"  Target: {meas_fe_combined_liter__100km} L/100km")
    print(f"  Simulation: {fe_l__100km_combined:.6f} L/100km ({(fe_l__100km_combined - meas_fe_combined_liter__100km)/meas_fe_combined_liter__100km * 100:+.6f}%)")
    print(f"  Rust Simulation: {fe_l__100km_combined_rust:.2f} L/100km ({(fe_l__100km_combined_rust - meas_fe_combined_liter__100km)/meas_fe_combined_liter__100km * 100:+.2f}%)")

if __name__ == '__main__':
    hybrid_eu_veh_wltp_fe_test()