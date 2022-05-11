import sys
import os
from pathlib import Path
import numpy as np
import time
import fastsim as fsim
from fastsim import simdrive
from fastsim.simdrivelabel import get_label_fe
import matplotlib.pyplot as plt

def TestMain():
    #veh_2020_golf = fsim.vehicle.Vehicle(veh_file='./resources/vehdb/2020_EU_VW_Golf_1.5TSI.csv')
    veh_2020_golf = fsim.vehicle.Vehicle(veh_file='./resources/vehdb/2020_EU_VW_Golf_2.0TDI.csv')
    LBF_PER_NEWTON = 0.2248
    KM_PER_MILES = 1.609
    a = 76.2 * LBF_PER_NEWTON # ad-hoc 0.2248 lbf per newton
    b = 0.4960  * LBF_PER_NEWTON * KM_PER_MILES 
    c = 0.02871 * LBF_PER_NEWTON * KM_PER_MILES * KM_PER_MILES# N / (kph)^2 lbf/mph^2

    # drag_coef, wheel_rr_coef = fsim.utils.abc_to_drag_coeffs(veh = veh_2020_golf,
    #                   a_lbf = a, 
    #                   b_lbf__mph = b,
    #                   c_lbf__mph2 = c, 
    #                   show_plots = True,
    #                   use_rust=True)

    wltp_low_cyc_3 = fsim.cycle.Cycle(cyc_file_path=str(Path(fsim.simdrive.__file__).parent / 'resources/cycles/wltc_class3_low3.csv'))
    wltp_med_cyc_3b = fsim.cycle.Cycle(cyc_file_path=str(Path(fsim.simdrive.__file__).parent /'resources/cycles/wltc_class3_med3b.csv'))
    wltp_high_cyc_3b = fsim.cycle.Cycle(cyc_file_path=str(Path(fsim.simdrive.__file__).parent /'resources/cycles/wltc_class3_high3b.csv'))
    wltp_extrahigh_cyc_3 = fsim.cycle.Cycle(cyc_file_path=str(Path(fsim.simdrive.__file__).parent /'resources/cycles/wltc_class3_extra_high3.csv'))
    #t0 = time.time()
    cyc_wltp_combo = fsim.cycle.concat([wltp_low_cyc_3.get_cyc_dict(), wltp_med_cyc_3b.get_cyc_dict(),
                                wltp_high_cyc_3b.get_cyc_dict(),wltp_extrahigh_cyc_3.get_cyc_dict()])
    cyc_wltp_combo = fsim.cycle.Cycle(cyc_dict=cyc_wltp_combo)


    def simdrive_get_mpg(cur_simdrive):
        cur_simdrive.sim_drive()
        return cur_simdrive.mpgge

    simdrive_lst = [
                    fsim.simdrive.SimDriveClassic(wltp_low_cyc_3, veh_2020_golf),
                    fsim.simdrive.SimDriveClassic(wltp_med_cyc_3b, veh_2020_golf),
                    fsim.simdrive.SimDriveClassic(wltp_high_cyc_3b, veh_2020_golf),
                    fsim.simdrive.SimDriveClassic(wltp_extrahigh_cyc_3, veh_2020_golf),
                    fsim.simdrive.SimDriveClassic(cyc_wltp_combo, veh_2020_golf),
                    ] 

    mpg_list = list(map(simdrive_get_mpg,simdrive_lst))

    print(mpg_list)

    plt.plot(cyc_wltp_combo.time_s,cyc_wltp_combo.get_cycMph()*KM_PER_MILES)
    plt.xlabel("Time [s]")
    plt.ylabel('Velocity [kph]')
    plt.show()

if __name__ == '__main__':
    TestMain()