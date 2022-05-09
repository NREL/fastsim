import sys
import os
from pathlib import Path
import numpy as np
import time
import fastsim as fsim
from fastsim import simdrive
from fastsim.simdrivelabel import get_label_fe
import matplotlib.pyplot as plt
#v = fsim.vehicle.Vehicle.from_file('./resources/vehdb/2016_EU_VW_Golf_1.4TSI.csv')
#print(get_label_fe(v))

veh_2020_golf = fsim.vehicle.Vehicle.from_file('./resources/vehdb/2020_EU_VW_Golf_1.5TSI.csv').to_rust()
# veh_2020_golf = fsim.vehicle.Vehicle.from_file('./resources/vehdb/2020_EU_VW_Golf_2.0TDI.csv').to_rust()
LBF_PER_NEWTON = 0.2248
KM_PER_MILES = 1.609
a = 76.2 * LBF_PER_NEWTON # ad-hoc 0.2248 lbf per newton
b = 0.4960  * LBF_PER_NEWTON * KM_PER_MILES 
c = 0.02871 * LBF_PER_NEWTON * KM_PER_MILES * KM_PER_MILES# N / (kph)^2 lbf/mph^2


# drag_coef, wheel_rr_coef = fsim.utils.abc_to_drag_coeffs(veh = veh_2020_golf,
#                    a_lbf = a, 
#                   b_lbf__mph = b,
#                   c_lbf__mph2 = c, 
#                   show_plots = True,
#                   use_rust=True)

wltp_low_cyc_3 = fsim.cycle.Cycle.from_file(
    str(Path(fsim.simdrive.__file__).parent / 'resources/cycles/wltc_class3_low3.csv')).to_rust()
wltp_med_cyc_3b = fsim.cycle.Cycle.from_file('wltc_class3_med3b.csv').to_rust()
wltp_high_cyc_3b = fsim.cycle.Cycle.from_file('wltc_class3_high3b.csv').to_rust()
wltp_extrahigh_cyc_3 = fsim.cycle.Cycle.from_file('wltc_class3_extra_high3.csv').to_rust()
#t0 = time.time()
cyc_wltp_combo = fsim.cycle.concat([wltp_low_cyc_3.get_cyc_dict(), wltp_med_cyc_3b.get_cyc_dict(),
                               wltp_high_cyc_3b.get_cyc_dict(),wltp_extrahigh_cyc_3.get_cyc_dict()])
cyc_wltp_combo = fsim.cycle.Cycle.from_dict(cyc_wltp_combo).to_rust()


def simdrive_get_mpg(cur_simdrive):
    cur_simdrive.sim_drive()
    # cur_simdrive_post = fsim.simdrive.SimDrivePost(cur_simdrive)
    # simdrive_out = cur_simdrive_post.get_diagnostics()
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

plt.plot(cyc_wltp_combo.time_s,np.array(cyc_wltp_combo.mph)*KM_PER_MILES)
plt.xlabel("Time [s]")
plt.ylabel('Velocity [kph]')
plt.show()
plt.savefig('WLTP.jpg')