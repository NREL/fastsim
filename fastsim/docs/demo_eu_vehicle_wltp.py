import sys
import os
from pathlib import Path
import numpy as np
import time
import fastsim as fsim
from fastsim import simdrive
from fastsim.simdrivelabel import get_label_fe
import matplotlib.pyplot as plt


veh_2020_golf = fsim.vehicle.Vehicle.from_file(str(Path(fsim.simdrive.__file__).parent /'resources/vehdb/2020_EU_VW_Golf_2.0TDI.csv')).to_rust()

wltp_low_cyc_3 = fsim.cycle.Cycle.from_file(str(Path(fsim.simdrive.__file__).parent / 'resources/cycles/wltc_class3_low3.csv')).to_rust()
wltp_med_cyc_3b = fsim.cycle.Cycle.from_file(str(Path(fsim.simdrive.__file__).parent / 'resources/cycles/wltc_class3_med3b.csv')).to_rust()
wltp_high_cyc_3b = fsim.cycle.Cycle.from_file(str(Path(fsim.simdrive.__file__).parent / 'resources/cycles/wltc_class3_high3b.csv')).to_rust()
wltp_extrahigh_cyc_3 = fsim.cycle.Cycle.from_file(str(Path(fsim.simdrive.__file__).parent / 'resources/cycles/wltc_class3_extra_high3.csv')).to_rust()
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
