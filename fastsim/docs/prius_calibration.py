# %%
import numpy as np
import pandas as pd
import fastsim
import matplotlib
from IPython import get_ipython
from IPython.display import display, HTML
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
veh_filename = "2016_TOYOTA_Prius_TWO.csv"
veh = fastsim.vehicle.Vehicle(veh_filename)

# %%
# https://www.fueleconomy.gov/feg/noframes/37163.shtml
target_adj_udds = 54  # mpg
target_adj_hwfet = 50  # mpg

# https://www.epa.gov/sites/default/files/2016-07/16tstcar.csv
use = "EPA"  # "MFR", "EPA", or "both"
if use == "MFR":
    target_us06_city = np.mean((34.266592, 32.8181915))  # mpg
    target_us06_hwy = np.mean((50.1828331, 46.7089802))  # mpg
    target_hwfet = np.mean((66, 63.6))  # mpg
elif use == "EPA":
    target_us06_city = np.mean((35.4375353, 34.0691171))  # mpg
    target_us06_hwy = np.mean((54.430553, 55.3448029))  # mpg
    target_hwfet = np.mean((71.2, 76.1))  # mpg
elif use == "both":
    target_us06_city = np.mean((34.266592, 35.4375353, 32.8181915, 34.0691171))  # mpg
    target_us06_hwy = np.mean((50.1828331, 54.430553, 46.7089802, 55.3448029))  # mpg
    target_hwfet = np.mean((66, 71.2, 63.6, 76.1))  # mpg

# %%
label_results = fastsim.simdrivelabel.get_label_fe(veh)
result_adj_udds = label_results["adjUddsMpgge"]
result_adj_hwfet = label_results["adjHwyMpgge"]

# %%
# UDDS
error_adj_udds = (result_adj_udds - target_adj_udds)/target_adj_udds * 100
print("UDDS")
print(f"Target: {target_adj_udds} mpg")
print(f"Result: {result_adj_udds:.4f} mpg ({error_adj_udds:+.2f}%)")

# %%
# HWFET
error_adj_hwfet = (result_adj_hwfet - target_adj_hwfet)/target_adj_hwfet * 100
print("HWFET")
print(f"Target: {target_adj_hwfet} mpg")
print(f"Result: {result_adj_hwfet:.4f} mpg ({error_adj_hwfet:+.2f}%)")

# %%
# US06
cyc = fastsim.cycle.Cycle("us06")
sim_us06 = fastsim.simdrive.SimDriveClassic(cyc, veh)
sim_us06.sim_drive()

city1 = slice(None, 130)
hwy = slice(130, 495)
city2 = slice(495, None)

miles_city = sim_us06.distMiles[city1].sum() + sim_us06.distMiles[city2].sum()
gallons_city = (sim_us06.fsKwhOutAch[city1].sum() + sim_us06.fsKwhOutAch[city2].sum()) / sim_us06.props.kWhPerGGE
result_us06_city = miles_city/gallons_city
error_us06_city = (result_us06_city - target_us06_city)/target_us06_city * 100
print("US06 city")
print(f"Target: {target_us06_city:.4f} mpg")
print(f"Result: {result_us06_city:.4f} mpg ({error_us06_city:+.2f}%)")

miles_hwy = sim_us06.distMiles[hwy].sum()
gallons_hwy = sim_us06.fsKwhOutAch[hwy].sum() / sim_us06.props.kWhPerGGE
result_us06_hwy = miles_hwy/gallons_hwy
error_us06_hwy = (result_us06_hwy - target_us06_hwy)/target_us06_hwy * 100
print("US06 highway")
print(f"Target: {target_us06_hwy:.4f} mpg")
print(f"Result: {result_us06_hwy:.4f} mpg ({error_us06_hwy:+.2f}%)")

# %%

df = pd.DataFrame(
    {
        "UDDS": [target_adj_udds, result_adj_udds, error_adj_udds],
        "HWFET": [target_adj_hwfet, result_adj_hwfet, error_adj_hwfet],
        "US06 city": [target_us06_city, result_us06_city, error_us06_city],
        "US06 highway": [target_us06_hwy, result_us06_hwy, error_us06_hwy],
    },
    index=["Target", "Result", "% Error"]
).round(2)
display(HTML(df.to_html()))

# %%
