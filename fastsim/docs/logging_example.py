# %%
import fastsim as fsim

# %%
cyc_py = fsim.cycle.Cycle.from_file("udds")
veh_py = fsim.vehicle.Vehicle.from_vehdb(1)
veh_py.veh_override_kg = 1e6
veh_py.set_derived()
sd_py = fsim.simdrive.SimDrive(cyc_py, veh_py)

cyc_rust = cyc_py.to_rust()
veh_rust = veh_py.to_rust()
sd_rust = fsim.simdrive.RustSimDrive(cyc_rust, veh_rust)
# sd_rust = sd_py.to_rust()

# %% Python with logging
sd_py.sim_drive()
sd_py.mpgge
print(sd_py.mpgge)

# %% Python with logging disabled
fsim.utils.disable_logging()
sd_py.sim_drive()
print(sd_py.mpgge)
fsim.utils.enable_logging()

# %% Rust with logging
sd_rust.sim_drive()
print(sd_rust.mpgge)

# %% Rust with logging "disabled"
fsim.utils.disable_logging()
sd_rust.sim_drive()
print("There shouldn't be logs here ^^^")
print(sd_rust.mpgge)
fsim.utils.enable_logging()

# We don't want this cell to log anything
# It could be that the way Rust logging is disabled in utilities.py line 115 isn't quite right
# Also, I get slightly different results between Python and Rust, which is weird:
# Python: 0.19727703248654188 MPGGE
# Rust:   0.20041036885249847 MPGGE

# https://docs.rs/pyo3-log/latest/pyo3_log/#performance-filtering-and-caching
# https://pyo3.rs/v0.12.3/logging.html
# https://github.com/vorner/pyo3-log/issues/21
