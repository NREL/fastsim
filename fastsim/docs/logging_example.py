# %%
import fastsim as fsim

# %%
cyc_py = fsim.cycle.Cycle.from_file("udds")
veh_py = fsim.vehicle.Vehicle.from_vehdb(1)
veh_py.veh_override_kg = 1e6  # Use very heavy vehicle to throw some warnings
veh_py.set_derived()
sd_py = fsim.simdrive.SimDrive(cyc_py, veh_py)

cyc_rust = cyc_py.to_rust()
veh_rust = veh_py.to_rust()
sd_rust = fsim.simdrive.RustSimDrive(cyc_rust, veh_rust)

# %% Python with logging
sd_py.sim_drive()
print("There SHOULD be logs here ^^^")
print(sd_py.mpgge)

# %% Python with logging disabled
with fsim.utils.suppress_logging():
    sd_py.sim_drive()
    print("There SHOULDN'T be logs here ^^^")
    print(sd_py.mpgge)
fsim.utils.enable_logging()

# %% Rust with logging
sd_rust.sim_drive()
print("There SHOULD be logs here ^^^")
print(sd_rust.mpgge)

# %% Rust with logging disabled
with fsim.utils.suppress_logging():
    sd_rust.sim_drive()
    print("There SHOULDN'T be logs here ^^^")
    print(sd_rust.mpgge)

# I get slightly different results between Python and Rust, which is weird:
# Python: 0.19727703248654188 MPGGE
# Rust:   0.20041036885249847 MPGGE

# https://docs.rs/pyo3-log/latest/pyo3_log/#performance-filtering-and-caching
# https://pyo3.rs/v0.12.3/logging.html
# https://github.com/vorner/pyo3-log/issues/21
