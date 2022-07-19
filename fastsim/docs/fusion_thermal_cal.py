# %%
import fastsim as fsim

# Prepare SimDrives
trips = ["trip_udds", "trip_us06"]
cycles = [
    fsim.cycle.Cycle.from_file("udds").to_rust(),
    fsim.cycle.Cycle.from_file("us06")
]
assert len(cycles) == len(trips)
vehicles = [fsim.vehicle.Vehicle.from_file("2012_Ford_Fusion.csv").to_rust()] * len(trips)
sim_drives = {trip: fsim.SimDriveHot(cyc, veh) for trip, cyc, veh in zip(trips, cycles, vehicles)}

objectives = fsim.calibration.ModelErrors(
    sim_drives=sim_drives,
    objectives=[
        ("net_kj",),
    ],
    params=[
        ("veh", "drag_coef"),
    ],
    verbose=False
)

problem = fsim.calibration.CalibrationProblem(
    err=objectives,
    param_bounds=[
        (1.5, 3),
    ],
)

# %%
problem.minimize()

print(problem.res.X)