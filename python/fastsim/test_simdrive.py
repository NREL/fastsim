import fastsim as fsim
import matplotlib.pyplot as plt
from fastsim.simdrive import SimulationDrive


# Load cycle and initialize vehicle
cyc = fsim.cycle.Cycle.from_file("udds")
veh = fsim.vehicle.Vehicle.from_vehdb(11)  


# Run simulation
sim_drive = fsim.simdrive.SimDrive(cyc, veh)
sim_drive.sim_drive()


# Call the plot function directly with specified parameters
fig, axes = SimulationDrive(sim_drive).plot(
    signal=['Fuel Cell Output Power Achieved', 'Fuel Cell Input Power Achieved',],  # Example signals
    fuzzy_search=True,
    feeling_lucky=True,
    speed_trace=True,
    difference='none',
    type='temporal'
)



# Display the plot
plt.show()
