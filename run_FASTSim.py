import FASTSim

cyc = FASTSim.get_standard_cycle("UDDS")
veh = FASTSim.get_veh(1)
output = FASTSim.sim_drive(cyc, veh)
