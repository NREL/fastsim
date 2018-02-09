import FASTSim

cyc = FASTSim.get_standard_cycle("UDDS")
veh = FASTSim.get_veh(10)
output = FASTSim.sim_drive_sub(cyc, veh,0.7854515)
