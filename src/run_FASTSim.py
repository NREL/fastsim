import FASTSim

cyc = FASTSim.get_standard_cycle("udds")
veh = FASTSim.get_veh(1)
output = FASTSim.sim_drive_sub(cyc, veh,0.7854515)
