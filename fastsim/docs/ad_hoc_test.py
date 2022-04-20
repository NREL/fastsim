import fastsim as fsim
from fastsim.utilities import abc_to_drag_coeffs
v = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()

a = 25.91
b = 0.1943
c = 0.01796

drag_coef, wheel_rr_coef = abc_to_drag_coeffs(veh = v,
                   a_lbf = a, 
                   b_lbf__mph = b,
                   c_lbf__mph2 = c, 
                   show_plots = True,
                   use_rust=True)
print(drag_coef)
print(wheel_rr_coef)