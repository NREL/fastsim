import fastsim as fsim
from fastsim.auxiliaries import abc_to_drag_coeffs, drag_coeffs_to_abc
v = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()
v2 = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()

a = 25.91
b = 0.1943
c = 0.01796

drag_coef, wheel_rr_coef = abc_to_drag_coeffs(veh = v,
                        a_lbf=a, 
                        b_lbf__mph=b, 
                        c_lbf__mph2=c,
                        custom_rho=False,
                        simdrive_optimize=True,
                        show_plots=True,
                        use_rust=True)
print(drag_coef)
print(wheel_rr_coef)

a_test, b_test, c_test = drag_coeffs_to_abc(veh=v,
                       fit_with_curve=False,
                       show_plots=True)
print(a_test,b_test,c_test)