from fastsim import set_param_from_path
import fastsim as fsim

# %% [markdown]

# `fastsim3` -- demo how to get and set extrapolation fields
# %%

# load 1D test vehicle from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
)
veh_no_save = veh.copy()
veh_no_save.save_interval = None

print("1D extrapolation fields:")

print('x: ', veh.fc.eff_interp.x)

print('f_x: ', veh.fc.eff_interp.f_x)

print('strategy: ', veh.fc.eff_interp.strategy)

print('extrapolate: ', veh.fc.eff_interp.extrapolate)

# # how to set extrapolation fields
# eff_interp_updated = veh.fc.eff_interp
# eff_interp_updated.reset_orphaned()
# eff_interp_updated.x = [0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.85, 1.0]
# veh.fc.eff_interp = eff_interp_updated

# print('updated x: ', veh.fc.eff_interp.x)

# # how to set extrapolated fields
# set_param_from_path(veh, veh.fc.eff_interp.x, [0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.85, 1.0])
# print('updated x: ', veh.fc.eff_interp.x)

# load 2D test vehicle from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion_2D_test.yaml")
)
veh_no_save = veh.copy()
veh_no_save.save_interval = None

print("2D extrapolation fields:")

print('x: ', veh.fc.eff_interp.x)

print('y: ', veh.fc.eff_interp.y)

print('f_xy: ', veh.fc.eff_interp.f_xy)

print('strategy: ', veh.fc.eff_interp.strategy)

print('extrapolate: ', veh.fc.eff_interp.extrapolate)

# load 3D test vehicle from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion_3D_test.yaml")
)
veh_no_save = veh.copy()
veh_no_save.save_interval = None

print("3D extrapolation fields:")

print('x: ', veh.fc.eff_interp.x)

print('y: ', veh.fc.eff_interp.y)

print('z: ', veh.fc.eff_interp.z)

print('f_xyz: ', veh.fc.eff_interp.f_xyz)

print('strategy: ', veh.fc.eff_interp.strategy)

print('extrapolate: ', veh.fc.eff_interp.extrapolate)

# load ND test vehicle from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion_ND_test.yaml")
)
veh_no_save = veh.copy()
veh_no_save.save_interval = None

print("ND extrapolation fields:")

print('grid: ', veh.fc.eff_interp.grid)

print('values: ', veh.fc.eff_interp.values)

print('strategy: ', veh.fc.eff_interp.strategy)

print('extrapolate: ', veh.fc.eff_interp.extrapolate)

print(type(veh.fc.eff_interp.values))