import fastsim as fsim

# %% [markdown]

# `fastsim3` -- demo how to get and set extrapolation fields
# %%

# load 1D test vehicle from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
)
veh_no_save = veh.copy()
veh_no_save.__save_interval = None

print("1D extrapolation fields:")
print('x: ', veh.fc.eff_interp.x)
print('f_x: ', veh.fc.eff_interp.f_x)
print('strategy: ', veh.fc.eff_interp.strategy)
print('extrapolate: ', veh.fc.eff_interp.extrapolate)
# load 2D test vehicle from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion_2D_test.yaml")
)
veh_no_save = veh.copy()
veh_no_save.__save_interval = None

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
veh_no_save.__save_interval = None

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
veh_no_save.__save_interval = None

print("ND extrapolation fields:")
print('grid: ', veh.fc.eff_interp.grid)
print('values: ', veh.fc.eff_interp.values)
print('strategy: ', veh.fc.eff_interp.strategy)
print('extrapolate: ', veh.fc.eff_interp.extrapolate)
print(type(veh.fc.eff_interp.values))
