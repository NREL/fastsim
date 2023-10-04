files_to_run = [
    'accel_demo.py',
    'cav_demo.py',
    'demo_abc_drag_coef_conv.py',
    'demo_eu_vehicle_wltp.py',
    'demo.py',
    'fusion_thermal_demo.py',
    'mp_parallel_demo.py',
    'stop_start_demo.py',
    'time_dilation_demo.py',
    'timing_demo.py',
    'test_demo.py',
    'test_cav_demo.py',
]

for file in files_to_run:
    with open(file) as f:
        exec(f.read())