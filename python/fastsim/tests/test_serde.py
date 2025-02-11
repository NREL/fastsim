import time
import fastsim as fsim

def get_solved_sd():
    # load 2012 Ford Fusion from file
    veh = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")

    # Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
    veh.set_save_interval(1)
    
    # load cycle from file
    cyc = fsim.Cycle.from_resource("udds.csv")

    # instantiate SimDrive 
    sd = fsim.SimDrive(veh, cyc)

    # run simulation
    sd.walk()

    return sd

def test_pydict():
    sd = get_solved_sd()

    t0 = time.perf_counter_ns()
    sd_dict_msg = sd.to_pydict(flatten=False, data_fmt="msg_pack")            
    sd_msg = fsim.SimDrive.from_pydict(sd_dict_msg, data_fmt="msg_pack", skip_init=True)
    t1 = time.perf_counter_ns()
    t_msg = t1 - t0
    print(f"\nElapsed time for MessagePack: {t_msg:.3e} ns ")

    t0 = time.perf_counter_ns()
    sd_dict_yaml = sd.to_pydict(flatten=False, data_fmt="yaml")            
    sd_yaml = fsim.SimDrive.from_pydict(sd_dict_yaml, data_fmt="yaml", skip_init=True)
    t1 = time.perf_counter_ns()
    t_yaml = t1 - t0
    print(f"Elapsed time for YAML: {t_yaml:.3e} ns ")
    print(f"YAML time per MessagePack time: {(t_yaml / t_msg):.3e} ")

    t0 = time.perf_counter_ns()
    sd_dict_json = sd.to_pydict(flatten=False, data_fmt="json")            
    _sd_json = fsim.SimDrive.from_pydict(sd_dict_json, data_fmt="json", skip_init=True)
    t1 = time.perf_counter_ns()
    t_json = t1 - t0
    print(f"Elapsed time for json: {t_json:.3e} ns ")
    print(f"JSON time per MessagePack time: {(t_json / t_msg):.3e} ")

    # `to_yaml` is probably needed because of NAN variables
    assert sd_msg == sd
    assert sd_yaml == sd
    # TODO: uncomment and investigate
    # assert sd_json == sd  

def test_dataframe():
    get_solved_sd().to_dataframe()
