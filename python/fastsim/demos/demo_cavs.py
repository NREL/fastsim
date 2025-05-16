"""
Demonstration of Connected Automated Vehicle (CAV) Functionality in FASTSim

This module demonstrates:
- cycle manipulation utilities
- eco-approach: utilizing vehicle coasting to conserve fuel use
- eco-cruise: use of trajectories to remove unnecessary accelerations
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import fastsim as fsim
import fastsim.demos.plot_utils as pu

sns.set_theme()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"
LIST_COLUMN_OPTIONS = False


def microtrip_demo():
    """Run a demonstration of cycle manipulation utilities"""
    cycle_name = "udds"
    cycle = fsim.Cycle.from_resource(f"{cycle_name}.csv")

    microtrips = cycle.to_microtrips(None)

    if SHOW_PLOTS:
        max_microtrips = 4
        fig, ax = plt.subplots()
        num = min(max_microtrips, len(microtrips))
        for idx, mt in enumerate(microtrips):
            mtd = mt.to_pydict()
            color = pu.BASE_COLORS[idx % len(pu.BASE_COLORS)]
            line = pu.BASE_LINE_STYLES[idx % len(pu.BASE_LINE_STYLES)]
            ax.plot(
                mtd["time_seconds"],
                mtd["speed_meters_per_second"],
                marker=".",
                color=color,
                linestyle=line,
                label=f"#{idx + 1}",
            )
            if idx >= max_microtrips:
                break
        ax.set_title(f"First {num + 1} Microtrips of {cycle_name.upper()}")
        ax.set_ylabel("Speed (m/s)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        fig.tight_layout()
        plt.show(block=True)


def plot_speed_by_time(df, c0, is_coast=None, save_interval=1, title=None):
    """Plot speed by time"""
    fig, ax = plt.subplots()
    ax.plot(
        np.array(c0["time_seconds"]),
        np.array(c0["speed_meters_per_second"]),
        "k-", label="original")
    ax.plot(
        np.array(df["cyc.time_seconds"])[:: save_interval],
        np.array(df["veh.history.speed_ach_meters_per_second"]),
        "b:", label="coast")
    if is_coast is not None:
        ax.plot(
            np.array(c0["time_seconds"]),
            np.array(is_coast),
            "r:", label="coast-mode")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.legend()
    fig.tight_layout()
    plt.show(block=True)


def plot_speed_by_dist(df, c0, is_coast=None, save_interval=1, title=None):
    """Plot speed by distance"""
    fig, ax = plt.subplots()
    ax.plot(
        np.array(c0["dist_meters"]),
        np.array(c0["speed_meters_per_second"]),
        "k-", label="original")
    ax.plot(
        np.array(df["cyc.dist_meters"])[:: save_interval],
        np.array(df["veh.history.speed_ach_meters_per_second"]),
        "b:", label="coast")
    if is_coast is not None:
        ax.plot(
            np.array(c0["dist_meters"]),
            np.array(is_coast),
            "r:", label="coast-mode")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.legend()
    fig.tight_layout()
    plt.show(block=True)


def setup_models(cyc_file="udds.csv", veh_file="2012_Ford_Fusion.yaml"):
    """Set up and return cycle and vehicle models"""
    veh = fsim.Vehicle.from_resource(veh_file)
    veh.set_save_interval(1)
    cyc = fsim.Cycle.from_resource(cyc_file)
    # Note the amount of idle time at the end of the reference cycle
    end_idle_duration_s = cyc.ending_idle_time_s()
    # Make a copy of the original cycle.
    cyc0 = cyc.copy()
    # Add 100 seconds and 10% to the cycle time to allow for delay
    # caused by coasting.
    cyc = cyc.extend_time(absolute_time_s=240.0, time_fraction=0.3)
    return {
        "cyc": cyc,
        "cyc0": cyc0,
        "veh": veh,
        "end_idle_duration_s": end_idle_duration_s,
    }


def basic_coasting_demo():
    """Demonstrate coasting starting from a given speed"""
    # veh = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")
    coast_speed_mps = 20.0
    veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    veh.set_save_interval(1)
    cyc = fsim.Cycle.from_resource("udds.csv")
    # We can query to see how much idle time exists at the end of
    # the reference cycle so we can (eventually) duplicate that on
    # the modified cycle.
    end_idle_duration_s = cyc.ending_idle_time_s()
    # Add time to allow for delay caused by coasting. Note: here we are
    # extending by absolute time but a time fraction (e.g., 0.1 to
    # extend it by 10%) is also possible. If both are specified, both
    # will be used: e.g., extending by 10% AND add 100 seconds in addition.
    cyc = cyc.extend_time(absolute_time_s=120.0, time_fraction=None)
    # Make a copy of the extended cycle.
    cyc0 = cyc.copy()
    man = fsim.Maneuver.create_from(cyc, veh.copy())
    # Set coasting variables
    d = man.to_pydict()
    # All coasting maneuvers require coast_allow to be set to True
    d["coast_allow"] = True
    # coast_start_speed is mainly used for testing. This
    # causes the vehicle to coast to a stop whenever the vehicle
    # passes the coast_start_speed_meters_per_second. This is
    # a "hello world" of sorts for eco-coast.
    d["coast_start_speed_meters_per_second"] = coast_speed_mps
    # Reset the Maneuver object using the python dictionary
    man = fsim.Maneuver.from_pydict(d)
    # Modify the cycle and return it
    cyc = man.apply_maneuvers()
    # Now we can trim the maneuver cycle to only have as much
    # idle time at end as the original reference cycle
    cyc = cyc.trim_ending_idle(idle_to_keep_s=end_idle_duration_s)
    # Run simdrive using the modified cycle
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    if SHOW_PLOTS:
        c0 = cyc0.to_pydict()
        df = sd.to_dataframe()
        if LIST_COLUMN_OPTIONS:
            print("Available Columns:")
            for column_name in df.columns:
                print(f"- {column_name}")
        plot_speed_by_time(df, c0, title=f"Coasting behavior from {coast_speed_mps} m/s")
        plot_speed_by_dist(df, c0,
                           title=f"Coasting Behavior from {coast_speed_mps} m/s (distance-based)")


def advanced_coasting_demo():
    """Demonstrate coasting starting from a given speed"""
    # veh = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")
    veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    veh.set_save_interval(1)
    cyc = fsim.Cycle.from_resource("udds.csv")
    # We can query to see how much idle time exists at the end of
    # the reference cycle so we can (eventually) duplicate that on
    # the modified cycle.
    end_idle_duration_s = cyc.ending_idle_time_s()
    # Make a copy of the original cycle.
    cyc0 = cyc.copy()
    # Add 100 seconds and 10% to the cycle time to allow for delay
    # caused by coasting.
    cyc = cyc.extend_time(absolute_time_s=120.0, time_fraction=0.25)
    man = fsim.Maneuver.create_from(cyc, veh.copy())
    # Set coasting variables
    d = man.to_pydict()
    # All coasting maneuvers require coast_allow to be set to True
    d["coast_allow"] = True
    # Speed at which a coasting vehicle initiates friction braking
    d["coast_brake_start_speed_meters_per_second"] = 8.9408  # 20 mph
    # Design deceleration while braking
    d["coast_brake_accel_meters_per_second_squared"] = -2.5
    # This parameter is only used when grade is present. If set to true,
    # the simulation will attempt to iterate to find a better representation
    # of grade over a step. If false, it will use a simple approximation.
    d["favor_grade_accuracy"] = True
    # If true, allow passing the "reference trace". Otherwise, coasting
    # vehicle will brake to stay at or behind the reference trace. If a
    # coasting vehicle is forced to apply brakes until the brake start
    # speed (and thus be short of coasting to the planned stop), the
    # vehicle will leave coasting mode and just follow the reference
    # trace.
    d["coast_allow_passing"] = True
    # The maximum allowable speed during coast. A vehicle can
    # increase speed during coast if going downhill. If going to
    # coast above this speed, friction brakes or regenerative braking
    # will be employed to prevent it.
    d["coast_max_speed_meters_per_second"] = 33.5280  # 75 mph
    # The time horizon for adjustement is a "look-ahead" metric for considering
    # whether to enter coast or not. The higher the time, the more chance of
    # taking advantage of coast. However, lower values may be more realistic
    # depending on what sensors and information technology the vehicle is
    # equipped with.
    d["coast_time_horizon_for_adjustment_seconds"] = 120.0
    # Reset the Maneuver object using the python dictionary
    man = fsim.Maneuver.from_pydict(d)
    # Modify the cycle and return it
    cyc = man.apply_maneuvers()
    # Now we can trim the maneuver cycle to only have as much
    # idle time at end as the original reference cycle
    cyc = cyc.trim_ending_idle(idle_to_keep_s=end_idle_duration_s)
    # Run simdrive using the modified cycle
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    if SHOW_PLOTS:
        c0 = cyc0.to_pydict()
        df = sd.to_dataframe()
        if LIST_COLUMN_OPTIONS:
            print("Available Columns:")
            for column_name in df.columns:
                print(f"- {column_name}")
        plot_speed_by_time(df, c0, title="Advanced Coasting Behavior")
        plot_speed_by_dist(df, c0, title="Advanced Coasting Behavior (distance-based)")


def basic_cruise_demo():
    """Demonstrate basic Eco-Cruise usage"""
    veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    veh.set_save_interval(1)
    cyc = fsim.Cycle.from_resource("udds.csv")
    end_idle_duration_s = cyc.ending_idle_time_s()
    # Add 100 seconds and 10% to the cycle time to allow for delay
    # caused by coasting.
    cyc0 = cyc.copy()
    cyc = cyc.extend_time(absolute_time_s=240.0, time_fraction=0.3)
    vavg = cyc0.average_speed_m_per_s(while_moving=True)
    man = fsim.Maneuver.create_from(cyc, veh.copy())
    # Set coasting variables
    d = man.to_pydict()
    d["idm_allow"] = True
    d["idm_v_desired_m_per_s"] = vavg
    d["idm_dt_headway_s"] = 1.0
    d["idm_minimum_gap_m"] = 1.0
    d["idm_delta"] = 4.0
    d["idm_accel_m_per_s2"] = 1.0
    d["idm_decel_m_per_s2"] = 2.5
    # Reset the Maneuver object using the python dictionary
    man = fsim.Maneuver.from_pydict(d)
    # Modify the cycle and return it
    cyc = man.apply_maneuvers()
    # Trim the manipulated cycle down so it has the same idle
    # duration as the original cycle
    cyc = cyc.trim_ending_idle(idle_to_keep_s=end_idle_duration_s)
    # Run simdrive using the modified cycle
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    if SHOW_PLOTS:
        c0 = cyc0.to_pydict()
        df = sd.to_dataframe()
        if LIST_COLUMN_OPTIONS:
            print("Available Columns:")
            for column_name in df.columns:
                print(f"- {column_name}")
        plot_speed_by_time(df, c0, title="Basic Cruise Behavior")
        plot_speed_by_dist(df, c0, title="Basic Cruise Behavior (distance-based)")


def cruise_and_coast_demo():
    """Demonstrate both cruise and coast"""
    # veh = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")
    veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    veh.set_save_interval(1)
    cyc = fsim.Cycle.from_resource("udds.csv")
    # We can query to see how much idle time exists at the end of
    # the reference cycle so we can (eventually) duplicate that on
    # the modified cycle.
    end_idle_duration_s = cyc.ending_idle_time_s()
    # Add 100 seconds and 10% to the cycle time to allow for delay
    # caused by coasting.
    cyc = cyc.extend_time(absolute_time_s=120.0, time_fraction=0.25)
    # Make a copy of the cycle AFTER it was extended.
    # NOTE: this is mainly needed for plotting purposes
    cyc0 = cyc.copy()
    vavg = cyc0.average_speed_m_per_s(while_moving=True)
    man = fsim.Maneuver.create_from(cyc, veh.copy())
    # Set coasting variables
    d = man.to_pydict()
    # All coasting maneuvers require coast_allow to be set to True
    d["coast_allow"] = True
    # Speed at which a coasting vehicle initiates friction braking
    d["coast_brake_start_speed_meters_per_second"] = 8.9408  # 20 mph
    # Design deceleration while braking
    d["coast_brake_accel_meters_per_second_squared"] = -2.5
    # This parameter is only used when grade is present. If set to true,
    # the simulation will attempt to iterate to find a better representation
    # of grade over a step. If false, it will use a simple approximation.
    d["favor_grade_accuracy"] = True
    # If true, allow passing the "reference trace". Otherwise, coasting
    # vehicle will brake to stay at or behind the reference trace. If a
    # coasting vehicle is forced to apply brakes until the brake start
    # speed (and thus be short of coasting to the planned stop), the
    # vehicle will leave coasting mode and just follow the reference
    # trace.
    d["coast_allow_passing"] = True
    # The maximum allowable speed during coast. A vehicle can
    # increase speed during coast if going downhill. If going to
    # coast above this speed, friction brakes or regenerative braking
    # will be employed to prevent it.
    d["coast_max_speed_meters_per_second"] = 33.5280  # 75 mph
    # The time horizon for adjustement is a "look-ahead" metric for considering
    # whether to enter coast or not. The higher the time, the more chance of
    # taking advantage of coast. However, lower values may be more realistic
    # depending on what sensors and information technology the vehicle is
    # equipped with.
    d["coast_time_horizon_for_adjustment_seconds"] = 120.0
    # Add intelligent driver model (IDM) parameters
    d["idm_allow"] = True
    d["idm_v_desired_m_per_s"] = vavg
    d["idm_dt_headway_s"] = 1.0
    d["idm_minimum_gap_m"] = 1.0
    d["idm_delta"] = 4.0
    d["idm_accel_m_per_s2"] = 1.0
    d["idm_decel_m_per_s2"] = 2.5
    # Reset the Maneuver object using the python dictionary
    man = fsim.Maneuver.from_pydict(d)
    mand = man.to_pydict()
    # Modify the cycle and return it
    cyc = man.apply_maneuvers()
    # Now we can trim the maneuver cycle to only have as much
    # idle time at end as the original reference cycle
    cyc = cyc.trim_ending_idle(idle_to_keep_s=end_idle_duration_s)
    # Run simdrive using the modified cycle
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    if SHOW_PLOTS:
        c0 = cyc0.to_pydict()
        df = sd.to_dataframe()
        is_coast = np.array(mand["impose_coast"]) * 5.0
        if LIST_COLUMN_OPTIONS:
            print("Available Columns:")
            for column_name in df.columns:
                print(f"- {column_name}")
        plot_speed_by_time(df, c0, is_coast, title="Coasting and Cruise")
        plot_speed_by_dist(df, c0, is_coast, title="Coasting and Cruise")


if __name__ == "__main__":
    microtrip_demo()
    basic_coasting_demo()
    advanced_coasting_demo()
    basic_cruise_demo()
    cruise_and_coast_demo()
    print("Done!")
