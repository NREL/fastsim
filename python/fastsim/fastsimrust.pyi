from __future__ import annotations
from typing_extensions import Self
from typing import Dict, List, Tuple, Optional, ByteString
from abc import ABC
from fastsim.vehicle import VEHICLE_DIR
import yaml
from pathlib import Path

class RustVec(ABC):
    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __getitem__(self, idx: int):
        ...

    def __setitem__(self, _idx, _new_value):
        ...

    def tolist(self):
        ...

    def __list__(self):
        ...

    def __len__(self):
        ...

    def is_empty(self):
        ...

class SerdeAPI(ABC):
    @classmethod
    def from_bincode(cls, encoded: ByteString) -> Self:
        ...

    def to_bincode(self) -> ByteString:
        ...

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        ...

    def to_yaml(self) -> str:
        ...

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        ...

    def to_json(self) -> str:
        ...

    @classmethod
    def from_file(cls, filename: str) -> Self:
        ...

    def to_file(self, filename: str) -> Self:
        ...



class Pyo3ArrayI32(SerdeAPI, RustVec):
    """Helper struct to allow Rust to return a Python class that will indicate to the user that it's
    a clone.  """

class Pyo3ArrayU32(SerdeAPI, RustVec):
    """Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  """

class Pyo3ArrayF64(SerdeAPI, RustVec):
    """Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  """


class Pyo3ArrayBool(SerdeAPI, RustVec):
    """Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  """


class Pyo3VecF64(SerdeAPI, RustVec):
    """Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  """


class SimDriveVec(SerdeAPI, RustVec): 
    """Vector of RustSimDrive"""
    def sim_drive(self, parallelize: bool = True, verbose: bool = False):
        ...

    def push(self, sd: RustSimDrive):
        """Push a new element onto the end"""
        ...

    def pop(self) -> Optional[RustSimDrive]: 
        """Removed and return the last element"""
        ...

    def remove(self, idx: int):
        """Remove element at `idx`"""
        ...
    
    def insert(self, idx: int, sd: RustSimDrive):
        """Insert `sd` before element `idx`"""
        ...


class RustPhysicalProperties(SerdeAPI):
    """Struct containing time trace data"""
    a_grav_mps2: float
    air_density_kg_per_m3: float
    fuel_afr_stoich: float
    fuel_rho_kg__L: float
    kwh_per_gge: float
    orphaned: bool

    def reset_orphaned(self) -> None:
        """Reset the orphaned flag to false."""
        ...

class RustCycle(SerdeAPI):
    """Struct for containing:
    * time_s, cycle time, $s$  
    * mps, vehicle speed, $\\frac{m}{s}$  
    * grade, road grade/slope, $\\frac{rise}{run}$  
    * road_type, $kW$  
    * legacy, will likely change to road charging capacity

    # Python Examples
    ```python
    import fastsim

    ## Load drive cycle by name
    cyc = fastsim.cycle.Cycle.from_file("udds").to_rust()
    ```"""
    delta_elev_m: list
    dist_m: list
    dt_s: list
    grade: Pyo3ArrayF64
    'array of grade [rise/run]'
    len: int
    mph: list
    mps: Pyo3ArrayF64
    'array of speed [m/s]'
    name: str
    orphaned: bool
    road_type: Pyo3ArrayF64
    'array of max possible charge rate from roadway'
    time_s: Pyo3ArrayF64
    'array of time [s]'

    def average_grade_over_range(self, distance_start_m: float, delta_distance_m: float) -> float:
        ...

    def calc_distance_to_next_stop_from(self, distance_m: float) -> float:
        ...

    def to_dict(self) -> Dict[str, List[float]]:
        """Return a HashMap representing the cycle"""
        ...

    def modify_by_const_jerk_trajectory(
        self, idx: int, n: int, jerk_m_per_s3: float, accel0_m_per_s2: float) -> float:
        ...

    def modify_with_braking_trajectory(
        self, 
        brake_accel_m_per_s2: float, 
        idx: int, 
        dts_m: Optional[float]
    ) -> Tuple[float, int]:
        ...

    def to_rust(self) -> Self:
        ...

    def copy(self) -> Self:
        ...

    def reset_orphaned(self):
        """Reset the orphaned flag to false."""
        ...

class RustVehicle(SerdeAPI):
    """Struct containing vehicle attributes

    # Python Examples
    ```python
    import fastsim

    ## Load vehicle from vehicle database
    vehno = 5
    # optional to_rust=True parameter skips Python-side calculations
    veh = fastsim.vehicle.Vehicle.from_vehdb(vehno, to_rust=True).to_rust()
    ```"""
    alt_eff: float
    'Alternator efficiency'
    aux_kw: float
    'Auxiliary power load, $kW$'
    cargo_kg: float
    'Cargo mass including passengers, $kg$'
    charging_on: bool
    chg_eff: float
    'Charger efficiency'
    comp_mass_multiplier: float
    'Component mass multiplier for vehicle mass calculation'
    drag_coef: float
    'Aerodynamic drag coefficient'
    drive_axle_weight_frac: float
    'Fraction of weight on the drive axle while stopped'
    ess_base_kg: float
    'Traction battery base mass, $kg$'
    ess_chg_to_fc_max_eff_perc: float
    'ESS charge effort toward max FC efficiency'
    ess_dischg_to_fc_max_eff_perc: float
    'ESS discharge effort toward max FC efficiency'
    ess_kg_per_kwh: float
    'Traction battery mass per energy, $\x0crac{kg}{kWh}$'
    ess_life_coef_a: float
    'Traction battery cycle life coefficient A, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)'
    ess_life_coef_b: float
    'Traction battery cycle life coefficient B, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)'
    ess_mass_kg: float
    ess_max_kw: float
    'Traction battery maximum power output, $kW$'
    ess_max_kwh: float
    'Traction battery energy capacity, $kWh$'
    ess_round_trip_eff: float
    'Traction battery round-trip efficiency'
    ess_to_fuel_ok_error: float
    'Maximum acceptable overall change in ESS energy relative to energy from fuel (HEV SOC balancing only), $\x0crac{\\Delta E_{ESS}}{\\Delta E_{fuel}}$'
    fc_base_kg: float
    'Fuel converter base mass, $kg$'
    fc_eff_array: Pyo3VecF64
    fc_eff_map: Pyo3ArrayF64
    'Fuel converter efficiency map'
    fc_eff_type: str
    'Fuel converter efficiency type, one of \\[[SI](SI), [ATKINSON](ATKINSON), [DIESEL](DIESEL), [H2FC](H2FC), [HD_DIESEL](HD_DIESEL)\\]  \n    Used for calculating [fc_eff_map](RustVehicle::fc_eff_map), and other calculations if H2FC'
    fc_kw_out_array: Pyo3VecF64
    fc_kw_per_kg: float
    'Fuel converter specific power (power-to-weight ratio), $\x0crac{kW}{kg}$'
    fc_mass_kg: float
    fc_max_kw: float
    'Fuel converter peak continuous power, $kW$'
    fc_peak_eff: float
    fc_peak_eff_override: Optional[float]
    'Fuel converter efficiency peak override, scales entire curve'
    fc_perc_out_array: Pyo3VecF64
    fc_pwr_out_perc: Pyo3ArrayF64
    'Fuel converter output power percentage map, x-values of [fc_eff_map](RustVehicle::fc_eff_map)'
    fc_sec_to_peak_pwr: float
    'Fuel converter time to peak power, $s$'
    force_aux_on_fc: bool
    'Force auxiliary power load to come from fuel converter'
    frontal_area_m2: float
    'Frontal area, $m^2$'
    fs_kwh: float
    'Fuel storage energy capacity, $kWh$'
    fs_kwh_per_kg: float
    'Fuel specific energy, $\x0crac{kWh}{kg}$'
    fs_mass_kg: float
    fs_max_kw: float
    'Fuel storage max power output, $kW$'
    fs_secs_to_peak_pwr: float
    'Fuel storage time to peak power, $s$'
    glider_kg: float
    'Vehicle mass excluding cargo, passengers, and powertrain components, $kg$'
    idle_fc_kw: float
    'Fuel converter idle power, $kW$'
    input_kw_out_array: Pyo3ArrayF64
    kw_demand_fc_on: float
    'Power demand above which to require fuel converter on, $kW$'
    large_motor_power_kw: float
    max_accel_buffer_mph: float
    'Speed where the battery reserved for accelerating is zero, $mph$'
    max_accel_buffer_perc_of_useable_soc: float
    'Percent of usable battery energy reserved to help accelerate'
    max_fc_eff_kw: float
    max_regen: float
    'Maximum regenerative braking efficiency'
    max_roadway_chg_kw: Pyo3ArrayF64
    max_soc: float
    'Traction battery maximum state of charge'
    max_trac_mps2: float
    mc_eff_array: Pyo3ArrayF64
    mc_eff_map: Pyo3ArrayF64
    'Electric motor efficiency map'
    mc_full_eff_array: Pyo3VecF64
    mc_kw_in_array: Pyo3VecF64
    mc_kw_out_array: Pyo3VecF64
    mc_mass_kg: float
    mc_max_elec_in_kw: float
    mc_max_kw: float
    'Peak continuous electric motor power, $kW$'
    mc_pe_base_kg: float
    'Motor power electronics base mass, $kg$'
    mc_pe_kg_per_kw: float
    'Motor power electronics mass per power output, $\x0crac{kg}{kW}$'
    mc_peak_eff: float
    mc_peak_eff_override: Optional[float]
    'Motor efficiency peak override, scales entire curve'
    mc_perc_out_array: Pyo3VecF64
    mc_pwr_out_perc: Pyo3ArrayF64
    'Electric motor output power percentage map, x-values of [mc_eff_map](RustVehicle::mc_eff_map)'
    mc_sec_to_peak_pwr: float
    'Electric motor time to peak power, $s$'
    min_fc_time_on: float
    'Minimum time fuel converter must be on before shutoff (for HEV, PHEV)'
    min_soc: float
    'Traction battery minimum state of charge'
    modern_max: float
    mph_fc_on: float
    'Speed at which the fuel converter must turn on, $mph$'
    no_elec_aux: bool
    no_elec_sys: bool
    num_wheels: float
    'Number of wheels'
    orphaned: bool
    perc_high_acc_buf: float
    'Percent SOC buffer for high accessory loads during cycles with long idle time'
    props: RustPhysicalProperties
    'Physical properties, see [RustPhysicalProperties](RustPhysicalProperties)'
    regen_a: float
    regen_b: float
    scenario_name: str
    'Vehicle name'
    selection: int
    'Vehicle database ID'
    small_motor_power_kw: float
    stop_start: bool
    'Stop/start micro-HEV flag'
    trans_eff: float
    'Transmission efficiency'
    trans_kg: float
    'Transmission mass, $kg$'
    val0_to60_mph: float
    val_cd_range_mi: float
    val_comb_kwh_per_mile: float
    val_comb_mpgge: float
    val_const45_mph_kwh_per_mile: float
    val_const55_mph_kwh_per_mile: float
    val_const60_mph_kwh_per_mile: float
    val_const65_mph_kwh_per_mile: float
    val_ess_life_miles: float
    val_hwy_kwh_per_mile: float
    val_hwy_mpgge: float
    val_msrp: float
    val_range_miles: float
    val_udds_kwh_per_mile: float
    val_udds_mpgge: float
    val_unadj_hwy_kwh_per_mile: float
    val_unadj_udds_kwh_per_mile: float
    val_veh_base_cost: float
    veh_cg_m: float
    'Vehicle center of mass height, $m$\n    **NOTE:** positive for FWD, negative for RWD, AWD, 4WD'
    veh_kg: float
    veh_override_kg: float
    'Total vehicle mass, overrides mass calculation, $kg$'
    veh_pt_type: str
    'Vehicle powertrain type, one of \\[[CONV](CONV), [HEV](HEV), [PHEV](PHEV), [BEV](BEV)\\]'
    veh_year: int
    'Vehicle year'
    wheel_base_m: float
    'Wheelbase, $m$'
    wheel_coef_of_fric: float
    'Wheel coefficient of friction'
    wheel_inertia_kg_m2: float
    'Mass moment of inertia per wheel, $kg \\cdot m^2$'
    wheel_radius_m: float
    'Wheel radius, $m$'
    wheel_rr_coef: float
    'Rolling resistance coefficient'

    def get_max_regen_kwh(self) -> float:
        ...

    def set_derived(self) -> None:
        ...

    def set_veh_mass(self) -> None:
        ...

    def to_rust(self) -> Self:
        """An identify function to allow RustVehicle to be used as a python vehicle and respond to this method
        Returns a clone of the current object"""
        ...

    def copy(self) -> Self:
        ...

    def reset_orphaned(self) -> None:
        """Reset the orphaned flag to false."""
        ...

class RustSimDrive(SerdeAPI):
    accel_buff_soc: Pyo3ArrayF64
    accel_kw: Pyo3ArrayF64
    add_kwh: Pyo3ArrayF64
    ascent_kj: float
    ascent_kw: Pyo3ArrayF64
    aux_in_kw: Pyo3ArrayF64
    aux_kj: float
    battery_kwh_per_mi: float
    brake_kj: float
    can_pwr_all_elec: Pyo3ArrayBool
    coast_delay_index: Pyo3ArrayI32
    cur_ess_max_kw_out: Pyo3ArrayF64
    cur_max_avail_elec_kw: Pyo3ArrayF64
    cur_max_elec_kw: Pyo3ArrayF64
    cur_max_ess_chg_kw: Pyo3ArrayF64
    cur_max_fc_kw_out: Pyo3ArrayF64
    cur_max_fs_kw_out: Pyo3ArrayF64
    cur_max_mc_elec_kw_in: Pyo3ArrayF64
    cur_max_mc_kw_out: Pyo3ArrayF64
    cur_max_roadway_chg_kw: Pyo3ArrayF64
    cur_max_trac_kw: Pyo3ArrayF64
    cur_max_trans_kw_out: Pyo3ArrayF64
    cur_soc_target: Pyo3ArrayF64
    cyc: RustCycle
    cyc0: RustCycle
    cyc_fric_brake_kw: Pyo3ArrayF64
    cyc_met: Pyo3ArrayBool
    cyc_regen_brake_kw: Pyo3ArrayF64
    cyc_tire_inertia_kw: Pyo3ArrayF64
    cyc_trac_kw_req: Pyo3ArrayF64
    cyc_trans_kw_out_req: Pyo3ArrayF64
    cyc_whl_kw_req: Pyo3ArrayF64
    cyc_whl_rad_per_sec: Pyo3ArrayF64
    desired_ess_kw_out_for_ae: Pyo3ArrayF64
    dist_m: Pyo3ArrayF64
    dist_mi: Pyo3ArrayF64
    dod_cycs: Pyo3ArrayF64
    drag_kj: float
    drag_kw: Pyo3ArrayF64
    elec_kw_req_4ae: Pyo3ArrayF64
    electric_kwh_per_mi: float
    energy_audit_error: float
    er_ae_kw_out: Pyo3ArrayF64
    er_kw_if_fc_req: Pyo3ArrayF64
    ess2fuel_kwh: float
    ess_accel_buff_chg_kw: Pyo3ArrayF64
    ess_accel_regen_dischg_kw: Pyo3ArrayF64
    ess_ae_kw_out: Pyo3ArrayF64
    ess_cap_lim_chg_kw: Pyo3ArrayF64
    ess_cap_lim_dischg_kw: Pyo3ArrayF64
    ess_cur_kwh: Pyo3ArrayF64
    ess_desired_kw_4fc_eff: Pyo3ArrayF64
    ess_dischg_kj: float
    ess_eff_kj: float
    ess_kw_if_fc_req: Pyo3ArrayF64
    ess_kw_out_ach: Pyo3ArrayF64
    cur_max_mech_mc_kw_in: Pyo3ArrayF64
    ess_lim_mc_regen_perc_kw: Pyo3ArrayF64
    ess_loss_kw: Pyo3ArrayF64
    ess_perc_dead: Pyo3ArrayF64
    ess_regen_buff_dischg_kw: Pyo3ArrayF64
    fc_forced_on: Pyo3ArrayBool
    fc_forced_state: Pyo3ArrayU32
    fc_kj: float
    fc_kw_gap_fr_eff: Pyo3ArrayF64
    fc_kw_in_ach: Pyo3ArrayF64
    fc_kw_out_ach: Pyo3ArrayF64
    fc_kw_out_ach_pct: Pyo3ArrayF64
    fc_time_on: Pyo3ArrayF64
    fc_trans_lim_kw: Pyo3ArrayF64
    fs_cumu_mj_out_ach: Pyo3ArrayF64
    fs_kw_out_ach: Pyo3ArrayF64
    fs_kwh_out_ach: Pyo3ArrayF64
    fuel_kj: float
    hev_sim_count: int
    high_acc_fc_on_tag: Pyo3ArrayBool
    i: int
    impose_coast: Pyo3ArrayBool
    ke_kj: float
    long_params: RustLongParams
    max_ess_accell_buff_dischg_kw: Pyo3ArrayF64
    max_ess_regen_buff_chg_kw: Pyo3ArrayF64
    max_trac_mps: Pyo3ArrayF64
    mc_elec_in_kw_for_max_fc_eff: Pyo3ArrayF64
    mc_elec_in_lim_kw: Pyo3ArrayF64
    mc_elec_kw_in_ach: Pyo3ArrayF64
    mc_elec_kw_in_if_fc_req: Pyo3ArrayF64
    mc_kj: float
    mc_kw_if_fc_req: Pyo3ArrayF64
    mc_mech_kw_4forced_fc: Pyo3ArrayF64
    mc_mech_kw_out_ach: Pyo3ArrayF64
    mc_transi_lim_kw: Pyo3ArrayF64
    min_ess_kw_2help_fc: Pyo3ArrayF64
    min_mc_kw_2help_fc: Pyo3ArrayF64
    mpgge: float
    mph_ach: Pyo3ArrayF64
    mps_ach: Pyo3ArrayF64
    net_kj: float
    newton_iters: Pyo3ArrayU32
    prev_fc_time_on: Pyo3ArrayF64
    props: RustPhysicalProperties
    reached_buff: Pyo3ArrayBool
    regen_buff_soc: Pyo3ArrayF64
    regen_contrl_lim_kw_perc: Pyo3ArrayF64
    roadway_chg_kj: float
    roadway_chg_kw_out_ach: Pyo3ArrayF64
    rr_kj: float
    rr_kw: Pyo3ArrayF64
    sim_params: RustSimDriveParams
    soc: Pyo3ArrayF64
    spare_trac_kw: Pyo3ArrayF64
    trace_miss: bool
    trace_miss_dist_frac: float
    trace_miss_iters: Pyo3ArrayU32
    trace_miss_speed_mps: float
    trace_miss_time_frac: float
    trans_kj: float
    trans_kw_in_ach: Pyo3ArrayF64
    trans_kw_out_ach: Pyo3ArrayF64
    veh: RustVehicle

    def __init__(self, cyc: RustCycle, veh: RustVehicle) -> Self:
        ...

    def gap_to_lead_vehicle_m(self) -> List[float]:
        """Provides the gap-with lead vehicle from start to finish"""
        ...

    def init_for_step(self, init_soc: float, aux_in_kw_override: Optional[List[float]]) -> None:
        """This is a specialty method which should be called prior to using
        sim_drive_step in a loop.
        Arguments
        ------------
        init_soc: initial battery state-of-charge (SOC) for electrified vehicles
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
        Default of None causes veh.aux_kw to be used."""
        ...

    def is_empty(self) -> bool:
        """Return self.cyc.time_is.is_empty()"""
        ...

    def len(self) -> int:
        """Return length of time arrays"""
        ...

    def set_ach_speed(self, i: int) -> None:
        ...

    def set_comp_lims(self, i: int) -> None:
        ...

    def set_fc_forced_state(self, i: int) -> None:
        """Calculate control variables related to engine on/off state
        Arguments
        ------------
        i: index of time step
        """
        ...

    def set_fc_power(self, i: int) -> None:
        """Sets power consumption values for the current time step.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_hybrid_cont_calcs(self, i: int) -> None:
        """Hybrid control calculations.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_hybrid_cont_decisions(self, i: int) -> None:
        """Hybrid control decisions.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_misc_calcs(self, i: int) -> None:
        """Sets misc. calculations at time step 'i'
        Arguments:
        ----------
        i: index of time step"""
        ...

    def set_post_scalars(self) -> None:
        """Sets scalar variables that can be calculated after a cycle is run.
        This includes mpgge, various energy metrics, and others"""
        ...

    def set_power_calcs(self, i: int) -> None:
        """Calculate power requirements to meet cycle and determine if
        cycle can be met.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_time_dilation(self, i: int) -> None:
        """Sets the time dilation for the current step.
        Arguments
        ------------
        i: index of time step"""
        ...

    def sim_drive(self, init_soc: Optional[float], aux_in_kw_override: Optional[List[float]]) -> None:
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        init_soc: initial SOC for electrified vehicles.  
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
            Default of None causes veh.aux_kw to be used."""
        ...

    def sim_drive_step(self) -> None:
        """Step through 1 time step."""
        ...

    def sim_drive_walk(self, init_soc: float, aux_in_kw_override: Optional[List[float]]) -> None:
        """Receives second-by-second cycle information, vehicle properties,
        and an initial state of charge and runs sim_drive_step to perform a
        backward facing powertrain simulation. Method 'sim_drive' runs this
        iteratively to achieve correct SOC initial and final conditions, as
        needed.

        Arguments
        ------------
        init_soc (optional): initial battery state-of-charge (SOC) for electrified vehicles
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
                None causes veh.aux_kw to be used."""
        ...

    def solve_step(self, i: int) -> None:
        """Perform all the calculations to solve 1 time step."""
        ...

    def copy(self) -> Self:
        ...

class RustSimDriveParams(SerdeAPI):
    """Struct containing time trace data"""
    coast_allow: bool
    coast_allow_passing: bool
    coast_brake_accel_m_per_s2: float
    coast_brake_start_speed_m_per_s: float
    coast_max_speed_m_per_s: float
    coast_start_speed_m_per_s: float
    coast_time_horizon_for_adjustment_s: float
    energy_audit_error_tol: float
    idm_allow: bool
    idm_accel_m_per_s2: float
    idm_decel_m_per_s2: float
    idm_delta: float
    idm_dt_headway_s: float
    idm_minimum_gap_m: float
    idm_v_desired_m_per_s: float
    max_epa_adj: float
    max_time_dilation: float
    max_trace_miss_iters: int
    min_time_dilation: float
    missed_trace_correction: bool
    newton_gain: float
    newton_max_iter: int
    newton_xtol: float
    orphaned: bool
    sim_count_max: int
    time_dilation_tol: float
    trace_miss_dist_tol: float
    trace_miss_speed_mps_tol: float
    trace_miss_time_tol: float

    def reset_orphaned(self) -> None:
        """Reset the orphaned flag to false."""
        ...

class ThermalState:
    """Struct containing thermal state variables for all thermal components"""
    fc_te_deg_c: float
    'fuel converter (engine) temperature [°C]'
    fc_eta_temp_coeff: float
    'fuel converter temperature efficiency correction'
    fc_qdot_per_net_heat: float
    'fuel converter heat generation per total heat release minus shaft power'
    fc_qdot_kw: float
    'fuel converter heat generation [kW]'
    fc_qdot_to_amb_kw: float
    'fuel converter convection to ambient [kW]'
    fc_qdot_to_htr_kw: float
    'fuel converter heat loss to heater core [kW]'
    fc_htc_to_amb: float
    'heat transfer coeff [W / (m ** 2 * K)] to amb after arbitration'
    fc_lambda: float
    'lambda (air/fuel ratio normalized w.r.t. stoich air/fuel ratio) -- 1 is reasonable default'
    fc_te_adiabatic_deg_c: float
    'lambda-dependent adiabatic flame temperature'
    cab_te_deg_c: float
    'cabin temperature [°C]'
    cab_qdot_solar_kw: float
    'cabin solar load [kw]'
    cab_qdot_to_amb_kw: float
    'cabin convection to ambient [kw]'
    cab_qdot_from_hvac: float
    'heat transfer to cabin from hvac system'
    cab_hvac_pwr_aux_kw: float
    'aux load from hvac'
    exh_mdot: float
    'exhaust mass flow rate [kg/s]'
    exh_hdot_kw: float
    'exhaust enthalpy flow rate [kw]'
    exhport_exh_te_in_deg_c: float
    'exhaust temperature at exhaust port inlet'
    exhport_qdot_to_amb: float
    'heat transfer from exhport to amb [kw]'
    exhport_te_deg_c: float
    'catalyst temperature [°C]'
    exhport_qdot_from_exh: float
    'convection from exhaust to exhport [W]\n\n    positive means exhport is receiving heat'
    exhport_qdot_net: float
    'net heat generation in cat [W]'
    cat_qdot: float
    'catalyst heat generation [W]'
    cat_htc_to_amb: float
    'catalytic converter convection coefficient to ambient [W / (m ** 2 * K)]'
    cat_qdot_to_amb: float
    'heat transfer from catalyst to ambient [W]'
    cat_te_deg_c: float
    'catalyst temperature [°C]'
    cat_exh_te_in_deg_c: float
    'exhaust temperature at cat inlet'
    cat_re_ext: float
    'catalyst external reynolds number'
    cat_qdot_from_exh: float
    'convection from exhaust to cat [W]\n\n    positive means cat is receiving heat'
    cat_qdot_net: float
    'net heat generation in cat [W]'
    amb_te_deg_c: float
    'ambient temperature'
    orphaned: bool

class VehicleThermal:
    """Struct for containing vehicle thermal (and related) parameters."""
    fc_l: float
    cab_c_kj__k: float
    fc_exp_minimum: float
    orphaned: bool
    fc_coeff_from_comb: float
    tstat_te_sto_deg_c: float
    fc_c_kj__k: float
    tstat_te_delta_deg_c: float
    cab_l_width: float
    cab_r_to_amb: float
    cab_l_length: float
    cat_c_kj__K: float
    exhport_ha_int: float
    cat_l: float
    fc_exp_offset: float
    cat_htc_to_amb_stop: float
    cat_fc_eta_coeff: float
    fc_htc_to_amb_stop: float
    exhport_ha_to_amb: float
    fc_exp_lag: float
    ess_c_kj_k: float
    cat_te_lightoff_deg_c: float
    cab_htc_to_amb_stop: float
    exhport_c_kj__k: float
    rad_eps: float
    ess_htc_to_amb: float

    @classmethod
    def default(cls) -> Self:
        ...

    def copy(self) -> Self:
        ...

    def set_cabin_model_internal(self, te_set_deg_c: float, p_cntrl_kw_per_deg_c: float, i_cntrl_kw_per_deg_c_scnds: float, i_cntrl_max_kw: float, te_deadband_deg_c: float):
        ...

    def set_cabin_model_external(self):
        ...

    def set_fc_model_internal_exponential(self, offset: float, lag: float, minimum: float, fc_temp_eff_component: str):
        ...

    def reset_orphaned(self) -> None:
        """Reset the orphaned flag to false."""
        ...

    @classmethod
    def from_file(cls, filename: str) -> Self:
        ...

class ThermalStateHistoryVec(SerdeAPI, RustVec):
    ...

class SimDriveHot(SerdeAPI):
    sd: RustSimDrive
    vehthrm: VehicleThermal
    state: ThermalState
    history: ThermalStateHistoryVec
    amb_te_deg_c: Optional[List[float]]

    def __init__(
        self, 
        cyc: RustCycle, 
        veh: RustVehicle, 
        vehthrm: VehicleThermal, 
        init_state: Optional[ThermalState], 
        amb_te_deg_c: Optional[List[float]]
    ) -> Self:
        ...

    def copy(self) -> Self:
        ...

    def gap_to_lead_vehicle_m(self) -> Pyo3VecF64:
        """Provides the gap-with lead vehicle from start to finish"""
        ...

    def init_for_step(
        self,
        init_soc: float, 
        aux_in_kw_override: Optional[List[float]]
    ) -> None:
        """This is a specialty method which should be called prior to using
        sim_drive_step in a loop.
        Arguments
        ------------
        init_soc: initial battery state-of-charge (SOC) for electrified vehicles
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
        Default of None causes veh.aux_kw to be used."""
        ...

    def is_empty(self) -> bool:
        """added to make clippy happy
        not sure whether there is any benefit to this or not for our purposes
        Return self.cyc.time_is.is_empty()"""
        ...

    def len(self) -> int:
        """Return length of time arrays"""
        ...

    def set_ach_speed(self, i: int) -> None:
        ...

    def set_comp_lims(self, i: int) -> None:
        ...

    def set_fc_forced_state(self, i: int) -> None:
        """Calculate control variables related to engine on/off state
        Arguments
        ------------
        i: index of time step
        """
        ...

    def set_fc_power(self, i: int) -> None:
        """Sets power consumption values for the current time step.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_hybrid_cont_calcs(self, i: int) -> None:
        """Hybrid control calculations.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_hybrid_cont_decisions(self, i: int) -> None:
        """Hybrid control decisions.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_misc_calcs(self, i: int) -> None:
        """Sets misc. calculations at time step 'i'
        Arguments:
        ----------
        i: index of time step"""
        ...

    def set_post_scalars(self) -> None:
        """Sets scalar variables that can be calculated after a cycle is run.
        This includes mpgge, various energy metrics, and others"""
        ...

    def set_power_calcs(self, i: int) -> None:
        """Calculate power requirements to meet cycle and determine if
        cycle can be met.
        Arguments
        ------------
        i: index of time step"""
        ...

    def set_time_dilation(self, i: int) -> None:
        """Sets the time dilation for the current step.
        Arguments
        ------------
        i: index of time step"""
        ...

    def sim_drive(
        self, 
        init_soc: float, 
        aux_in_kw_override: Optional[List[float]]
    ) -> None:
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        init_soc: initial SOC for electrified vehicles.
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
            Default of None causes veh.aux_kw to be used."""
        ...

    def sim_drive_step(self) -> None:
        """Step through 1 time step."""
        ...

    def sim_drive_walk(
        self, 
        init_soc: float, 
        aux_in_kw_override: Optional[List[float]]
    ) -> None:
        """Receives second-by-second cycle information, vehicle properties,
        and an initial state of charge and runs sim_drive_step to perform a
        backward facing powertrain simulation. Method 'sim_drive' runs this
        iteratively to achieve correct SOC initial and final conditions, as
        needed.

        Arguments
        ------------
        init_soc (optional): initial battery state-of-charge (SOC) for electrified vehicles
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
                None causes veh.aux_kw to be used."""
        ...

    def solve_step(self, i: int) -> None:
        """Perform all the calculations to solve 1 time step."""
        ...

def abc_to_drag_coeffs(
    veh: RustVehicle,
    a_lbf: float,
    b_lbf__mph: float,
    c_lbf__mph2: float,
    custom_rho: Optional[bool],
    custom_rho_temp_degC: Optional[float],
    custom_rho_elevation_m: Optional[float],
    simdrive_optimize: Optional[bool],
    _show_plots: Optional[bool],
) -> Tuple[float, float]:
    ...
    
class LabelFe(SerdeAPI):
    veh: RustVehicle
    adj_params: AdjCoef
    lab_udds_mpgge: float
    lab_hwy_mpgge: float
    lab_comb_mpgge: float
    lab_udds_kwh_per_mi: float
    lab_hwy_kwh_per_mi: float
    lab_comb_kwh_per_mi: float
    adj_udds_mpgge: float
    adj_hwy_mpgge: float
    adj_comb_mpgge: float
    adj_udds_kwh_per_mi: float
    adj_hwy_kwh_per_mi: float
    adj_comb_kwh_per_mi: float
    adj_udds_ess_kwh_per_mi: float
    adj_hwy_ess_kwh_per_mi: float
    adj_comb_ess_kwh_per_mi: float
    net_range_miles: float
    uf: float
    net_accel: float
    res_found: str
    phev_calcs: Optional[LabelFePHEV]
    adj_cs_comb_mpgge: Optional[float]
    adj_cd_comb_mpgge: Optional[float]
    net_phev_cd_miles: Optional[float]
    trace_miss_speed_mph: float
    

class LabelFePHEV(SerdeAPI):
    regen_soc_buffer: float
    udds: PHEVCycleCalc
    hwy: PHEVCycleCalc


class PHEVCycleCalc(SerdeAPI):
    cd_ess_kwh: float
    cd_ess_kwh_per_mi: float
    cd_fs_gal: float
    cd_fs_kwh: float
    cd_mpg: float
    cd_cycs: float
    cd_miles: float
    cd_lab_mpg: float
    cd_adj_mpg: float
    cd_frac_in_trans: float
    trans_init_soc: float
    trans_ess_kwh: float
    trans_ess_kwh_per_mi: float
    trans_fs_gal: float
    trans_fs_kwh: float
    cs_ess_kwh: float
    cs_ess_kwh_per_mi: float
    cs_fs_gal: float
    cs_fs_kwh: float
    cs_mpg: float
    lab_mpgge: float
    lab_kwh_per_mi: float
    lab_uf: float
    lab_uf_gpm: List[float]
    lab_iter_uf: List[float]
    lab_iter_uf_kwh_per_mi: List[float]
    lab_iter_kwh_per_mi: List[float]
    adj_iter_mpgge: List[float]
    adj_iter_kwh_per_mi: List[float]
    adj_iter_cd_miles: List[float]
    adj_iter_uf: List[float]
    adj_iter_uf_gpm: List[float]
    adj_iter_uf_kwh_per_mi: List[float]
    adj_cd_miles: float
    adj_cd_mpgge: float
    adj_cs_mpgge: float
    adj_uf: float
    adj_mpgge: float
    adj_kwh_per_mi: float
    adj_ess_kwh_per_mi: float
    delta_soc: float
    total_cd_miles: float


def make_accel_trace() -> RustCycle:
    ...
def get_net_accel(sd_accel: RustSimDrive, scenario_name: str) -> float:
    ...
class AdjCoef(SerdeAPI):
    def __init__(self):
        self.city_intercept: float = 0.0
        self.city_slope: float = 0.0
        self.hwy_intercept: float = 0.0
        self.hwy_slope: float = 0.0
        
        
class AdjCoefMap:
    def __init__(self):
        self.adj_coef_map: Dict[str, AdjCoef] = {}
        
        
class RustLongParams(SerdeAPI):
    def __init__(self):
        self.rechg_freq_miles: List[float] = []
        self.uf_array: List[float] = []
        self.ld_fe_adj_coef: AdjCoefMap = {}
                        
def get_label_fe(
    veh: RustVehicle,
    full_detail: Optional[bool],
    verbose: Optional[bool],
) -> Tuple[LabelFe, Optional[Dict[str, RustSimDrive]]]:   
    ...
    
def get_label_fe_phev(
    veh: RustVehicle,
    sd: Dict[str, RustSimDrive],
    long_params: RustLongParams,
    adj_params: AdjCoef,
    sim_params: RustSimDriveParams,
    props: RustPhysicalProperties,
) -> LabelFePHEV:
    ...

def get_label_fe_conv(veh: RustVehicle) -> LabelFe:
   ...
   
   
