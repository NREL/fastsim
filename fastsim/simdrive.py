"""Module containing classes and methods for simulating vehicle drive cycle."""
import sys

# Import necessary python modules
from dataclasses import dataclass
from logging import debug
from typing import Optional, List, Tuple
import numpy as np
import re
import copy

from .rustext import RUST_AVAILABLE

if RUST_AVAILABLE:
    import fastsimrust as fsr
    from fastsimrust import RustSimDrive
from . import params, cycle, vehicle, inspect_utils

# these imports are needed for numba to type these correctly
from .vehicle import CONV, HEV, PHEV, BEV
from .vehicle import SI, ATKINSON, DIESEL, H2FC, HD_DIESEL

# Logging
import logging
logger = logging.getLogger(__name__)


class SimDriveParams(object):
    """Class containing attributes used for configuring sim_drive.
    Usually the defaults are ok, and there will be no need to use this.

    See comments in code for descriptions of various parameters that
    affect simulation behavior. If, for example, you want to suppress
    warning messages, use the following pastable code EXAMPLE:

    >>> import logging
    >>> logging.getLogger("fastsim").setLevel(logging.DEBUG)
    """

    @classmethod
    def from_dict(cls, sdp_dict):
        """Create from a dictionary"""
        sdp = cls()
        for k, v in sdp_dict.items():
            sdp.__setattr__(k, v)
        return sdp

    def __init__(self):
        """Default values that affect simulation behavior.  
        Can be modified after instantiation."""
        # if True, accuracy will be favored over performance for grade per step estimates
        # Specifically, for performance, grade for a step will be assumed to be the grade
        # looked up at step start distance. For accuracy, the actual elevations will be
        # used. This distinciton only makes a difference for CAV maneuvers.
        self.favor_grade_accuracy = True
        self.missed_trace_correction = False  # if True, missed trace correction is active, default = False
        # maximum time dilation factor to "catch up" with trace -- e.g. 1.0 means 100% increase in step size
        self.max_time_dilation = 1.0
        # minimum time dilation margin to let trace "catch up" -- e.g. -0.5 means 50% reduction in step size
        self.min_time_dilation = -0.5
        self.time_dilation_tol = 5e-4  # convergence criteria for time dilation
        # number of iterations to achieve time dilation correction
        self.max_trace_miss_iters = 5
        # threshold of error in speed [m/s] that triggers warning
        self.trace_miss_speed_mps_tol = 1.0
        # threshold for printing warning when time dilation is active
        self.trace_miss_time_tol = 1e-3
        # threshold of fractional eror in distance that triggers warning
        self.trace_miss_dist_tol = 1e-3
        self.sim_count_max = 30  # max allowable number of HEV SOC iterations
        self.newton_gain = 0.9  # newton solver gain
        self.newton_max_iter = 100  # newton solver max iterations
        self.newton_xtol = 1e-9  # newton solver tolerance
        # tolerance for energy audit error warning, i.e. 0.1%
        self.energy_audit_error_tol = 0.002
        self.coast_allow = False  # if True, coasting to stops are allowed
        # if True, coasting vehicle can eclipse the shadow trace
        self.coast_allow_passing = False
        self.coast_max_speed_m_per_s = 40.0  # maximum allowable speed under coast
        self.coast_brake_accel_m_per_s2 = -2.5
        # speed when coasting uses friction brakes
        self.coast_brake_start_speed_m_per_s = 7.5
        # m/s, if > 0, initiates coast when vehicle hits this speed; mostly for testing
        self.coast_start_speed_m_per_s = 0.0
        # time-ahead for speed changes to be considered to hit distance mark
        self.coast_time_horizon_for_adjustment_s = 20.0
        # IDM - Intelligent Driver Model, Adaptive Cruise Control version
        self.idm_allow = False
        self.idm_v_desired_m_per_s: float = 33.33
        self.idm_dt_headway_s: float = 1.0
        self.idm_minimum_gap_m: float = 2.0
        self.idm_delta: float = 4.0
        self.idm_accel_m_per_s2: float = 1.0
        self.idm_decel_m_per_s2: float = 1.5
        self.idm_v_desired_in_m_per_s_by_distance_m: Optional[List[Tuple[float, float]]] = None

        # EPA fuel economy adjustment parameters
        self.max_epa_adj = 0.3  # maximum EPA adjustment factor

    def to_rust(self):
        """Change to the Rust version"""
        return copy_sim_params(self, 'rust')
    
    def reset_orphaned(self):
        """Dummy method for flexibility between Rust/Python version interfaces"""
        pass

ref_sim_drive_params = SimDriveParams()


def copy_sim_params(sdp: SimDriveParams, return_type: str = None):
    """
    Returns copy of SimDriveParams.
    Arguments:
    sdp: instantianed SimDriveParams or RustSimDriveParams
    return_type: 
        default: infer from type of sdp
        'dict': dict
        'python': SimDriveParams 
        'rust': RustSimDriveParams
    deep: if True, uses deepcopy on everything
    """
    sdp_dict = {}

    for key in inspect_utils.get_attrs(ref_sim_drive_params):
        sdp_dict[key] = sdp.__getattribute__(key)

    if return_type is None:
        if RUST_AVAILABLE and type(sdp) == fsr.RustSimDriveParams:
            return_type = 'rust'
        elif type(sdp) == SimDriveParams:
            return_type = 'python'
        else:
            raise NotImplementedError(
                "Only implemented for rust, python, or legacy.")

    if return_type == 'dict':
        return sdp_dict
    elif return_type == 'python':
        return SimDriveParams.from_dict(sdp_dict)
    elif RUST_AVAILABLE and return_type == 'rust':
        return fsr.RustSimDriveParams(**sdp_dict)
    else:
        raise ValueError(f"Invalid return_type: '{return_type}'")


def sim_params_equal(a: SimDriveParams, b: SimDriveParams) -> bool:
    """
    Returns True if objects are structurally equal (i.e., equal by value), else false.
    Arguments:
    a: instantiated SimDriveParams object
    b: instantiated SimDriveParams object
    """
    if a is b:
        return True
    a_dict = copy_sim_params(a, 'dict')
    b_dict = copy_sim_params(b, 'dict')
    if len(a_dict) != len(b_dict):
        a_keyset = {k for k in a.keys()}
        b_keyset = {k for k in b.keys()}
        logger.debug(
            "key sets not equal:\n" +
            f"in a but not b: {a_keyset - b_keyset}\n" +
            f"in b but not a: {b_keyset - a_keyset}"
        )
        return False
    for k in a_dict.keys():
        if a_dict[k] != b_dict[k]:
            logger.debug(
                f'unequal for key "{k}"\n' +
                f"a['{k}'] = {repr(a_dict[k])}\n" +
                f"b['{k}'] = {repr(b_dict[k])}"
            )
            return False
    return True


@dataclass
class _RendezvousTrajectory:
    found_trajectory: bool
    idx: int
    n: int
    full_brake_steps: int
    jerk_m_per_s3: float
    accel0_m_per_s2: float
    accel_spread: float

class SimDrive(object):
    """Class containing methods for running FASTSim vehicle 
    fuel economy simulations. This class is not compiled and will 
    run slower for large batch runs.
    Arguments:
    ----------
    cyc: cycle.Cycle instance
    veh: vehicle.Vehicle instance"""

    def __init__(self, cyc: cycle.Cycle, veh: vehicle.Vehicle):
        """Initalizes arrays, given vehicle.Vehicle() and cycle.Cycle() as arguments.
        sim_params is needed only if non-default behavior is desired."""
        self.__init_objects__(cyc, veh)
        self.init_arrays()
        # initialized here for downstream classes that do not run sim_drive
        self.hev_sim_count = 0

    def __init_objects__(self, cyc: cycle.Cycle, veh: vehicle.Vehicle):
        self.veh = veh
        self.cyc = cycle.copy_cycle(cyc)  # this cycle may be manipulated
        # this cycle is not to be manipulated
        self.cyc0 = cycle.copy_cycle(cyc)
        self.sim_params = SimDriveParams()
        self.props = params.PhysicalProperties()

    def init_arrays(self):
        self.i = 1  # initialize step counter for possible use outside sim_drive_walk()
        cyc_len = self.cyc.len

        # Component Limits -- calculated dynamically
        self.cur_max_fs_kw_out = np.zeros(cyc_len, dtype=np.float64)
        self.fc_trans_lim_kw = np.zeros(cyc_len, dtype=np.float64)
        self.fc_fs_lim_kw = np.zeros(cyc_len, dtype=np.float64)
        self.fc_max_kw_in = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_fc_kw_out = np.zeros(cyc_len, dtype=np.float64)
        self.ess_cap_lim_dischg_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cur_ess_max_kw_out = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_avail_elec_kw = np.zeros(cyc_len, dtype=np.float64)
        self.ess_cap_lim_chg_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_ess_chg_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_elec_kw = np.zeros(cyc_len, dtype=np.float64)
        self.mc_elec_in_lim_kw = np.zeros(cyc_len, dtype=np.float64)
        self.mc_transi_lim_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_mc_kw_out = np.zeros(cyc_len, dtype=np.float64)
        self.ess_lim_mc_regen_perc_kw = np.zeros(
            cyc_len, dtype=np.float64)
        self.ess_lim_mc_regen_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_mech_mc_kw_in = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_trans_kw_out = np.zeros(cyc_len, dtype=np.float64)

        # Drive Train
        self.cyc_trac_kw_req = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_trac_kw = np.zeros(cyc_len, dtype=np.float64)
        self.spare_trac_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cyc_whl_rad_per_sec = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.cyc_tire_inertia_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cyc_whl_kw_req = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.regen_contrl_lim_kw_perc = np.zeros(
            cyc_len, dtype=np.float64)
        self.cyc_regen_brake_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cyc_fric_brake_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cyc_trans_kw_out_req = np.zeros(cyc_len, dtype=np.float64)
        self.cyc_met = np.array([False] * cyc_len, dtype=np.bool_)
        self.trans_kw_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.trans_kw_in_ach = np.zeros(cyc_len, dtype=np.float64)
        self.cur_soc_target = np.zeros(cyc_len, dtype=np.float64)
        self.min_mc_kw_2help_fc = np.zeros(cyc_len, dtype=np.float64)
        self.mc_mech_kw_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.mc_elec_kw_in_ach = np.zeros(cyc_len, dtype=np.float64)
        self.aux_in_kw = np.zeros(cyc_len, dtype=np.float64)
        self.impose_coast = np.array([False] * cyc_len, dtype=np.bool_)
        self.roadway_chg_kw_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.min_ess_kw_2help_fc = np.zeros(cyc_len, dtype=np.float64)
        self.ess_kw_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.fc_kw_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.fc_kw_out_ach_pct = np.zeros(cyc_len, dtype=np.float64)
        self.fc_kw_in_ach = np.zeros(cyc_len, dtype=np.float64)
        self.fs_kw_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.fs_cumu_mj_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.fs_kwh_out_ach = np.zeros(cyc_len, dtype=np.float64)
        self.ess_cur_kwh = np.zeros(cyc_len, dtype=np.float64)
        self.soc = np.zeros(cyc_len, dtype=np.float64)

        # Vehicle Attributes, Control Variables
        self.regen_buff_soc = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.ess_regen_buff_dischg_kw = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.max_ess_regen_buff_chg_kw = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.ess_accel_buff_chg_kw = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.accel_buff_soc = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.max_ess_accell_buff_dischg_kw = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.ess_accel_regen_dischg_kw = np.zeros(
            cyc_len, dtype=np.float64)
        self.mc_elec_in_kw_for_max_fc_eff = np.zeros(
            cyc_len, dtype=np.float64)
        self.elec_kw_req_4ae = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.can_pwr_all_elec = np.array(  # oddball
            [False] * cyc_len, dtype=np.bool_)
        self.desired_ess_kw_out_for_ae = np.zeros(
            cyc_len, dtype=np.float64)
        self.ess_ae_kw_out = np.zeros(cyc_len, dtype=np.float64)
        self.er_ae_kw_out = np.zeros(cyc_len, dtype=np.float64)
        self.ess_desired_kw_4fc_eff = np.zeros(cyc_len, dtype=np.float64)
        self.ess_kw_if_fc_req = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.cur_max_mc_elec_kw_in = np.zeros(cyc_len, dtype=np.float64)
        self.fc_kw_gap_fr_eff = np.zeros(cyc_len, dtype=np.float64)
        self.er_kw_if_fc_req = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.mc_elec_kw_in_if_fc_req = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.mc_kw_if_fc_req = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.fc_forced_on = np.array([False] * cyc_len, dtype=np.bool_)
        self.fc_forced_state = np.zeros(cyc_len, dtype=np.int32)
        self.mc_mech_kw_4forced_fc = np.zeros(cyc_len, dtype=np.float64)
        self.fc_time_on = np.zeros(cyc_len, dtype=np.float64)
        self.prev_fc_time_on = np.zeros(cyc_len, dtype=np.float64)

        # Additional Variables
        self.mps_ach = np.zeros(cyc_len, dtype=np.float64)
        self.mph_ach = np.zeros(cyc_len, dtype=np.float64)
        self.dist_m = np.zeros(cyc_len, dtype=np.float64)  # oddbal
        self.dist_mi = np.zeros(cyc_len, dtype=np.float64)  # oddball
        self.high_acc_fc_on_tag = np.array(
            [False] * cyc_len, dtype=np.bool_)
        self.reached_buff = np.array([False] * cyc_len, dtype=np.bool_)
        self.max_trac_mps = np.zeros(cyc_len, dtype=np.float64)
        self.add_kwh = np.zeros(cyc_len, dtype=np.float64)
        self.dod_cycs = np.zeros(cyc_len, dtype=np.float64)
        self.ess_perc_dead = np.zeros(
            cyc_len, dtype=np.float64)  # oddball
        self.drag_kw = np.zeros(cyc_len, dtype=np.float64)
        self.ess_loss_kw = np.zeros(cyc_len, dtype=np.float64)
        self.accel_kw = np.zeros(cyc_len, dtype=np.float64)
        self.ascent_kw = np.zeros(cyc_len, dtype=np.float64)
        self.rr_kw = np.zeros(cyc_len, dtype=np.float64)
        self.cur_max_roadway_chg_kw = np.zeros(cyc_len, dtype=np.float64)
        self.trace_miss_iters = np.zeros(cyc_len, dtype=np.float64)
        self.newton_iters = np.zeros(cyc_len, dtype=np.float64)
        self.coast_delay_index = np.zeros(cyc_len, dtype=np.int32)
        self.impose_coast = np.array([False] * cyc_len, dtype=np.bool_)
        self.idm_target_speed_m_per_s = np.zeros(cyc_len, dtype=np.float64)
        self.cyc0_cache = self.cyc0.build_cache()

    @property
    def gap_to_lead_vehicle_m(self):
        "Provides the gap-with lead vehicle from start to finish"
        # TODO: consider basing on dist_m?
        gaps_m = cycle.trapz_step_distances(self.cyc0).cumsum() - cycle.trapz_step_distances(self.cyc).cumsum()
        if self.sim_params.idm_allow:
            gaps_m += self.sim_params.idm_minimum_gap_m
        return gaps_m

    def sim_drive(self, init_soc: Optional[float] = None, aux_in_kw_override: Optional[np.ndarray] = None):
        """
        Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        init_soc: initial SOC for electrified vehicles.  
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
            Default of None causes veh.aux_kw to be used. 
        """

        self.hev_sim_count = 0

        if init_soc is None:
            if self.veh.veh_pt_type == CONV:  # Conventional
                # If no EV / Hybrid components, no SOC considerations.
                init_soc = (self.veh.max_soc + self.veh.min_soc) / 2.0

            elif self.veh.veh_pt_type == HEV:  # HEV
                #####################################
                ### Charge Balancing Vehicle SOC ###
                #####################################
                # Charge balancing SOC for HEV vehicle types. Iterating init_soc and comparing to final SOC.
                # Iterating until tolerance met or 30 attempts made.
                init_soc = (self.veh.max_soc + self.veh.min_soc) / 2.0
                ess_2fuel_kwh = 1.0
                while ess_2fuel_kwh > self.veh.ess_to_fuel_ok_error and self.hev_sim_count < self.sim_params.sim_count_max:
                    self.hev_sim_count += 1
                    self.sim_drive_walk(init_soc, aux_in_kw_override)
                    fuel_kj = np.sum(self.fs_kw_out_ach * self.cyc.dt_s)
                    roadway_chg_kj = np.sum(
                        self.roadway_chg_kw_out_ach * self.cyc.dt_s)
                    if (fuel_kj + roadway_chg_kj) > 0:
                        ess_2fuel_kwh = np.abs(
                            (self.soc[0] - self.soc[-1]) * self.veh.ess_max_kwh *
                            3_600 / (fuel_kj + roadway_chg_kj)
                        )
                    else:
                        ess_2fuel_kwh = 0.0
                    init_soc = min(1.0, max(0.0, self.soc[-1]))

            elif self.veh.veh_pt_type == PHEV or self.veh.veh_pt_type == BEV:  # PHEV and BEV
                # If EV, initializing initial SOC to maximum SOC.
                init_soc = self.veh.max_soc

        self.sim_drive_walk(init_soc, aux_in_kw_override)
    
    def init_for_step(self, init_soc: float, aux_in_kw_override: Optional[np.ndarray] = None):
        """
        This is a specialty method which should be called prior to using
        sim_drive_step in a loop.

        Arguments
        ------------
        init_soc: initial battery state-of-charge (SOC) for electrified vehicles
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
                Default of None causes veh.aux_kw to be used. 
        """
        if (self.veh.veh_pt_type != CONV) and (init_soc > self.veh.max_soc or init_soc < self.veh.min_soc):
            raise ValueError(f"provided init_soc={init_soc} is outside range min_soc={self.veh.min_soc} to max_soc={self.veh.max_soc}")

        ############################
        ###   Loop Through Time  ###
        ############################

        ###  Assign First Values  ###
        # Drive Train
        self.init_arrays()  # reinitialize arrays for each new run

        if aux_in_kw_override is not None:
            if len(aux_in_kw_override) == len(self.aux_in_kw):
                self.aux_in_kw = aux_in_kw_override
            else:
                logger.warning(
                    f"provided aux_in_kw_override is not the right length; "
                    + f"needs {len(self.aux_in_kw)} elements"
                )

        self.cyc_met[0] = True
        self.cur_soc_target[0] = self.veh.max_soc
        self.ess_cur_kwh[0] = init_soc * self.veh.ess_max_kwh
        self.soc[0] = init_soc
        self.mps_ach[0] = self.cyc0.mps[0]
        self.mph_ach[0] = self.cyc0.mph[0]

        if self.sim_params.missed_trace_correction or self.sim_params.idm_allow or self.sim_params.coast_allow:
            # reset the cycle in case it has been manipulated
            self.cyc = cycle.copy_cycle(self.cyc0)

        self.i = 1  # time step counter

    def sim_drive_walk(self, init_soc: float, aux_in_kw_override: Optional[np.ndarray] = None):
        """
        Receives second-by-second cycle information, vehicle properties, 
        and an initial state of charge and runs sim_drive_step to perform a 
        backward facing powertrain simulation. Method 'sim_drive' runs this
        iteratively to achieve correct SOC initial and final conditions, as 
        needed.

        Arguments
        ------------
        init_soc: initial battery state-of-charge (SOC) for electrified vehicles
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
                Default of None causes veh.aux_kw to be used. 
        """
        self.init_for_step(init_soc, aux_in_kw_override)

        while self.i < len(self.cyc.time_s):
            self.sim_drive_step()

        if (self.cyc.dt_s > 5).any():
            if self.sim_params.missed_trace_correction:
                logger.info(
                    f"max time dilation factor = " +
                    f"{round((self.cyc.dt_s / self.cyc0.dt_s).max(), 3)}"
                )
            logger.warning(
                "large time steps affect accuracy significantly; " +
                f"max time step = {round(self.cyc.dt_s.max(), 3)}"
            )

        self.set_post_scalars()
    
    def activate_eco_cruise(
        self,
        by_microtrip: bool=False,
        extend_fraction: float=0.1,
        blend_factor: float=0.0,
        min_target_speed_m_per_s: float=8.0
    ):
        """
        Sets the intelligent driver model parameters for an eco-cruise driving trajectory.
        This is a convenience method instead of setting the sim_params.idm* parameters yourself.

        - by_microtrip: bool, if True, target speed is set by microtrip, else by cycle
        - extend_fraction: float, the fraction of time to extend the cycle to allow for catch-up
            of the following vehicle
        - blend_factor: float, a value between 0 and 1; only used of by_microtrip is True, blends
            between microtrip average speed and microtrip average speed when moving. Must be
            between 0 and 1 inclusive
        """
        params = self.sim_params
        params.idm_allow = True
        if not by_microtrip:
            if self.cyc0.len > 0 and self.cyc0.time_s[-1] > 0.0:
                params.idm_v_desired_m_per_s = self.cyc0.dist_m.sum() / self.cyc0.time_s[-1]
            else:
                params.idm_v_desired_m_per_s = 0.0
        else:
            if blend_factor > 1.0 or blend_factor < 0.0:
                raise TypeError(f"blend_factor must be between 0 and 1 but got {blend_factor}")
            if min_target_speed_m_per_s < 0.0:
                raise TypeError(f"min_target_speed_m_per_s must be >= 0 but got {min_target_speed_m_per_s}")
            params.idm_v_desired_in_m_per_s_by_distance_m = cycle.create_dist_and_target_speeds_by_microtrip(
                self.cyc0, blend_factor=blend_factor, min_target_speed_mps=min_target_speed_m_per_s
            )
        self.sim_params = params
        # Extend the duration of the base cycle
        if extend_fraction < 0.0:
            raise TypeError(f"extend_fraction must be >= 0.0 but got {extend_fraction}")
        if extend_fraction > 0.0:
            self.cyc0 = cycle.extend_cycle(self.cyc0, time_fraction=extend_fraction)
            self.cyc = self.cyc0.copy()
    
    def _next_speed_by_idm(self, i, a_m_per_s2, b_m_per_s2, dt_headway_s, s0_m, v_desired_m_per_s, delta=4.0):
        """
        Calculate the next speed by the Intelligent Driver Model
        - i: int, the index
        - a_m_per_s2: number, max acceleration (m/s2)
        - b_m_per_s2: number, max deceleration (m/s2)
        - dt_headway_s: number, the headway between us and the lead vehicle in seconds
        - s0_m: number, the initial gap between us and the lead vehicle in meters
        - v_desired_m_per_s: number, the desired speed in (m/s)
        - delta: number, a shape parameter; typical value is 4.0
        RETURN: number, the next speed (m/s)

        REFERENCE:
        Treiber, Martin and Kesting, Arne. 2013. "Chapter 11: Car-Following Models Based on Driving Strategies".
            Traffic Flow Dynamics: Data, Models and Simulation. Springer-Verlag. Springer, Berlin, Heidelberg.
            DOI: https://doi.org/10.1007/978-3-642-32460-4.
        """
        if (v_desired_m_per_s <= 0.0):
            return 0.0
        a_m_per_s2 = abs(a_m_per_s2)  # acceleration (m/s2)
        b_m_per_s2 = abs(b_m_per_s2)  # deceleration (m/s2)
        dt_headway_s = max(0.0, dt_headway_s)
        # we assume the vehicles start out a "minimum gap" apart
        s0_m = max(0.0, s0_m)
        # DERIVED VALUES
        sqrt_ab = (a_m_per_s2 * b_m_per_s2)**0.5
        v0_m__s = self.mps_ach[i-1]
        v0_lead_m_per_s = self.cyc0.mps[i-1]
        dv0_m_per_s = v0_m__s - v0_lead_m_per_s
        d0_lead_m = self.cyc0_cache.trapz_distances_m[max(i-1, 0)] + s0_m
        d0_m = cycle.trapz_step_start_distance(self.cyc, i)
        s_m = max(d0_lead_m - d0_m, 0.01)
        # IDM EQUATIONS
        s_target_m = s0_m + \
            max(0.0, (v0_m__s * dt_headway_s) +
                ((v0_m__s * dv0_m_per_s)/(2.0 * sqrt_ab)))
        accel_target_m_per_s2 = a_m_per_s2 * \
            (1.0 - ((v0_m__s / v_desired_m_per_s) ** delta) - ((s_target_m / s_m)**2))
        return max(v0_m__s + (accel_target_m_per_s2 * self.cyc.dt_s_at_i(i)), 0.0)

    def _set_speed_for_target_gap_using_idm(self, i):
        """
        Set gap
        - i: non-negative integer, the step index
        RETURN: None
        EFFECTS:
        - sets the next speed (m/s)
        EQUATION:
        parameters:
            - v_desired: the desired speed (m/s)
            - delta: number, typical value is 4.0
            - a: max acceleration, (m/s2)
            - b: max deceleration, (m/s2)
        s = d_lead - d
        dv/dt = a * (1 - (v/v_desired)**delta - (s_desired(v,v-v_lead)/s)**2)
        s_desired(v, dv) = s0 + max(0, v*dt_headway + (v * dv)/(2.0 * sqrt(a*b)))
        """
        if self.idm_target_speed_m_per_s[i] > 0:
            v_desired_m_per_s = self.idm_target_speed_m_per_s[i]
        else:
            v_desired_m_per_s = self.cyc0.mps.max()
        self.cyc.mps[i] = self._next_speed_by_idm(
            i,
            a_m_per_s2=self.sim_params.idm_accel_m_per_s2,
            b_m_per_s2=self.sim_params.idm_decel_m_per_s2,
            dt_headway_s=self.sim_params.idm_dt_headway_s,
            s0_m=self.sim_params.idm_minimum_gap_m,
            v_desired_m_per_s=v_desired_m_per_s,
            delta=self.sim_params.idm_delta,
        )

    def _set_speed_for_target_gap(self, i):
        """
        - i: non-negative integer, the step index
        RETURN: None
        EFFECTS:
        - sets the next speed (m/s)
        """
        self._set_speed_for_target_gap_using_idm(i)
    
    def _estimate_grade_for_step(self, i: int) -> float:
        """
        Provides a quick estimate for grade based only on the distance traveled
        at the start of the current step. If the grade is constant over the
        step, this is both quick and accurate.

        NOTE:
            If not allowing coasting (i.e., sim_params.coast_allow == False)
            and not allowing IDM/following (i.e., self.sim_params.idm_allow == False)
            then returns self.cyc.grade[i]
        """
        if self.cyc0_cache.grade_all_zero:
            return 0.0
        if not self.sim_params.coast_allow and not self.sim_params.idm_allow:
            return self.cyc.grade[i]
        return self.cyc0_cache.interp_grade(cycle.trapz_step_start_distance(self.cyc, i))
    
    def _lookup_grade_for_step(self, i: int, mps_ach: Optional[float] = None) -> float:
        """
        For situations where cyc can deviate from cyc0, this method
        looks up and accurately interpolates what the average grade over
        the step should be.

        If mps_ach is not None, the mps_ach value is used to predict the
        distance traveled over the step.

        NOTE:
            If not allowing coasting (i.e., sim_params.coast_allow == False)
            and not allowing IDM/following (i.e., self.sim_params.idm_allow == False)
            then returns self.cyc.grade[i]
        """
        if self.cyc0_cache.grade_all_zero:
            return 0.0
        if not self.sim_params.coast_allow and not self.sim_params.idm_allow:
            return self.cyc.grade[i]
        if mps_ach is not None:
            return self.cyc0.average_grade_over_range(
                cycle.trapz_step_start_distance(self.cyc, i),
                0.5 * (mps_ach + self.mps_ach[i - 1]) * self.cyc.dt_s_at_i(i),
                cache=self.cyc0_cache)
        return self.cyc0.average_grade_over_range(
                cycle.trapz_step_start_distance(self.cyc, i),
                cycle.trapz_distance_for_step(self.cyc, i),
                cache=self.cyc0_cache)

    def sim_drive_step(self):
        """
        Step through 1 time step.
        TODO: create self.set_speed_for_target_gap(self.i):
        TODO: consider implementing for battery SOC dependence
        """
        if self.sim_params.idm_allow:
            if self.sim_params.idm_v_desired_in_m_per_s_by_distance_m is not None:
                found_v_target = self.sim_params.idm_v_desired_in_m_per_s_by_distance_m[0][1]
                current_d = self.cyc.dist_m[:self.i].sum()
                for d, v_target in self.sim_params.idm_v_desired_in_m_per_s_by_distance_m:
                    if current_d >= d:
                        found_v_target = v_target
                    else:
                        break
                self.idm_target_speed_m_per_s[self.i] = found_v_target
            else:
                self.idm_target_speed_m_per_s[self.i] = self.sim_params.idm_v_desired_m_per_s
            self._set_speed_for_target_gap(self.i)
        if self.sim_params.coast_allow:
            self._set_coast_speed(self.i)
        self.solve_step(self.i)
        if self.sim_params.missed_trace_correction and (self.cyc0.dist_m[:self.i].sum() > 0):
            self.set_time_dilation(self.i)
        # TODO: shouldn't this below always get set whether we're coasting or following or not?
        if self.sim_params.coast_allow or self.sim_params.idm_allow:
            self.cyc.mps[self.i] = self.mps_ach[self.i]
            self.cyc.grade[self.i] = self._lookup_grade_for_step(self.i)

        self.i += 1  # increment time step counter

    def solve_step(self, i):
        """Perform all the calculations to solve 1 time step."""
        self.set_misc_calcs(i)
        self.set_comp_lims(i)
        self.set_power_calcs(i)
        self.set_ach_speed(i)
        self.set_hybrid_cont_calcs(i)
        # can probably be *mostly* done with list comprehension in post processing
        self.set_fc_forced_state(i)
        self.set_hybrid_cont_decisions(i)
        self.set_fc_power(i)

    def set_misc_calcs(self, i):
        """Sets misc. calculations at time step 'i'
        Arguments:
        ----------
        i: index of time step"""
        # if cycle iteration is used, auxInKw must be re-zeroed to trigger the below if statement
        if (self.aux_in_kw[i:] == 0).all():
            # if all elements after i-1 are zero, trigger default behavior; otherwise, use override value
            if self.veh.no_elec_aux:
                self.aux_in_kw[i] = self.veh.aux_kw / self.veh.alt_eff
            else:
                self.aux_in_kw[i] = self.veh.aux_kw
        # Is SOC below min threshold?
        if self.soc[i-1] < (self.veh.min_soc + self.veh.perc_high_acc_buf):
            self.reached_buff[i] = False
        else:
            self.reached_buff[i] = True

        # Does the engine need to be on for low SOC or high acceleration
        if self.soc[i-1] < self.veh.min_soc or (self.high_acc_fc_on_tag[i-1] and not(self.reached_buff[i])):
            self.high_acc_fc_on_tag[i] = True
        else:
            self.high_acc_fc_on_tag[i] = False
        self.max_trac_mps[i] = self.mps_ach[i-1] + \
            (self.veh.max_trac_mps2 * self.cyc.dt_s_at_i(i))

    def set_comp_lims(self, i):
        """
        Sets component limits for time step 'i'
        Arguments
        ------------
        i: index of time step
        init_soc: initial SOC for electrified vehicles
        """

        # max fuel storage power output
        self.cur_max_fs_kw_out[i] = min(
            self.veh.fs_max_kw,
            self.fs_kw_out_ach[i-1] + (
                (self.veh.fs_max_kw / self.veh.fs_secs_to_peak_pwr) * (self.cyc.dt_s_at_i(i))))
        # maximum fuel storage power output rate of change
        self.fc_trans_lim_kw[i] = self.fc_kw_out_ach[i-1] + (
            self.veh.fc_max_kw / self.veh.fc_sec_to_peak_pwr * self.cyc.dt_s_at_i(i)
        )

        self.fc_max_kw_in[i] = min(
            self.cur_max_fs_kw_out[i], self.veh.fs_max_kw)
        self.fc_fs_lim_kw[i] = self.veh.fc_max_kw
        self.cur_max_fc_kw_out[i] = min(
            self.veh.fc_max_kw, self.fc_fs_lim_kw[i], self.fc_trans_lim_kw[i])

        if self.veh.ess_max_kwh == 0 or self.soc[i-1] < self.veh.min_soc:
            self.ess_cap_lim_dischg_kw[i] = 0.0

        else:
            self.ess_cap_lim_dischg_kw[i] = self.veh.ess_max_kwh * np.sqrt(self.veh.ess_round_trip_eff) * 3.6e3 * (
                self.soc[i-1] - self.veh.min_soc) / self.cyc.dt_s_at_i(i)
        self.cur_ess_max_kw_out[i] = min(
            self.veh.ess_max_kw, self.ess_cap_lim_dischg_kw[i])

        if self.veh.ess_max_kwh == 0 or self.veh.ess_max_kw == 0:
            self.ess_cap_lim_chg_kw[i] = 0

        else:
            self.ess_cap_lim_chg_kw[i] = max(
                (self.veh.max_soc - self.soc[i-1]) * self.veh.ess_max_kwh * 1 / np.sqrt(self.veh.ess_round_trip_eff) /
                (self.cyc.dt_s_at_i(i) * 1 / 3.6e3),
                0
            )

        self.cur_max_ess_chg_kw[i] = min(
            self.ess_cap_lim_chg_kw[i], self.veh.ess_max_kw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if self.veh.fc_eff_type == H2FC:
            self.cur_max_elec_kw[i] = self.cur_max_fc_kw_out[i] + \
                self.cur_max_roadway_chg_kw[i] + \
                self.cur_ess_max_kw_out[i] - self.aux_in_kw[i]

        else:
            self.cur_max_elec_kw[i] = self.cur_max_roadway_chg_kw[i] + \
                self.cur_ess_max_kw_out[i] - self.aux_in_kw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        self.cur_max_avail_elec_kw[i] = min(
            self.cur_max_elec_kw[i], self.veh.mc_max_elec_in_kw)

        if self.cur_max_elec_kw[i] > 0:
            # limit power going into e-machine controller to
            if self.cur_max_avail_elec_kw[i] == max(self.veh.mc_kw_in_array):
                self.mc_elec_in_lim_kw[i] = min(
                    self.veh.mc_kw_out_array[-1], self.veh.mc_max_kw)
            else:
                self.mc_elec_in_lim_kw[i] = min(
                    self.veh.mc_kw_out_array[
                        np.argmax(self.veh.mc_kw_in_array > min(
                            max(self.veh.mc_kw_in_array) - 0.01,
                            self.cur_max_avail_elec_kw[i]
                        )) - 1],
                    self.veh.mc_max_kw)
        else:
            self.mc_elec_in_lim_kw[i] = 0.0

        # Motor transient power limit
        self.mc_transi_lim_kw[i] = abs(
            self.mc_mech_kw_out_ach[i-1]) + self.veh.mc_max_kw / self.veh.mc_sec_to_peak_pwr * self.cyc.dt_s_at_i(i)

        self.cur_max_mc_kw_out[i] = max(
            min(
                self.mc_elec_in_lim_kw[i],
                self.mc_transi_lim_kw[i],
                np.float64(0 if self.veh.stop_start else 1) * self.veh.mc_max_kw),
            -self.veh.mc_max_kw
        )

        if self.cur_max_mc_kw_out[i] == 0:
            self.cur_max_mc_elec_kw_in[i] = 0
        else:
            if self.cur_max_mc_kw_out[i] == self.veh.mc_max_kw:
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i] / \
                    self.veh.mc_full_eff_array[-1]
            else:
                self.cur_max_mc_elec_kw_in[i] = (self.cur_max_mc_kw_out[i] / self.veh.mc_full_eff_array[
                    max(1, np.argmax(
                        self.veh.mc_kw_out_array > min(
                            self.veh.mc_max_kw - 0.01, self.cur_max_mc_kw_out[i])
                    ) - 1
                    )
                ]
                )

        if self.veh.mc_max_kw == 0:
            self.ess_lim_mc_regen_perc_kw[i] = 0.0

        else:
            self.ess_lim_mc_regen_perc_kw[i] = min(
                (self.cur_max_ess_chg_kw[i] + self.aux_in_kw[i]) / self.veh.mc_max_kw, 1)
        if self.cur_max_ess_chg_kw[i] == 0:
            self.ess_lim_mc_regen_kw[i] = 0.0

        else:
            if self.veh.mc_max_kw == self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i]:
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.mc_max_kw, self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[-1])
            else:
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.mc_max_kw,
                    self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[
                        max(1,
                            np.argmax(
                                self.veh.mc_kw_out_array > min(
                                    self.veh.mc_max_kw - 0.01,
                                    self.cur_max_ess_chg_kw[i] -
                                    self.cur_max_roadway_chg_kw[i]
                                )
                            ) - 1
                            )
                    ]
                )

        self.cur_max_mech_mc_kw_in[i] = min(
            self.ess_lim_mc_regen_kw[i], self.veh.mc_max_kw)
        self.cur_max_trac_kw[i] = (
            self.veh.wheel_coef_of_fric * self.veh.drive_axle_weight_frac *
            self.veh.veh_kg * self.props.a_grav_mps2
            / (1 + self.veh.veh_cg_m * self.veh.wheel_coef_of_fric / self.veh.wheel_base_m
            ) / 1_000 * self.max_trac_mps[i]
        )

        if self.veh.fc_eff_type == H2FC:
            if self.veh.no_elec_sys or self.veh.no_elec_aux or self.high_acc_fc_on_tag[i]:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] -
                     self.aux_in_kw[i]) * self.veh.trans_eff,
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )

            else:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] -
                     min(self.cur_max_elec_kw[i], 0)) * self.veh.trans_eff,
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )

        else:
            if self.veh.no_elec_sys or self.veh.no_elec_aux or self.high_acc_fc_on_tag[i]:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i] -
                     self.aux_in_kw[i]) * self.veh.trans_eff,
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )

            else:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i] -
                     min(self.cur_max_elec_kw[i], 0)) * self.veh.trans_eff,
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )
        if self.impose_coast[i]:
            self.cur_max_trans_kw_out[i] = 0.0

    def set_power_calcs(self, i):
        """
        Calculate power requirements to meet cycle and determine if
        cycle can be met.  
        Arguments
        ------------
        i: index of time step
        """

        if self.newton_iters[i] > 0:
            mps_ach = self.mps_ach[i]
        else:
            mps_ach = self.cyc.mps[i]

        # TODO: use of self.cyc.mph[i] in regenContrLimKwPerc[i] calculation seems wrong. Shouldn't it be mpsAch or self.cyc0.mph[i]?

        grade = self._lookup_grade_for_step(i, mps_ach=mps_ach)
        self.drag_kw[i] = 0.5 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * self.veh.frontal_area_m2 * (
            (self.mps_ach[i-1] + mps_ach) / 2.0) ** 3 / 1_000
        self.accel_kw[i] = self.veh.veh_kg / \
            (2.0 * self.cyc.dt_s_at_i(i)) * \
            (mps_ach ** 2 - self.mps_ach[i-1] ** 2) / 1_000
        self.ascent_kw[i] = self.props.a_grav_mps2 * np.sin(np.arctan(
            grade)) * self.veh.veh_kg * ((self.mps_ach[i-1] + mps_ach) / 2.0) / 1_000
        self.cyc_trac_kw_req[i] = self.drag_kw[i] + \
            self.accel_kw[i] + self.ascent_kw[i]
        self.spare_trac_kw[i] = self.cur_max_trac_kw[i] - \
            self.cyc_trac_kw_req[i]
        self.rr_kw[i] = self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef * np.cos(
            np.arctan(grade)) * (self.mps_ach[i-1] + mps_ach) / 2.0 / 1_000
        self.cyc_whl_rad_per_sec[i] = mps_ach / self.veh.wheel_radius_m
        self.cyc_tire_inertia_kw[i] = (
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * self.cyc_whl_rad_per_sec[i] ** 2.0 / self.cyc.dt_s_at_i(i) -
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels *
            (self.mps_ach[i-1] /
             self.veh.wheel_radius_m) ** 2.0 / self.cyc.dt_s_at_i(i)
        ) / 1_000

        self.cyc_whl_kw_req[i] = self.cyc_trac_kw_req[i] + \
            self.rr_kw[i] + self.cyc_tire_inertia_kw[i]
        # TODO: check below, should we be using self.cyc.mph[i] OR should it be mpsAch converted to mph?
        self.regen_contrl_lim_kw_perc[i] = self.veh.max_regen / (1 + self.veh.regen_a * np.exp(-self.veh.regen_b * (
            (self.cyc.mph[i] + self.mps_ach[i-1] * params.MPH_PER_MPS) / 2.0 + 1.0)))
        self.cyc_regen_brake_kw[i] = max(min(
            self.cur_max_mech_mc_kw_in[i] * self.veh.trans_eff,
            self.regen_contrl_lim_kw_perc[i] * -self.cyc_whl_kw_req[i]),
            0
        )
        self.cyc_fric_brake_kw[i] = - \
            min(self.cyc_regen_brake_kw[i] + self.cyc_whl_kw_req[i], 0)
        self.cyc_trans_kw_out_req[i] = self.cyc_whl_kw_req[i] + \
            self.cyc_fric_brake_kw[i]

        if self.cyc_trans_kw_out_req[i] <= self.cur_max_trans_kw_out[i]:
            self.cyc_met[i] = True
            self.trans_kw_out_ach[i] = self.cyc_trans_kw_out_req[i]

        else:
            self.cyc_met[i] = False
            self.trans_kw_out_ach[i] = self.cur_max_trans_kw_out[i]

        if self.trans_kw_out_ach[i] > 0:
            self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] / \
                self.veh.trans_eff
        else:
            self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] * \
                self.veh.trans_eff

        if self.cyc_met[i]:

            if self.veh.fc_eff_type == H2FC:
                self.min_mc_kw_2help_fc[i] = max(
                    self.trans_kw_in_ach[i], -self.cur_max_mech_mc_kw_in[i])

            else:
                self.min_mc_kw_2help_fc[i] = max(
                    self.trans_kw_in_ach[i] - self.cur_max_fc_kw_out[i], -self.cur_max_mech_mc_kw_in[i])
        else:
            self.min_mc_kw_2help_fc[i] = max(
                self.cur_max_mc_kw_out[i], -self.cur_max_mech_mc_kw_in[i])

    def set_ach_speed(self, i):
        """
        Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
        Arguments
        ------------
        i: index of time step
        """
        # Cycle is met
        if self.cyc_met[i]:
            self.mps_ach[i] = self.cyc.mps[i]

        #Cycle is not met
        else:
            def newton_mps_estimate(totals):
                t3 = totals[0]
                t2 = totals[1]
                t1 = totals[2]
                t0 = totals[3]
                xs = []
                ys = []
                ms = []
                bs = []
                # initial guess
                xi = max(1.0, self.mps_ach[i-1])
                # stop criteria
                max_iter = self.sim_params.newton_max_iter
                xtol = self.sim_params.newton_xtol
                # solver gain
                g = self.sim_params.newton_gain
                yi = t3 * xi ** 3 + t2 * xi ** 2 + t1 * xi + t0
                mi = 3 * t3 * xi ** 2 + 2 * t2 * xi + t1
                bi = yi - xi * mi
                xs.append(xi)
                ys.append(yi)
                ms.append(mi)
                bs.append(bi)
                iterate = 1
                converged = False
                while iterate < max_iter and not(converged):
                    xi = xs[-1] * (1 - g) - g * bs[-1] / ms[-1]
                    yi = t3 * xi ** 3 + t2 * xi ** 2 + t1 * xi + t0
                    mi = 3 * t3 * xi ** 2 + 2 * t2 * xi + t1
                    bi = yi - xi * mi
                    xs.append(xi)
                    ys.append(yi)
                    ms.append(mi)
                    bs.append(bi)
                    converged = abs((xs[-1] - xs[-2]) / xs[-2]) < xtol
                    iterate += 1

                self.newton_iters[i] = iterate

                _ys = [abs(y) for y in ys]
                return max(xs[_ys.index(min(_ys))], 0.0)

            grade_estimate = self._estimate_grade_for_step(i)
            grade_tol = 1e-6
            grade_diff = grade_tol + 1.0
            max_grade_iter = 3
            grade_iter = 0
            while grade_diff > grade_tol and grade_iter < max_grade_iter:
                grade_iter += 1
                grade = grade_estimate

                drag3 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * \
                    self.veh.drag_coef * self.veh.frontal_area_m2
                accel2 = 0.5 * self.veh.veh_kg / self.cyc.dt_s_at_i(i)
                drag2 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * \
                    self.veh.drag_coef * \
                    self.veh.frontal_area_m2 * self.mps_ach[i-1]
                wheel2 = 0.5 * self.veh.wheel_inertia_kg_m2 * \
                    self.veh.num_wheels / \
                    (self.cyc.dt_s_at_i(i) * self.veh.wheel_radius_m ** 2)
                drag1 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * \
                    self.veh.frontal_area_m2 * self.mps_ach[i-1] ** 2
                roll1 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef \
                    * np.cos(np.arctan(grade))
                ascent1 = 0.5 * self.props.a_grav_mps2 * \
                    np.sin(np.arctan(grade)) * self.veh.veh_kg
                accel0 = -0.5 * self.veh.veh_kg * \
                    self.mps_ach[i-1] ** 2 / self.cyc.dt_s_at_i(i)
                drag0 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * \
                    self.veh.frontal_area_m2 * self.mps_ach[i-1] ** 3
                roll0 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * \
                    self.veh.wheel_rr_coef * np.cos(np.arctan(grade)) \
                    * self.mps_ach[i-1]
                ascent0 = 0.5 * self.props.a_grav_mps2 * np.sin(np.arctan(grade)) \
                    * self.veh.veh_kg * self.mps_ach[i-1]
                wheel0 = -0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * \
                    self.mps_ach[i-1] ** 2 / \
                    (self.cyc.dt_s_at_i(i) * self.veh.wheel_radius_m ** 2)

                total3 = drag3 / 1_000
                total2 = (accel2 + drag2 + wheel2) / 1_000
                total1 = (drag1 + roll1 + ascent1) / 1_000
                total0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) / \
                    1_000 - self.cur_max_trans_kw_out[i]

                total = np.array([total3, total2, total1, total0])
                self.mps_ach[i] = newton_mps_estimate(total)
                grade_estimate = self._lookup_grade_for_step(i, mps_ach=self.mps_ach[i])
                grade_diff = np.abs(grade - grade_estimate)
            self.set_power_calcs(i)

        self.mph_ach[i] = self.mps_ach[i] * params.MPH_PER_MPS
        self.dist_m[i] = self.mps_ach[i] * self.cyc.dt_s_at_i(i)
        self.dist_mi[i] = self.dist_m[i] * (1.0 / params.M_PER_MI)

    def set_hybrid_cont_calcs(self, i):
        """Hybrid control calculations.  
        Arguments
        ------------
        i: index of time step"""

        if self.veh.no_elec_sys:
            self.regen_buff_soc[i] = 0

        elif self.veh.charging_on:
            self.regen_buff_soc[i] = max(
                self.veh.max_soc - (self.veh.max_regen_kwh / self.veh.ess_max_kwh), (self.veh.max_soc + self.veh.min_soc) / 2)

        else:
            self.regen_buff_soc[i] = max(
                (self.veh.ess_max_kwh * self.veh.max_soc -
                    0.5 * self.veh.veh_kg * (self.cyc.mps[i] ** 2) * (1.0 / 1_000) * (1.0 / 3_600) *
                    self.veh.mc_peak_eff * self.veh.max_regen) / self.veh.ess_max_kwh,
                self.veh.min_soc
            )

            self.ess_regen_buff_dischg_kw[i] = min(self.cur_ess_max_kw_out[i], max(
                0, (self.soc[i-1] - self.regen_buff_soc[i]) * self.veh.ess_max_kwh * 3_600 / self.cyc.dt_s_at_i(i)))

            self.max_ess_regen_buff_chg_kw[i] = min(max(
                0,
                (self.regen_buff_soc[i] - self.soc[i-1]) * self.veh.ess_max_kwh * 3.6e3 / self.cyc.dt_s_at_i(i)),
                self.cur_max_ess_chg_kw[i]
            )

        if self.veh.no_elec_sys:
            self.accel_buff_soc[i] = 0

        else:
            self.accel_buff_soc[i] = min(
                max(
                    ((self.veh.max_accel_buffer_mph / params.MPH_PER_MPS) ** 2 - self.cyc.mps[i] ** 2) /
                    (self.veh.max_accel_buffer_mph / params.MPH_PER_MPS) ** 2 * min(
                        self.veh.max_accel_buffer_perc_of_useable_soc *
                        (self.veh.max_soc - self.veh.min_soc),
                        self.veh.max_regen_kwh / self.veh.ess_max_kwh
                    ) * self.veh.ess_max_kwh / self.veh.ess_max_kwh + self.veh.min_soc,
                    self.veh.min_soc
                ),
                self.veh.max_soc
            )

            self.ess_accel_buff_chg_kw[i] = max(
                0, (self.accel_buff_soc[i] - self.soc[i-1]) * self.veh.ess_max_kwh * 3.6e3 / self.cyc.dt_s_at_i(i))
            self.max_ess_accell_buff_dischg_kw[i] = min(
                max(
                    0,
                    (self.soc[i-1] - self.accel_buff_soc[i]) * self.veh.ess_max_kwh * 3_600 / self.cyc.dt_s_at_i(i)),
                self.cur_ess_max_kw_out[i]
            )

        if self.regen_buff_soc[i] < self.accel_buff_soc[i]:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(
                    (self.soc[i-1] - (self.regen_buff_soc[i] + self.accel_buff_soc[i]
                                      ) / 2) * self.veh.ess_max_kwh * 3.6e3 / self.cyc.dt_s_at_i(i),
                    self.cur_ess_max_kw_out[i]
                ),
                -self.cur_max_ess_chg_kw[i]
            )

        elif self.soc[i-1] > self.regen_buff_soc[i]:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(
                    self.ess_regen_buff_dischg_kw[i],
                    self.cur_ess_max_kw_out[i]),
                -self.cur_max_ess_chg_kw[i]
            )

        elif self.soc[i-1] < self.accel_buff_soc[i]:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(-1.0 * self.ess_accel_buff_chg_kw[i], self.cur_ess_max_kw_out[i]), -self.cur_max_ess_chg_kw[i])

        else:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(0, self.cur_ess_max_kw_out[i]), -self.cur_max_ess_chg_kw[i])

        self.fc_kw_gap_fr_eff[i] = abs(
            self.trans_kw_out_ach[i] - self.veh.max_fc_eff_kw)

        if self.veh.no_elec_sys:
            self.mc_elec_in_kw_for_max_fc_eff[i] = 0

        elif self.trans_kw_out_ach[i] < self.veh.max_fc_eff_kw:
            if self.fc_kw_gap_fr_eff[i] == self.veh.mc_max_kw:
                self.mc_elec_in_kw_for_max_fc_eff[i] = - \
                    self.fc_kw_gap_fr_eff[i] / self.veh.mc_full_eff_array[-1]
            else:
                self.mc_elec_in_kw_for_max_fc_eff[i] = (-self.fc_kw_gap_fr_eff[i] /
                                                        self.veh.mc_full_eff_array[max(1,
                                                                                       np.argmax(self.veh.mc_kw_out_array > min(self.veh.mc_max_kw - 0.01, self.fc_kw_gap_fr_eff[i])) - 1)]
                                                        )

        else:
            if self.fc_kw_gap_fr_eff[i] == self.veh.mc_max_kw:
                self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[len(
                    self.veh.mc_kw_in_array) - 1]
            else:
                self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[np.argmax(
                    self.veh.mc_kw_out_array > min(self.veh.mc_max_kw - 0.01, self.fc_kw_gap_fr_eff[i])) - 1]

        if self.veh.no_elec_sys:
            self.elec_kw_req_4ae[i] = 0

        elif self.trans_kw_in_ach[i] > 0:
            if self.trans_kw_in_ach[i] == self.veh.mc_max_kw:
                self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i] / \
                    self.veh.mc_full_eff_array[-1] + self.aux_in_kw[i]
            else:
                self.elec_kw_req_4ae[i] = (self.trans_kw_in_ach[i] /
                                           self.veh.mc_full_eff_array[max(1, np.argmax(
                                               self.veh.mc_kw_out_array > min(self.veh.mc_max_kw - 0.01, self.trans_kw_in_ach[i])) - 1)] + self.aux_in_kw[i]
                                           )

        else:
            self.elec_kw_req_4ae[i] = 0

        self.prev_fc_time_on[i] = self.fc_time_on[i-1]

        # some conditions in the following if statement have a buffer of 1e-6 to prevent false positives/negatives because these have been encountered in practice.
        if self.veh.fc_max_kw == 0:
            self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i-1] and  \
                (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i] and \
                (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i]
                 or self.veh.fc_max_kw == 0)

        else:
            self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i-1] and \
                (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i] and \
                (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i] or self.veh.fc_max_kw == 0) \
                and ((self.cyc.mph[i] - 1e-6) <= self.veh.mph_fc_on or self.veh.charging_on) and \
                self.elec_kw_req_4ae[i] <= self.veh.kw_demand_fc_on

        if self.can_pwr_all_elec[i]:

            if self.trans_kw_in_ach[i] < self.aux_in_kw[i]:
                self.desired_ess_kw_out_for_ae[i] = self.aux_in_kw[i] + \
                    self.trans_kw_in_ach[i]

            elif self.regen_buff_soc[i] < self.accel_buff_soc[i]:
                self.desired_ess_kw_out_for_ae[i] = self.ess_accel_regen_dischg_kw[i]

            elif self.soc[i-1] > self.regen_buff_soc[i]:
                self.desired_ess_kw_out_for_ae[i] = self.ess_regen_buff_dischg_kw[i]

            elif self.soc[i-1] < self.accel_buff_soc[i]:
                self.desired_ess_kw_out_for_ae[i] = - \
                    self.ess_accel_buff_chg_kw[i]

            else:
                self.desired_ess_kw_out_for_ae[i] = self.trans_kw_in_ach[i] + \
                    self.aux_in_kw[i] - self.cur_max_roadway_chg_kw[i]

        else:
            self.desired_ess_kw_out_for_ae[i] = 0

        if self.can_pwr_all_elec[i]:
            self.ess_ae_kw_out[i] = max(
                -self.cur_max_ess_chg_kw[i],
                -self.max_ess_regen_buff_chg_kw[i],
                min(0, self.cur_max_roadway_chg_kw[i] -
                    self.trans_kw_in_ach[i] + self.aux_in_kw[i]),
                min(self.cur_ess_max_kw_out[i],
                    self.desired_ess_kw_out_for_ae[i])
            )

        else:
            self.ess_ae_kw_out[i] = 0

        self.er_ae_kw_out[i] = min(
            max(0, self.trans_kw_in_ach[i] +
                self.aux_in_kw[i] - self.ess_ae_kw_out[i]),
            self.cur_max_roadway_chg_kw[i])

    def set_fc_forced_state(self, i):
        """
        Calculate control variables related to engine on/off state
        Arguments       
        ------------
        i: index of time step
        """
        # force fuel converter on if it was on in the previous time step, but only if fc
        # has not been on longer than minFcTimeOn
        if self.prev_fc_time_on[i] > 0 and self.prev_fc_time_on[i] < self.veh.min_fc_time_on - self.cyc.dt_s_at_i(i):
            self.fc_forced_on[i] = True
        else:
            self.fc_forced_on[i] = False

        if not(self.fc_forced_on[i]) or not(self.can_pwr_all_elec[i]):
            self.fc_forced_state[i] = 1
            self.mc_mech_kw_4forced_fc[i] = 0

        elif self.trans_kw_in_ach[i] < 0:
            self.fc_forced_state[i] = 2
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i]

        elif self.veh.max_fc_eff_kw == self.trans_kw_in_ach[i]:
            self.fc_forced_state[i] = 3
            self.mc_mech_kw_4forced_fc[i] = 0

        elif self.veh.idle_fc_kw > self.trans_kw_in_ach[i] and self.accel_kw[i] >= 0:
            self.fc_forced_state[i] = 4
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - \
                self.veh.idle_fc_kw

        elif self.veh.max_fc_eff_kw > self.trans_kw_in_ach[i]:
            self.fc_forced_state[i] = 5
            self.mc_mech_kw_4forced_fc[i] = 0

        else:
            self.fc_forced_state[i] = 6
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - \
                self.veh.max_fc_eff_kw

    def set_hybrid_cont_decisions(self, i):
        """
        Hybrid control decisions.
        Arguments
        ------------
        i: index of time step
        """

        if (-self.mc_elec_in_kw_for_max_fc_eff[i] - self.cur_max_roadway_chg_kw[i]) > 0:
            self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i] -
                                              self.cur_max_roadway_chg_kw[i]) * self.veh.ess_dischg_to_fc_max_eff_perc

        else:
            self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i] -
                                              self.cur_max_roadway_chg_kw[i]) * self.veh.ess_chg_to_fc_max_eff_perc

        if self.accel_buff_soc[i] > self.regen_buff_soc[i]:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_ess_max_kw_out[i],
                self.veh.mc_max_elec_in_kw +
                self.aux_in_kw[i], self.cur_max_mc_elec_kw_in[i] +
                self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i], self.ess_accel_regen_dischg_kw[i]))

        elif self.ess_regen_buff_dischg_kw[i] > 0:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_ess_max_kw_out[i],
                self.veh.mc_max_elec_in_kw +
                self.aux_in_kw[i], self.cur_max_mc_elec_kw_in[i] +
                self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i],
                    min(self.ess_accel_regen_dischg_kw[i],
                        self.mc_elec_in_lim_kw[i] + self.aux_in_kw[i],
                        max(self.ess_regen_buff_dischg_kw[i],
                            self.ess_desired_kw_4fc_eff[i])
                        )
                    )
            )

        elif self.ess_accel_buff_chg_kw[i] > 0:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_ess_max_kw_out[i],
                self.veh.mc_max_elec_in_kw +
                self.aux_in_kw[i], self.cur_max_mc_elec_kw_in[i] +
                self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i],
                    max(-1 * self.max_ess_regen_buff_chg_kw[i],
                        min(-self.ess_accel_buff_chg_kw[i],
                            self.ess_desired_kw_4fc_eff[i])
                        )
                    )
            )

        elif self.ess_desired_kw_4fc_eff[i] > 0:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_ess_max_kw_out[i],
                self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i],
                    min(self.ess_desired_kw_4fc_eff[i],
                        self.max_ess_accell_buff_dischg_kw[i])
                    )
            )

        else:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_ess_max_kw_out[i],
                self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i],
                    max(self.ess_desired_kw_4fc_eff[i], -
                        self.max_ess_regen_buff_chg_kw[i])
                    )
            )

        self.er_kw_if_fc_req[i] = max(0,
                                      min(
                                          self.cur_max_roadway_chg_kw[i], self.cur_max_mech_mc_kw_in[i],
                                          self.ess_kw_if_fc_req[i] -
                                          self.mc_elec_in_lim_kw[i] +
                                          self.aux_in_kw[i]
                                      )
                                      )

        self.mc_elec_kw_in_if_fc_req[i] = self.ess_kw_if_fc_req[i] + \
            self.er_kw_if_fc_req[i] - self.aux_in_kw[i]

        if self.veh.no_elec_sys:
            self.mc_kw_if_fc_req[i] = 0

        elif self.mc_elec_kw_in_if_fc_req[i] == 0:
            self.mc_kw_if_fc_req[i] = 0

        elif self.mc_elec_kw_in_if_fc_req[i] > 0:

            if self.mc_elec_kw_in_if_fc_req[i] == max(self.veh.mc_kw_in_array):
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] * \
                    self.veh.mc_full_eff_array[-1]
            else:
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] * self.veh.mc_full_eff_array[
                    max(1, np.argmax(
                        self.veh.mc_kw_in_array > min(
                            max(self.veh.mc_kw_in_array) - 0.01, self.mc_elec_kw_in_if_fc_req[i])
                    ) - 1
                    )
                ]

        else:
            if self.mc_elec_kw_in_if_fc_req[i] * -1 == max(self.veh.mc_kw_in_array):
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] / \
                    self.veh.mc_full_eff_array[-1]
            else:
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] / (self.veh.mc_full_eff_array[
                    max(1, np.argmax(
                        self.veh.mc_kw_in_array > min(max(self.veh.mc_kw_in_array) - 0.01, self.mc_elec_kw_in_if_fc_req[i] * -1)) - 1
                        )
                ]
                )

        if self.veh.mc_max_kw == 0:
            self.mc_mech_kw_out_ach[i] = 0

        elif self.fc_forced_on[i] and self.can_pwr_all_elec[i] and (self.veh.veh_pt_type == HEV or
                                                                    self.veh.veh_pt_type == PHEV) and (self.veh.fc_eff_type != H2FC):
            self.mc_mech_kw_out_ach[i] = self.mc_mech_kw_4forced_fc[i]

        elif self.trans_kw_in_ach[i] <= 0:

            if self.veh.fc_eff_type != H2FC and self.veh.fc_max_kw > 0:
                if self.can_pwr_all_elec[i] == 1:
                    self.mc_mech_kw_out_ach[i] = - \
                        min(self.cur_max_mech_mc_kw_in[i], -
                            self.trans_kw_in_ach[i])
                else:
                    self.mc_mech_kw_out_ach[i] = min(
                        -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]),
                        max(-self.cur_max_fc_kw_out[i],
                            self.mc_kw_if_fc_req[i])
                    )
            else:
                self.mc_mech_kw_out_ach[i] = min(
                    -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]),
                    -self.trans_kw_in_ach[i]
                )

        elif self.can_pwr_all_elec[i] == 1:
            self.mc_mech_kw_out_ach[i] = self.trans_kw_in_ach[i]

        else:
            self.mc_mech_kw_out_ach[i] = max(
                self.min_mc_kw_2help_fc[i], self.mc_kw_if_fc_req[i])

        if self.mc_mech_kw_out_ach[i] == 0:
            self.mc_elec_kw_in_ach[i] = 0.0

        elif self.mc_mech_kw_out_ach[i] < 0:

            if self.mc_mech_kw_out_ach[i] * -1 == max(self.veh.mc_kw_in_array):
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] * \
                    self.veh.mc_full_eff_array[-1]
            else:
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] * self.veh.mc_full_eff_array[
                    max(1, np.argmax(self.veh.mc_kw_in_array > min(
                        max(self.veh.mc_kw_in_array) - 0.01,
                        self.mc_mech_kw_out_ach[i] * -1)) - 1
                        )
                ]

        else:
            if self.veh.mc_max_kw == self.mc_mech_kw_out_ach[i]:
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] / \
                    self.veh.mc_full_eff_array[-1]
            else:
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] / self.veh.mc_full_eff_array[
                    max(1, np.argmax(self.veh.mc_kw_out_array > min(
                        self.veh.mc_max_kw - 0.01,
                        self.mc_mech_kw_out_ach[i])) - 1
                        )
                ]

        if self.cur_max_roadway_chg_kw[i] == 0:
            self.roadway_chg_kw_out_ach[i] = 0

        elif self.veh.fc_eff_type == H2FC:
            self.roadway_chg_kw_out_ach[i] = max(
                0,
                self.mc_elec_kw_in_ach[i],
                self.max_ess_regen_buff_chg_kw[i],
                self.ess_regen_buff_dischg_kw[i],
                self.cur_max_roadway_chg_kw[i])

        elif self.can_pwr_all_elec[i] == 1:
            self.roadway_chg_kw_out_ach[i] = self.er_ae_kw_out[i]

        else:
            self.roadway_chg_kw_out_ach[i] = self.er_kw_if_fc_req[i]

        self.min_ess_kw_2help_fc[i] = self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - \
            self.cur_max_fc_kw_out[i] - self.roadway_chg_kw_out_ach[i]

        if self.veh.ess_max_kw == 0 or self.veh.ess_max_kwh == 0:
            self.ess_kw_out_ach[i] = 0

        elif self.veh.fc_eff_type == H2FC:

            if self.trans_kw_out_ach[i] >= 0:
                self.ess_kw_out_ach[i] = min(max(
                    self.min_ess_kw_2help_fc[i],
                    self.ess_desired_kw_4fc_eff[i],
                    self.ess_accel_regen_dischg_kw[i]),
                    self.cur_ess_max_kw_out[i],
                    self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] -
                    self.roadway_chg_kw_out_ach[i]
                )

            else:
                self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] + \
                    self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i]

        elif self.high_acc_fc_on_tag[i] or self.veh.no_elec_aux:
            self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] - \
                self.roadway_chg_kw_out_ach[i]

        else:
            self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] + \
                self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i]

        if self.veh.no_elec_sys:
            self.ess_cur_kwh[i] = 0

        elif self.ess_kw_out_ach[i] < 0:
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s_at_i(i) /\
                3.6e3 * np.sqrt(self.veh.ess_round_trip_eff)

        else:
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s_at_i(i) / \
                3.6e3 * (1 / np.sqrt(self.veh.ess_round_trip_eff))

        if self.veh.ess_max_kwh == 0:
            self.soc[i] = 0.0

        else:
            self.soc[i] = self.ess_cur_kwh[i] / self.veh.ess_max_kwh

        if self.can_pwr_all_elec[i] and not(self.fc_forced_on[i]) and self.fc_kw_out_ach[i] == 0.0:
            self.fc_time_on[i] = 0
        else:
            self.fc_time_on[i] = self.fc_time_on[i-1] + self.cyc.dt_s_at_i(i)

    def set_fc_power(self, i):
        """
        Sets fcKwOutAch and fcKwInAch.
        Arguments
        ------------
        i: index of time step
        """

        if self.veh.fc_max_kw == 0:
            self.fc_kw_out_ach[i] = 0

        elif self.veh.fc_eff_type == H2FC:
            self.fc_kw_out_ach[i] = min(
                self.cur_max_fc_kw_out[i],
                max(0,
                    self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] -
                    self.ess_kw_out_ach[i] - self.roadway_chg_kw_out_ach[i]
                    )
            )

        elif self.veh.no_elec_sys or self.veh.no_elec_aux or self.high_acc_fc_on_tag[i]:
            self.fc_kw_out_ach[i] = min(
                self.cur_max_fc_kw_out[i],
                max(
                    0,
                    self.trans_kw_in_ach[i] -
                    self.mc_mech_kw_out_ach[i] + self.aux_in_kw[i]
                )
            )

        else:
            self.fc_kw_out_ach[i] = min(self.cur_max_fc_kw_out[i], max(
                0, self.trans_kw_in_ach[i] - self.mc_mech_kw_out_ach[i]))

        if self.veh.fc_max_kw == 0:
            self.fc_kw_out_ach_pct[i] = 0
        else:
            self.fc_kw_out_ach_pct[i] = self.fc_kw_out_ach[i] / \
                self.veh.fc_max_kw

        if self.fc_kw_out_ach[i] == 0:
            self.fc_kw_in_ach[i] = 0
            self.fc_kw_out_ach_pct[i] = 0

        else:
            self.fc_kw_in_ach[i] = (
                self.fc_kw_out_ach[i] / (self.veh.fc_eff_array[np.argmax(
                    self.veh.fc_kw_out_array > min(self.fc_kw_out_ach[i], self.veh.fc_max_kw)) - 1])
                if self.veh.fc_eff_array[np.argmax(
                    self.veh.fc_kw_out_array > min(self.fc_kw_out_ach[i], self.veh.fc_max_kw)) - 1] != 0
                else 0)

        self.fs_kw_out_ach[i] = self.fc_kw_in_ach[i]

        self.fs_kwh_out_ach[i] = self.fs_kw_out_ach[i] * \
            self.cyc.dt_s_at_i(i) * (1 / 3.6e3)

    def set_time_dilation(self, i):
        trace_met = (
            ((abs(self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum()) / self.cyc0.dist_m[:i+1].sum()
              ) < self.sim_params.time_dilation_tol) or
            # if prescribed speed is zero, trace is met to avoid div-by-zero errors and other possible wackiness
            (self.cyc.mps[i] == 0)
        )

        if not(trace_met):
            self.trace_miss_iters[i] += 1

            # positive if behind trace
            d_short = [self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum()]
            t_dilation = [
                0.0,  # no time dilation initially
                min(max(
                    # initial guess, speed that needed to be achived per speed that was achieved
                    d_short[-1] / self.cyc0.dt_s_at_i(i) / self.mps_ach[i],
                    self.sim_params.min_time_dilation
                ),
                    self.sim_params.max_time_dilation
                )
            ]

            # add time dilation factor * step size to current and subsequent times
            self.cyc.time_s[i:] += self.cyc.dt_s_at_i(i) * t_dilation[-1]
            self.solve_step(i)
            trace_met = (
                # convergence criteria
                (abs(self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum()) /
                    self.cyc0.dist_m[:i+1].sum() < self.sim_params.time_dilation_tol) or
                # exceeding max time dilation
                (t_dilation[-1] >= self.sim_params.max_time_dilation) or
                # lower than min time dilation
                (t_dilation[-1] <= self.sim_params.min_time_dilation)
            )

        while not(trace_met):
            # iterate newton's method until time dilation has converged or other exit criteria trigger trace_met == True
            # distance shortfall [m]
            # correct time steps
            d_short.append(
                self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum())
            t_dilation.append(
                min(
                    max(
                        t_dilation[-1] - (t_dilation[-1] - t_dilation[-2]) /
                        (d_short[-1] - d_short[-2]) * d_short[-1],
                        self.sim_params.min_time_dilation,
                    ),
                    self.sim_params.max_time_dilation
                )
            )
            self.cyc.time_s[i:] += self.cyc.dt_s_at_i(i) * t_dilation[-1]

            self.solve_step(i)
            self.trace_miss_iters[i] += 1

            trace_met = (
                # convergence criteria
                (abs(self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum()) /
                    self.cyc0.dist_m[:i+1].sum() < self.sim_params.time_dilation_tol) or
                # max iterations
                (self.trace_miss_iters[i] >= self.sim_params.max_trace_miss_iters) or
                # exceeding max time dilation
                (t_dilation[-1] >= self.sim_params.max_time_dilation) or
                # lower than min time dilation
                (t_dilation[-1] <= self.sim_params.min_time_dilation)
            )

    def _calc_dvdd(self, v, grade):
        """
        Calculates the derivative dv/dd (change in speed by change in distance)
        - v: number, the speed at which to evaluate dv/dd (m/s)
        - grade: number, the road grade as a decimal fraction
        RETURN: number, the dv/dd for these conditions
        """
        if v <= 0.0:
            return 0.0
        atan_grade_sin = 0.0
        atan_grade_cos = 1.0
        if grade != 0.0:
            atan_grade = float(np.arctan(grade))
            atan_grade_sin = np.sin(atan_grade)
            atan_grade_cos = np.cos(atan_grade)
        g = self.props.a_grav_mps2
        M = self.veh.veh_kg
        rho_CDFA = self.props.air_density_kg_per_m3 * \
            self.veh.drag_coef * self.veh.frontal_area_m2
        rrc = self.veh.wheel_rr_coef
        return -1.0 * (
            (g/v) * (atan_grade_sin + rrc * atan_grade_cos)
            + (0.5 * rho_CDFA * (1.0/M) * v)
        )

    def _generate_coast_trajectory(self, i:int, modify_cycle:bool=False) -> float:
        """
        Generate a coast trajectory without actually modifying the cycle.
        This can be used to calculate the distance to stop via coast using
        actual time-stepping and dynamically changing grade.
        RETURN: float, the distance to stop via coast in meters
        NOTE: if not found, a value of -1.0 is returned
        """
        NOT_FOUND = -1.0
        v0 = self.mps_ach[i-1]
        v_brake = self.sim_params.coast_brake_start_speed_m_per_s
        a_brake = self.sim_params.coast_brake_accel_m_per_s2
        assert a_brake <= 0
        ds = self.cyc0_cache.trapz_distances_m
        gs = self.cyc0.grade
        d0 = cycle.trapz_step_start_distance(self.cyc, i)
        ds_mask = ds >= d0
        if not np.any(ds_mask):
            return 0.0
        distances_m = ds[ds_mask] - d0
        grade_by_distance = gs[ds_mask]
        # distance traveled while stopping via friction-braking (i.e., distance to brake)
        max_items = len(self.cyc.time_s)
        if v0 <= v_brake:
            if modify_cycle:
                _, n = self.cyc.modify_with_braking_trajectory(a_brake, i)
                self.impose_coast[i:] = False
                for ni in range(n):
                    if (i + ni) < max_items:
                        self.impose_coast[i + ni] = True
            return -0.5 * v0 * v0 / a_brake
        dtb = -0.5 * v_brake * v_brake / a_brake
        d = 0.0
        d_max = distances_m[-1] - dtb
        unique_grades = np.unique(grade_by_distance)
        unique_grade = unique_grades[0] if len(unique_grades) == 1 else None
        MAX_ITER = 180
        ITERS_PER_STEP = 2 if self.sim_params.favor_grade_accuracy else 1
        new_speeds_m_per_s = []
        v = v0
        iter = 0
        idx = i
        dts0 = self.cyc0.calc_distance_to_next_stop_from(d0, cache=self.cyc0_cache)
        while v > v_brake and v >= 0.0 and d <= d_max and iter < MAX_ITER and idx < len(self.mps_ach):
            dt_s = self.cyc0.dt_s_at_i(idx)
            gr = unique_grade if unique_grade is not None else self.cyc0_cache.interp_grade(d + d0)
            k = self._calc_dvdd(v, gr)
            v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s)
            vavg = 0.5 * (v + v_next)
            dd = vavg * dt_s
            for _ in range(ITERS_PER_STEP):
                k = self._calc_dvdd(vavg, gr)
                v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s)
                vavg = 0.5 * (v + v_next)
                dd = vavg * dt_s
                if self.sim_params.favor_grade_accuracy:
                    gr = unique_grade if unique_grade is not None else self.cyc0.average_grade_over_range(d + d0, dd, cache=self.cyc0_cache)
            if k >= 0.0 and unique_grade is not None:
                # there is no solution for coastdown -- speed will never decrease
                return NOT_FOUND
            if v_next <= v_brake:
                break
            vavg = 0.5 * (v + v_next)
            dd = vavg * dt_s
            dtb = -0.5 * v_next * v_next / a_brake
            d += dd
            new_speeds_m_per_s.append(v_next)
            v = v_next
            if d + dtb >= dts0:
                break
            iter += 1
            idx += 1
        if iter < MAX_ITER and idx < len(self.mps_ach):
            dtb = -0.5 * v * v / a_brake
            dtb_target = min(max(dts0 - d, 0.5 * dtb), 2.0 * dtb)
            dtsc = d + dtb_target
            if modify_cycle:
                for di, new_speed in enumerate(new_speeds_m_per_s):
                    idx = min(i + di, len(self.mps_ach) - 1)
                    self.cyc.mps[idx] = new_speed
                _, n = self.cyc.modify_with_braking_trajectory(a_brake, i+len(new_speeds_m_per_s), dts_m=dtb_target)
                self.impose_coast[i:] = False
                for di in range(len(new_speeds_m_per_s) + n):
                    idx = min(i + di, len(self.mps_ach) - 1)
                    self.impose_coast[idx] = True
            return dtsc
        return NOT_FOUND

    def _calc_distance_to_stop_coast_v2(self, i:int):
        """
        Calculate the distance to stop via coasting in meters.
        - i: non-negative-integer, the current index
        RETURN: non-negative-number or -1.0
        - if -1.0, it means there is no solution to a coast-down distance.
            This can happen due to being too close to the given
            stop or perhaps due to coasting downhill
        - if a non-negative-number, the distance in meters that the vehicle
            would freely coast if unobstructed. Accounts for grade between
            the current point and end-point
        """
        NOT_FOUND = -1.0
        v0 = self.cyc.mps[i-1]
        v_brake = self.sim_params.coast_brake_start_speed_m_per_s
        a_brake = self.sim_params.coast_brake_accel_m_per_s2
        ds = self.cyc0_cache.trapz_distances_m
        gs = self.cyc0.grade
        d0 = cycle.trapz_step_start_distance(self.cyc, i)
        ds_mask = ds >= d0
        grade_by_distance = gs[ds_mask]
        veh_mass_kg = self.veh.veh_kg
        air_density_kg__m3 = self.props.air_density_kg_per_m3
        CDFA_m2 = self.veh.drag_coef * self.veh.frontal_area_m2
        rrc = self.veh.wheel_rr_coef
        gravity_m__s2 = self.props.a_grav_mps2
        # distance traveled while stopping via friction-braking (i.e., distance to brake)
        dtb = -0.5 * v_brake * v_brake / a_brake
        if v0 <= v_brake:
            return -0.5 * v0 * v0 / a_brake
        unique_grades = np.unique(grade_by_distance)
        unique_grade = unique_grades[0] if len(unique_grades) == 1 else None
        if unique_grade is not None:
            # if there is only one grade, there may be a closed-form solution
            theta = np.arctan(unique_grade)
            c1 = gravity_m__s2 * (np.sin(theta) + rrc * np.cos(theta))
            c2 = (air_density_kg__m3 * CDFA_m2) / (2.0 * veh_mass_kg)
            v02 = v0 * v0
            vb2 = v_brake * v_brake
            d = NOT_FOUND
            a1 = c1 + c2 * v02
            b1 = c1 + c2 * vb2
            if c2 == 0.0:
                if c1 > 0.0:
                    d = (1.0 / (2.0 * c1)) * (v02 - vb2)
            elif a1 > 0.0 and b1 > 0.0:
                d = (1.0 / (2.0 * c2)) * (np.log(a1) - np.log(b1))
            if d != NOT_FOUND:
                return d + dtb
        return self._generate_coast_trajectory(i, False)

    def _should_impose_coast(self, i):
        """
        - i: non-negative integer, the current position in cyc
        RETURN: Bool if vehicle should initiate coasting
        Coast logic is that the vehicle should coast if it is within coasting distance of a stop:
        - if distance to coast from start of step is <= distance to next stop
        - AND distance to coast from end of step (using prescribed speed) is > distance to next stop
        - ALSO, vehicle must have been at or above the coast brake start speed at beginning of step
        - AND, must be at least 4 x distances-to-break away
        """
        if self.sim_params.coast_start_speed_m_per_s > 0.0:
            return self.cyc.mps[i] >= self.sim_params.coast_start_speed_m_per_s
        v0 = self.mps_ach[i-1]
        if v0 < self.sim_params.coast_brake_start_speed_m_per_s:
            return False
        # distance to stop by coasting from start of step (i-1)
        dtsc0 = self._calc_distance_to_stop_coast_v2(i)
        if dtsc0 < 0.0:
            return False
        # distance to next stop (m)
        d0 = cycle.trapz_step_start_distance(self.cyc, i)
        dts0 = self.cyc0.calc_distance_to_next_stop_from(d0, cache=self.cyc0_cache)
        dtb = -0.5 * v0 * v0 / self.sim_params.coast_brake_accel_m_per_s2
        return dtsc0 >= dts0 and dts0 >= (4.0 * dtb)

    def _calc_next_rendezvous_trajectory(self, i, min_accel_m__s2, max_accel_m__s2):
        """
        Calculate next rendezvous trajectory for eco-coasting.
        - i: non-negative integer, the index into cyc for the end of first step
            (i.e., the step that may be modified; should be i)
        - min_accel_m__s2: number, the minimum acceleration permitted (m/s2)
        - max_accel_m__s2: number, the maximum acceleration permitted (m/s2)
        RETURN: (Tuple
            found_rendezvous: Bool, if True the remainder of the data is valid; if False, no rendezvous found
            n: positive integer, the number of steps ahead to rendezvous at
            jerk_m__s3: number, the Jerk or first-derivative of acceleration (m/s3)
            accel_m__s2: number, the initial acceleration of the trajectory (m/s2)
        )
        If no rendezvous exists within the scope, the returned tuple has False for the first item.
        Otherwise, returns the next closest rendezvous in time/space
        """
        # v0 is where n=0, i.e., idx-1
        v0 = self.cyc.mps[i-1]
        brake_start_speed_m_per_s = self.sim_params.coast_brake_start_speed_m_per_s
        brake_accel_m_per_s2 = self.sim_params.coast_brake_accel_m_per_s2
        time_horizon_s = max(self.sim_params.coast_time_horizon_for_adjustment_s, 1.0)
        # distance_horizon_m = 1000.0
        not_found_n = 0
        not_found_jerk_m_per_s3 = 0.0
        not_found_accel_m_per_s2 = 0.0
        not_found = (False, not_found_n, not_found_jerk_m_per_s3,
                     not_found_accel_m_per_s2)
        if v0 < (brake_start_speed_m_per_s + 1e-6):
            return not_found
        if min_accel_m__s2 > max_accel_m__s2:
            min_accel_m__s2, max_accel_m__s2 = max_accel_m__s2, min_accel_m__s2
        num_samples = len(self.cyc.mps)
        d0 = cycle.trapz_step_start_distance(self.cyc, i)
        # a_proposed = (v1 - v0) / dt
        # distance to stop from start of time-step
        dts0 = self.cyc0.calc_distance_to_next_stop_from(d0, cache=self.cyc0_cache)
        if dts0 < 0.0:
            # no stop to coast towards or we're there...
            print("WARNING! Exiting as there is no stop to coast towards")
            print(f"... i = {i}")
            print(f"... v0 = {v0}")
            print(f"... self.mps_ach[i-1] = {self.mps_ach[i-1]}")
            print(f"... brake_start_speed_m_per_s = {brake_start_speed_m_per_s}")
            print(f"... d0 = {d0}")
            print(f"... dts0 = {dts0}")
            return not_found
        v1 = self.cyc.mps[i]
        dt = self.cyc.dt_s_at_i(i)
        # distance to brake from the brake start speed (m/s)
        dtb = -0.5 * brake_start_speed_m_per_s * brake_start_speed_m_per_s / brake_accel_m_per_s2
        # distance to brake initiation from start of time-step (m)
        dtbi0 = dts0 - dtb
        if dtbi0 < 0.0:
            return not_found
        # Now, check rendezvous trajectories
        step_idx = i
        dt_plan = 0.0
        r_best_found = False
        r_best_n = 0
        r_best_jerk_m__s3 = 0.0
        r_best_accel_m__s2 = 0.0
        r_best_accel_spread_m__s2 = 0.0
        while dt_plan <= time_horizon_s and step_idx < num_samples:
            dt_plan += self.cyc0.dt_s_at_i(step_idx)
            step_ahead = step_idx - (i - 1)
            if step_ahead == 1:
                # for brake init rendezvous
                accel = (brake_start_speed_m_per_s - v0) / dt
                v1 = max(0.0, v0 + accel * dt)
                dd_proposed = ((v0 + v1) / 2.0) * dt
                if np.abs(v1 - brake_start_speed_m_per_s) < 1e-6 and np.abs(dtbi0 - dd_proposed) < 1e-6:
                    r_best_found = True
                    r_best_n = 1
                    r_best_accel_m__s2 = accel
                    r_best_jerk_m__s3 = 0.0
                    r_best_accel_spread_m__s2 = 0.0
                    break
            else:
                # rendezvous trajectory for brake-start -- assumes fixed time-steps
                if dtbi0 > 0.0:
                    r_bi_jerk_m__s3, r_bi_accel_m__s2 = cycle.calc_constant_jerk_trajectory(
                        step_ahead, 0.0, v0, dtbi0, brake_start_speed_m_per_s, dt)
                    if r_bi_accel_m__s2 < max_accel_m__s2 and r_bi_accel_m__s2 > min_accel_m__s2 and r_bi_jerk_m__s3 >= 0.0:
                        as_bi = np.array([
                            cycle.accel_for_constant_jerk(
                                n, r_bi_accel_m__s2, r_bi_jerk_m__s3, dt)
                            for n in range(step_ahead)
                        ])
                        accel_spread = np.abs(as_bi.max() - as_bi.min())
                        flag = (
                            (as_bi.max() < (max_accel_m__s2 + 1e-6)
                                and as_bi.min() > (min_accel_m__s2 - 1e-6))
                            and
                            (not r_best_found or (
                                accel_spread < r_best_accel_spread_m__s2))
                        )
                        if flag:
                            r_best_found = True
                            r_best_n = step_ahead
                            r_best_accel_m__s2 = r_bi_accel_m__s2
                            r_best_jerk_m__s3 = r_bi_jerk_m__s3
                            r_best_accel_spread_m__s2 = accel_spread
            step_idx += 1
        if r_best_found:
            return (r_best_found, r_best_n, r_best_jerk_m__s3, r_best_accel_m__s2)
        return not_found

    def _set_coast_delay(self, i):
        """
        Coast Delay allows us to represent coasting to a stop when the lead
        vehicle has already moved on from that stop.  In this case, the coasting
        vehicle need not dwell at this or any stop while it is lagging behind
        the lead vehicle in distance. Instead, the vehicle comes to a stop and
        resumes mimicing the lead-vehicle trace at the first time-step the
        lead-vehicle moves past the stop-distance. This index is the "coast delay index".
        
        Arguments
        ---------
        - i: integer, the step index

        Resets the coast_delay_index to 0 and calculates and sets the next
        appropriate coast_delay_index if appropriate
        """
        SPEED_TOL = 0.01 # m/s
        DIST_TOL = 0.1 # meters
        self.coast_delay_index[i:] = 0 # clear all future coast-delays
        coast_delay = None
        if not self.sim_params.idm_allow and self.cyc.mps[i] < SPEED_TOL:
            d0 = cycle.trapz_step_start_distance(self.cyc, i)
            d0_lv = self.cyc0_cache.trapz_distances_m[i-1]
            dtlv0 = d0_lv - d0
            if np.abs(dtlv0) > DIST_TOL:
                d_lv = 0.0
                min_dtlv = None
                for idx, (dd, v) in enumerate(zip(cycle.trapz_step_distances(self.cyc0), self.cyc0.mps)):
                    d_lv += dd
                    dtlv = np.abs(d_lv - d0)
                    if v < SPEED_TOL and (min_dtlv is None or dtlv <= min_dtlv):
                        if min_dtlv is None or dtlv < min_dtlv or (d0 < d0_lv and min_dtlv == dtlv):
                            coast_delay = i - idx
                        min_dtlv = dtlv
                    if min_dtlv is not None and dtlv > min_dtlv:
                        break
        if coast_delay is not None:
            if coast_delay < 0:
                for idx in range(i, len(self.cyc0.mps)):
                    self.coast_delay_index[idx] = coast_delay
                    coast_delay += 1
                    if coast_delay == 0:
                        break
            else:
                self.coast_delay_index[i:] = coast_delay

    def _prevent_collisions(self, i: int, passing_tol_m: Optional[float] = None) -> bool:
        """
        Prevent collision between the vehicle in cyc and the one in cyc0.
        If a collision will take place, reworks the cyc such that a rendezvous occurs instead.
        Arguments
        - i: int, index for consideration
        - passing_tol_m: None | float, tolerance for how far we have to go past the lead vehicle to be considered "passing"
        RETURN: Bool, True if cyc was modified
        """
        passing_tol_m = 1.0 if passing_tol_m is None else passing_tol_m
        collision = cycle.detect_passing(self.cyc, self.cyc0, i, passing_tol_m)
        if not collision.has_collision:
            return False
        best = _RendezvousTrajectory(
            found_trajectory=False,
            idx=0,
            n=0,
            full_brake_steps=0,
            jerk_m_per_s3=0.0,
            accel0_m_per_s2=0.0,
            accel_spread=0.0,
        )
        a_brake_m_per_s2 = self.sim_params.coast_brake_accel_m_per_s2
        assert a_brake_m_per_s2 < 0.0, f"brake acceleration must be negative; got {a_brake_m_per_s2} m/s2"
        for full_brake_steps in range(4):
            for di in range(len(self.mps_ach) - i):
                idx = i + di
                if not self.impose_coast[idx]:
                    if idx == i:
                        break
                    else:
                        continue
                n = collision.idx - idx + 1 - full_brake_steps
                if n < 2:
                    break
                if (idx - 1 + full_brake_steps) >= len(self.cyc.time_s):
                    break
                dt = collision.time_step_duration_s
                v_start_m_per_s = self.cyc.mps[idx - 1]
                dt_full_brake = self.cyc.time_s[idx - 1 + full_brake_steps] - self.cyc.time_s[idx - 1]
                dv_full_brake = dt_full_brake * a_brake_m_per_s2
                v_start_jerk_m_per_s = max(v_start_m_per_s + dv_full_brake, 0.0)
                dd_full_brake = 0.5 * (v_start_m_per_s + v_start_jerk_m_per_s) * dt_full_brake
                d_start_m = cycle.trapz_step_start_distance(self.cyc, idx) + dd_full_brake
                if collision.distance_m <= d_start_m:
                    continue
                jerk_m_per_s3, accel0_m_per_s2 = cycle.calc_constant_jerk_trajectory(
                    n,
                    d_start_m,
                    v_start_jerk_m_per_s,
                    collision.distance_m,
                    collision.speed_m_per_s,
                    dt
                )
                accels_m_per_s2 = np.array([
                    cycle.accel_for_constant_jerk(ni, accel0_m_per_s2, jerk_m_per_s3, dt)
                    for ni in range(n)
                    if (ni + idx + full_brake_steps) < len(self.cyc.time_s)
                ])
                trace_accels_m_per_s2 = np.array([
                    (self.cyc.mps[ni + idx + full_brake_steps] - self.cyc.mps[ni + idx - 1 + full_brake_steps])
                    / self.cyc.dt_s_at_i(ni + idx - 1 + full_brake_steps)
                    for ni in range(n)
                    if (ni + idx + full_brake_steps) < len(self.cyc.time_s)
                ])
                all_sub_coast = (trace_accels_m_per_s2 >= accels_m_per_s2).all()
                min_accel_m_per_s2 = accels_m_per_s2.min()
                max_accel_m_per_s2 = accels_m_per_s2.max()
                accept = all_sub_coast
                accel_spread = np.abs(max_accel_m_per_s2 - min_accel_m_per_s2)
                if accept and (not best.found_trajectory or accel_spread < best.accel_spread):
                    best = _RendezvousTrajectory(
                        found_trajectory=True,
                        idx=idx,
                        n=n,
                        full_brake_steps=full_brake_steps,
                        jerk_m_per_s3=jerk_m_per_s3,
                        accel0_m_per_s2=accel0_m_per_s2,
                        accel_spread=accel_spread,
                    )
            if best.found_trajectory:
                break
        if not best.found_trajectory:
            new_passing_tol_m = 10.0 if passing_tol_m < 10.0 else passing_tol_m + 5.0
            if new_passing_tol_m > 60.0:
                return False
            return self._prevent_collisions(i, new_passing_tol_m)
        for fbs in range(best.full_brake_steps):
            dt = self.cyc.time_s[best.idx + fbs] - self.cyc.time_s[best.idx - 1]
            dv = a_brake_m_per_s2 * dt
            v_start = self.cyc.mps[best.idx - 1]
            self.cyc.mps[best.idx + fbs] = max(v_start + dv, 0.0)
            self.impose_coast[best.idx + fbs] = True
            self.coast_delay_index[best.idx + fbs] = 0
        self.cyc.modify_by_const_jerk_trajectory(
            best.idx + best.full_brake_steps,
            best.n,
            best.jerk_m_per_s3,
            best.accel0_m_per_s2
        )
        self.impose_coast[(best.idx + best.n):] = False 
        self.coast_delay_index[(best.idx + best.n):] = 0
        return True

    def _set_coast_speed(self, i):
        """
        Placeholder for method to impose coasting.
        Might be good to include logic for deciding when to coast.
        Solve for the next-step speed that will yield a zero roadload
        """
        TOL = 1e-6
        v0 = self.mps_ach[i-1]
        if v0 > TOL and not self.impose_coast[i]:
            if self._should_impose_coast(i):
                d = self._generate_coast_trajectory(i, True)
                if d < 0:
                    self.impose_coast[i:] = False
                if not self.sim_params.coast_allow_passing:
                    self._prevent_collisions(i)
        if not self.impose_coast[i]:
            if not self.sim_params.idm_allow:
                self.cyc.mps[i] = self.cyc0.mps[min(max(i - self.coast_delay_index[i], 0), len(self.cyc0.mps) - 1)]
            return
        v1_traj = self.cyc.mps[i]
        if v0 > self.sim_params.coast_brake_start_speed_m_per_s:
            if self.sim_params.coast_allow_passing:
                # we could be coasting downhill so could in theory go to a higher speed
                # since we can pass, allow vehicle to go up to max coasting speed (m/s)
                # the solver will show us what we can actually achieve
                self.cyc.mps[i] = self.sim_params.coast_max_speed_m_per_s
            else:
                self.cyc.mps[i] = min(v1_traj, self.sim_params.coast_max_speed_m_per_s)
        # Solve for the actual coasting speed
        self.solve_step(i)
        self.newton_iters[i] = 0  # reset newton iters
        self.cyc.mps[i] = self.mps_ach[i]
        accel_proposed = (self.cyc.mps[i] -
                          self.cyc.mps[i-1]) / self.cyc.dt_s_at_i(i)
        if self.cyc.mps[i] < TOL:
            self.impose_coast[i:] = False
            self._set_coast_delay(i)
            self.cyc.mps[i] = 0.0
            return
        if np.abs(self.cyc.mps[i] - v1_traj) > TOL:
            max_idx = len(self.cyc.time_s) - 1
            adjusted_current_speed = False
            brake_speed_start_tol_m_per_s = 0.1
            if self.cyc.mps[i] < (self.sim_params.coast_brake_start_speed_m_per_s - brake_speed_start_tol_m_per_s):
                _, num_steps = self.cyc.modify_with_braking_trajectory(
                    self.sim_params.coast_brake_accel_m_per_s2, i)
                self.impose_coast[i:] = False
                for di in range(0, num_steps):
                    if (i + di) <= max_idx:
                        self.impose_coast[i + di] = True
                adjusted_current_speed = True
            else:
                traj_found, traj_n, traj_jerk_m__s3, traj_accel_m__s2 = self._calc_next_rendezvous_trajectory(
                    i,
                    min_accel_m__s2=self.sim_params.coast_brake_accel_m_per_s2,
                    max_accel_m__s2=min(accel_proposed, 0.0)
                )
                if traj_found:
                    # adjust cyc to perform the trajectory
                    final_speed_m_per_s = self.cyc.modify_by_const_jerk_trajectory(
                        i, traj_n, traj_jerk_m__s3, traj_accel_m__s2)
                    self.impose_coast[i:] = False
                    for di in range(0, traj_n):
                        if i + di <= max_idx:
                            self.impose_coast[i + di] = True
                    adjusted_current_speed = True
                    i_for_brake = i + traj_n
                    if np.abs(final_speed_m_per_s - self.sim_params.coast_brake_start_speed_m_per_s) < 0.1:
                        _, num_steps = self.cyc.modify_with_braking_trajectory(
                            self.sim_params.coast_brake_accel_m_per_s2,
                            i_for_brake,
                        )
                        for di in range(0, num_steps):
                            if i_for_brake + di <= max_idx:
                                self.impose_coast[i_for_brake + di] = True
                        adjusted_current_speed = True
                    else:
                        print(f"WARNING! final_speed_m_per_s not close to coast_brake_start_speed for i = {i}")
                        print(f"... final_speed_m_per_s = {final_speed_m_per_s}")
                        print(f"... self.sim_params.coast_brake_start_speed_m_per_s = {self.sim_params.coast_brake_start_speed_m_per_s}")
                        print(f"... i_for_brake = {i_for_brake}")
                        print(f"... traj_n = {traj_n}")
            if adjusted_current_speed:
                if not self.sim_params.coast_allow_passing:
                    self._prevent_collisions(i)
                self.solve_step(i)
                self.newton_iters[i] = 0  # reset newton iters
                self.cyc.mps[i] = self.mps_ach[i]

    def set_post_scalars(self):
        """Sets scalar variables that can be calculated after a cycle is run. 
        This includes mpgge, various energy metrics, and others"""

        self.fs_cumu_mj_out_ach = (
            self.fs_kw_out_ach * self.cyc.dt_s).cumsum() * 1e-3

        if self.fs_kwh_out_ach.sum() == 0:
            self.mpgge = 0.0

        else:
            self.mpgge = self.dist_mi.sum() / (self.fs_kwh_out_ach.sum() / self.props.kwh_per_gge)

        self.roadway_chg_kj = (
            self.roadway_chg_kw_out_ach * self.cyc.dt_s).sum()
        self.ess_dischg_kj = - \
            (self.soc[-1] - self.soc[0]) * self.veh.ess_max_kwh * 3.6e3
        dist_mi = self.dist_mi.sum()
        self.battery_kwh_per_mi = (
            self.ess_dischg_kj / 3.6e3) / dist_mi if dist_mi > 0 else 0.0
        self.electric_kwh_per_mi = (
            (self.roadway_chg_kj + self.ess_dischg_kj) / 3.6e3) / dist_mi if dist_mi > 0 else 0.0
        self.fuel_kj = (self.fs_kw_out_ach * self.cyc.dt_s).sum()

        if (self.fuel_kj + self.roadway_chg_kj) == 0:
            self.ess2fuel_kwh = 1.0

        else:
            self.ess2fuel_kwh = self.ess_dischg_kj / \
                (self.fuel_kj + self.roadway_chg_kj)

        # energy audit calcs
        self.drag_kj = (self.drag_kw * self.cyc.dt_s).sum()
        self.ascent_kj = (self.ascent_kw * self.cyc.dt_s).sum()
        self.rr_kj = (self.rr_kw * self.cyc.dt_s).sum()

        self.ess_loss_kw[1:] = np.array(
            [0 if (self.veh.ess_max_kw == 0 or self.veh.ess_max_kwh == 0)
             else -self.ess_kw_out_ach[i] - (-self.ess_kw_out_ach[i] * np.sqrt(self.veh.ess_round_trip_eff))
                if self.ess_kw_out_ach[i] < 0
             else self.ess_kw_out_ach[i] * (1.0 / np.sqrt(self.veh.ess_round_trip_eff)) - self.ess_kw_out_ach[i]
             for i in range(1, len(self.cyc.time_s))]
        )

        self.brake_kj = (self.cyc_fric_brake_kw * self.cyc.dt_s).sum()
        self.trans_kj = (
            (self.trans_kw_in_ach - self.trans_kw_out_ach) * self.cyc.dt_s).sum()
        self.mc_kj = ((self.mc_elec_kw_in_ach -
                       self.mc_mech_kw_out_ach) * self.cyc.dt_s).sum()
        self.ess_eff_kj = (self.ess_loss_kw * self.cyc.dt_s).sum()
        self.aux_kj = (self.aux_in_kw * self.cyc.dt_s).sum()
        self.fc_kj = ((self.fc_kw_in_ach - self.fc_kw_out_ach)
                      * self.cyc.dt_s).sum()

        self.net_kj = self.drag_kj + self.ascent_kj + self.rr_kj + self.brake_kj + self.trans_kj \
            + self.mc_kj + self.ess_eff_kj + self.aux_kj + self.fc_kj

        self.ke_kj = 0.5 * self.veh.veh_kg * \
            (self.mps_ach[0] ** 2 - self.mps_ach[-1] ** 2) / 1_000

        self.energy_audit_error = ((self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj) - self.net_kj
                                   ) / (self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj)

        if (np.abs(self.energy_audit_error) > self.sim_params.energy_audit_error_tol):
            logger.warning(
                "problem detected with conservation of energy; " +
                f"energy audit error: {np.round(self.energy_audit_error, 5)}"
            )

        self.accel_kw[1:] = (self.veh.veh_kg / (2.0 * (self.cyc.dt_s[1:]))) * (
            self.mps_ach[1:] ** 2 - self.mps_ach[:-1] ** 2) / 1_000

        self.trace_miss = False
        dist_m = self.cyc0.dist_m.sum()
        self.trace_miss_dist_frac = abs(
            self.dist_m.sum() - self.cyc0.dist_m.sum()) / dist_m if dist_m > 0 else 0.0
        self.trace_miss_time_frac = abs(
            self.cyc.time_s[-1] - self.cyc0.time_s[-1]) / self.cyc0.time_s[-1]

        if not(self.sim_params.missed_trace_correction):
            if self.trace_miss_dist_frac > self.sim_params.trace_miss_dist_tol:
                self.trace_miss = True
                logger.warning(
                    f"trace miss distance fraction {np.round(self.trace_miss_dist_frac, 5)} " +
                    f"exceeds tolerance of {np.round(self.sim_params.trace_miss_dist_tol, 5)}"
                )
        else:
            if self.trace_miss_time_frac > self.sim_params.trace_miss_time_tol:
                self.trace_miss = True
                logger.warning(
                    f"trace miss time fraction {np.round(self.trace_miss_time_frac, 5)} " +
                    f"exceeds tolerance of {np.round(self.sim_params.trace_miss_time_tol, 5)}"
                )

        # NOTE: I believe this should be accessing self.cyc0.mps[i] instead of self.cyc.mps[i]; self.cyc may be modified...
        self.trace_miss_speed_mps = max([
            abs(self.mps_ach[i] - self.cyc.mps[i]) for i in range(len(self.cyc.time_s))
        ])
        if self.trace_miss_speed_mps > self.sim_params.trace_miss_speed_mps_tol:
            self.trace_miss = True
            logger.warning(
                f"trace miss speed {np.round(self.trace_miss_speed_mps, 5)} m/s " +
                f"exceeds tolerance of {np.round(self.sim_params.trace_miss_speed_mps_tol, 5)} m/s"
            )


    def to_rust(self):
        "Create a rust version of SimDrive"
        return copy_sim_drive(self, 'rust', True)


class LegacySimDrive(object):
    pass


ref_sim_drive = SimDrive(cycle.ref_cyc, vehicle.ref_veh)
sd_params = inspect_utils.get_attrs(ref_sim_drive)


def copy_sim_drive(sd: SimDrive, return_type: str = None, deep: bool = True) -> SimDrive:
    """Returns copy of SimDriveClassic or SimDriveJit as SimDriveClassic.
    Arguments:
    ----------
    sd: instantiated SimDriveClassic or SimDriveJit
    return_type: 
        default: infer from type of sd
        'python': Cycle 
        'legacy': LegacyCycle
        'rust': RustCycle
    deep: if True, uses deepcopy on everything
    """

    # TODO: if the rust version is input, make sure to copy lists to numpy arrays
    # TODO: no need to implement dict for copy_sim_drive, but please do for the subordinate classes

    if return_type is None:
        # if type(sd) == RustSimDrive:
        #    return_type = 'rust'
        if type(sd) == SimDrive:
            return_type = 'python'
        elif type(sd) == RustSimDrive:
            return_type = 'rust'
        elif type(sd) == LegacySimDrive:
            return_type = "legacy"
        else:
            raise NotImplementedError(
                "Only implemented for rust, python, or legacy.")

    cyc_return_type = 'python' if return_type == 'python' else return_type
    veh_return_type = 'vehicle' if return_type == 'python' else return_type
    cyc = cycle.copy_cycle(sd.cyc0, cyc_return_type, deep)
    veh = vehicle.copy_vehicle(sd.veh, veh_return_type, deep)

    if return_type == 'rust':
        return RustSimDrive(cyc, veh)

    sd_copy = SimDrive(cyc, veh)
    for key in inspect_utils.get_attrs(sd):
        if key == 'cyc':
            sd_copy.__setattr__(
                key,
                cycle.copy_cycle(sd.__getattribute__(key), cyc_return_type, deep))
        elif key == 'cyc0':
            pass
        elif key == 'veh':
            pass
        elif key == 'sim_params':
            sp_return_type = 'python' if (
                return_type == 'python' or return_type == 'legacy') else return_type
            sd_copy.sim_params = copy_sim_params(sd.sim_params, sp_return_type)
        elif key == 'props':
            pp_return_type = 'python' if (
                return_type == 'python' or return_type == 'legacy') else return_type
            sd_copy.props = params.copy_physical_properties(
                sd.props, pp_return_type)
        else:
            # should be ok to deep copy
            val = sd.__getattribute__(key)
            sd_copy.__setattr__(key, copy.deepcopy(val) if deep else val)

    return sd_copy


def sim_drive_equal(a: SimDrive, b: SimDrive) -> bool:
    ""
    if a is b:
        return True
    for k in ref_sim_drive.__dict__.keys():
        if k == 'cyc0_cache':
            continue
        a_val = a.__getattribute__(k)
        b_val = b.__getattribute__(k)
        if k == 'cyc' or k == 'cyc0':
            if not cycle.cyc_equal(a_val, b_val):
                logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif k == 'veh':
            if not vehicle.veh_equal(a_val, b_val):
                logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif k == 'props':
            if not params.physical_properties_equal(a_val, b_val):
                logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif k == 'sim_params':
            if not sim_params_equal(a_val, b_val):
                logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif 'tolist' in a_val.__dir__() + b_val.__dir__():
            if 'tolist' in a_val.__dir__():
                a_val = np.array(a_val.tolist())
            if 'tolist' in b_val.__dir__():
                b_val = np.array(b_val.tolist())
            if not (a_val == b_val).all():
                logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif type(a_val) == np.ndarray or type(b_val) == np.ndarray:
            if not (a_val == b_val).all():
                logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif type(a_val) == list and type(b_val) == list:
            if not a_val == b_val:
                logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif a_val != b_val:
            logger.debug(f"unequal at key {k}: {a_val} != {b_val}")
            return False
    return True


def run_simdrive_for_accel_test(sd: SimDrive):
    """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType."""
    if sd.veh.veh_pt_type == CONV:  # Conventional
        # If no EV / Hybrid components, no SOC considerations.
        init_soc = (sd.veh.max_soc + sd.veh.min_soc) / 2.0
        sd.sim_drive_walk(init_soc)
    elif sd.veh.veh_pt_type == HEV:  # HEV
        init_soc = (sd.veh.max_soc + sd.veh.min_soc) / 2.0
        sd.sim_drive_walk(init_soc)
    else:
        # If EV, initializing initial SOC to maximum SOC.
        init_soc = sd.veh.max_soc
        sd.sim_drive_walk(init_soc)
    sd.set_post_scalars()


class SimDrivePost(object):
    """Class for post-processing of SimDrive instance.  Requires already-run 
    SimDrive instance."""

    def __init__(self, sim_drive: SimDrive):
        """Arguments:
        ---------------
        sim_drive: solved sim_drive object"""

        for item in inspect_utils.get_attrs(sim_drive):
            self.__setattr__(item, sim_drive.__getattribute__(item))

    # optional post-processing methods
    def get_diagnostics(self):
        """This method is to be run after runing sim_drive if diagnostic variables 
        are needed.  Diagnostic variables are returned in a dict.  Diagnostic variables include:
        - final integrated value of all positive powers
        - final integrated value of all negative powers
        - total distance traveled
        - miles per gallon gasoline equivalent (mpgge)"""

        base_var_list = list(self.__dict__.keys())
        pw_var_list = [var for var in base_var_list if re.search(
            r'\w*_kw(?!h)\w*', var)]
        # find all vars containing 'Kw' but not 'Kwh'

        prog = re.compile(r'(\w*)_kw(?!h)(\w*)')
        # find all vars containing 'Kw' but not Kwh and capture parts before and after 'Kw'
        # using compile speeds up iteration

        # create positive and negative versions of all time series with units of kW
        # then integrate to find cycle end pos and negative energies
        tempvars = {}  # dict for contaning intermediate variables
        output = {}
        for var in pw_var_list:
            tempvars[var + '_pos'] = [x if x >= 0
                                      else 0
                                      for x in np.array(self.__getattribute__(var))]
            tempvars[var + '_neg'] = [x if x < 0
                                      else 0
                                      for x in np.array(self.__getattribute__(var))]

            # Assign values to output dict for positive and negative energy variable names
            search = prog.search(var)
            output[search[1] + '_kj' + search[2] +
                   '_pos'] = np.trapz(np.array(tempvars[var + '_pos']), np.array(self.cyc.time_s))
            output[search[1] + '_kj' + search[2] +
                   '_neg'] = np.trapz(np.array(tempvars[var + '_neg']), np.array(self.cyc.time_s))

        output['dist_miles_final'] = sum(np.array(self.dist_mi))
        if sum(np.array(self.fs_kwh_out_ach)) > 0:
            output['mpgge'] = sum(
                np.array(self.dist_mi)) / sum(np.array(self.fs_kwh_out_ach)) * self.props.kwh_per_gge
        else:
            output['mpgge'] = 0

        return output

    def set_battery_wear(self):
        """Battery wear calcs"""

        add_kwh = self.add_kwh if type(self.add_kwh) is np.ndarray else np.array(self.add_kwh)
        add_kwh[1:] = np.array([
            (self.ess_cur_kwh[i] - self.ess_cur_kwh[i-1]) + add_kwh[i-1]
            if self.ess_cur_kwh[i] > self.ess_cur_kwh[i-1]
            else 0
            for i in range(1, len(self.ess_cur_kwh))])
        self.add_kwh = add_kwh

        dod_cycs = self.dod_cycs if type(self.dod_cycs) is np.ndarray else np.array(self.dod_cycs)
        if self.veh.ess_max_kwh == 0:
            dod_cycs[1:] = np.array(
                [0.0 for i in range(1, len(self.ess_cur_kwh))])
        else:
            dod_cycs[1:] = np.array([
                self.add_kwh[i-1] / self.veh.ess_max_kwh if self.add_kwh[i] == 0
                else 0
                for i in range(1, len(self.ess_cur_kwh))])
        self.dod_cycs = dod_cycs

        self.ess_perc_dead = np.array([
            np.power(self.veh.ess_life_coef_a, 1.0 / self.veh.ess_life_coef_b) / np.power(self.dod_cycs[i],
                                                                                          1.0 / self.veh.ess_life_coef_b)
            if self.dod_cycs[i] != 0
            else 0
            for i in range(0, len(self.ess_cur_kwh))])


def SimDriveJit(cyc_jit, veh_jit):
    """
    deprecated
    """
    raise NotImplementedError("This function has been deprecated.")


def estimate_soc_corrected_fuel_kJ(sd: SimDrive) -> float:
    """
    - sd: SimDriveClassic, the simdrive instance after simulation
    RETURN: number, the kJ of fuel corrected for SOC imbalance
    """
    if sd.veh.veh_pt_type != HEV:
        raise ValueError(
            f"SimDrive instance must have a vehPtType of HEV; found {sd.veh.veh_pt_type}")

    def f(mask, numer, denom, default):
        m = np.array(mask)
        n = np.array(numer)
        d = np.array(denom)
        if not m.any():
            return default
        return n[m].sum() / d[m].sum()
    kJ__kWh = 3600.0
    delta_soc = sd.soc[-1] - sd.soc[0]
    ess_eff = np.sqrt(sd.veh.ess_round_trip_eff)
    mc_chg_eff = f(np.array(sd.mc_mech_kw_out_ach) < 0.0,
                   np.array(sd.mc_elec_kw_in_ach), np.array(sd.mc_mech_kw_out_ach), sd.veh.mc_peak_eff)
    mc_dis_eff = f(np.array(sd.mc_mech_kw_out_ach)> 0.0,
                   np.array(sd.mc_mech_kw_out_ach), np.array(sd.mc_elec_kw_in_ach), mc_chg_eff)
    ess_traction_frac = f(np.array(sd.mc_elec_kw_in_ach)
                          > 0.0, np.array(sd.mc_elec_kw_in_ach), np.array(sd.ess_kw_out_ach), 1.0)
    fc_eff = f(
        np.array(sd.trans_kw_in_ach) > 0.0,
        np.array(sd.fc_kw_out_ach),
        np.array(sd.fc_kw_in_ach),
        np.array(sd.fc_kw_out_ach).sum() / np.array(sd.fc_kw_in_ach).sum()
    )
    if delta_soc >= 0.0:
        k = (sd.veh.ess_max_kwh * kJ__kWh * ess_eff *
             mc_dis_eff * ess_traction_frac) / fc_eff
        equivalent_fuel_kJ = -1.0 * delta_soc * k
    else:
        k = (sd.veh.ess_max_kwh * kJ__kWh) / (ess_eff * mc_chg_eff * fc_eff)
        equivalent_fuel_kJ = -1.0 * delta_soc * k
    return sd.fuel_kj + equivalent_fuel_kJ
