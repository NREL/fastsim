"""Module containing classes and methods for simulating vehicle drive
cycle. For example usage, see ../README.md"""

# Import necessary python modules
from logging import debug
from typing import Optional
import numpy as np
import re
import copy

from .rustext import RUST_AVAILABLE

if RUST_AVAILABLE:
    import fastsimrust as fsr
from . import params, cycle, vehicle, inspect_utils

# these imports are needed for numba to type these correctly
from .vehicle import CONV, HEV, PHEV, BEV
from .vehicle import SI, ATKINSON, DIESEL, H2FC, HD_DIESEL


class SimDriveParams(object):
    """Class containing attributes used for configuring sim_drive.
    Usually the defaults are ok, and there will be no need to use this.

    See comments in code for descriptions of various parameters that
    affect simulation behavior. If, for example, you want to suppress
    warning messages, use the following pastable code EXAMPLE:

    >>> cyc = cycle.Cycle.from_file('udds')
    >>> veh = vehicle.Vehicle.from_vehdb(1)
    >>> sim_drive = simdrive.SimDriveClassic(cyc, veh)
    >>> sim_drive.sim_params.verbose = False # turn off error messages for large time steps
    >>> sim_drive.sim_drive()"""

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
        self.verbose = True  # show warning and other messages
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
        self.follow_allow = False
        # IDM - Intelligent Driver Model, Adaptive Cruise Control version
        self.idm_v_desired_m_per_s: float = 33.33
        self.idm_dt_headway_s: float = 1.0
        self.idm_minimum_gap_m: float = 2.0
        self.idm_delta: float = 4.0
        self.idm_accel_m_per_s2: float = 1.0
        self.idm_decel_m_per_s2: float = 1.5

        # EPA fuel economy adjustment parameters
        self.max_epa_adj = 0.3  # maximum EPA adjustment factor

    def to_rust(self):
        """Change to the Rust version"""
        return copy_sim_params(self, 'rust')


ref_sim_drive_params = SimDriveParams()


def copy_sim_params(sdp: SimDriveParams, return_type: str = None):
    """
    Returns copy of SimDriveParams.
    Arguments:
    sdp: instantianed SimDriveParams or RustSimDriveParams
    return_type: 
        default: infer from type of sdp
        'dict': dict
        'sim_params': SimDriveParams 
        TODO: 'legacy', NOT IMPLEMENTED - do we need it? 'legacy': LegacySimDriveParams 
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
            return_type = 'sim_params'
        # elif type(cyc) == LegacyCycle:
        #    return_type = "legacy"
        else:
            raise NotImplementedError(
                "Only implemented for rust, cycle, or legacy.")

    # if return_type == 'dict':
    #    return sdp_dict
    # elif return_type == 'sim_params':
    #    return SimDriveParams.from_dict(sdp_dict)
    # elif return_type == 'legacy':
    #    return LegacyCycle(cyc_dict)
    if return_type == 'dict':
        return sdp_dict
    elif return_type == 'sim_params':
        return SimDriveParams.from_dict(sdp_dict)
    elif RUST_AVAILABLE and return_type == 'rust':
        return fsr.RustSimDriveParams(**sdp_dict)
    else:
        raise ValueError(f"Invalid return_type: '{return_type}'")


def sim_params_equal(a: SimDriveParams, b: SimDriveParams, verbose: bool = False):
    """
    Returns True if objects are structurally equal (i.e., equal by value), else false.
    Arguments:
    a: instantiated SimDriveParams object
    b: instantiated SimDriveParams object
    verbose: bool, (optional, default: False), if True, prints out why not equal
    """
    if a is b:
        return True
    a_dict = copy_sim_params(a, 'dict')
    b_dict = copy_sim_params(b, 'dict')
    if len(a_dict) != len(b_dict):
        if verbose:
            a_keyset = {k for k in a.keys()}
            b_keyset = {k for k in b.keys()}
            print("KEY SETs NOT EQUAL")
            print(f"in a but not b: {a_keyset - b_keyset}")
            print(f"in b but not a: {b_keyset - a_keyset}")
        return False
    for k in a_dict.keys():
        if a_dict[k] != b_dict[k]:
            if verbose:
                print(f"UNEQUAL FOR KEY \"{k}\"")
                print(f"a['{k}'] = {repr(a_dict[k])}")
                print(f"b['{k}'] = {repr(b_dict[k])}")
            return False
    return True


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

        # Component Limits -- calculated dynamically
        self.cur_max_fs_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_trans_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_fs_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_max_kw_in = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_fc_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_cap_lim_dischg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_ess_max_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_avail_elec_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_cap_lim_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_ess_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_elec_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mc_elec_in_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mc_transi_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_mc_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_lim_mc_regen_perc_kw = np.zeros(
            self.cyc.len, dtype=np.float64)
        self.ess_lim_mc_regen_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_mech_mc_kw_in = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_trans_kw_out = np.zeros(self.cyc.len, dtype=np.float64)

        # Drive Train
        self.cyc_trac_kw_req = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_trac_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.spare_trac_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_whl_rad_per_sec = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.cyc_tire_inertia_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_whl_kw_req = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.regen_contrl_lim_kw_perc = np.zeros(
            self.cyc.len, dtype=np.float64)
        self.cyc_regen_brake_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_fric_brake_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_trans_kw_out_req = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_met = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.trans_kw_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.trans_kw_in_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_soc_target = np.zeros(self.cyc.len, dtype=np.float64)
        self.min_mc_kw_2help_fc = np.zeros(self.cyc.len, dtype=np.float64)
        self.mc_mech_kw_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.mc_elec_kw_in_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.aux_in_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.impose_coast = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.roadway_chg_kw_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.min_ess_kw_2help_fc = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_kw_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_kw_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_kw_out_ach_pct = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_kw_in_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.fs_kw_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.fs_cumu_mj_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.fs_kwh_out_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_cur_kwh = np.zeros(self.cyc.len, dtype=np.float64)
        self.soc = np.zeros(self.cyc.len, dtype=np.float64)

        # Vehicle Attributes, Control Variables
        self.regen_buff_soc = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.ess_regen_buff_dischg_kw = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.max_ess_regen_buff_chg_kw = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.ess_accel_buff_chg_kw = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.accel_buff_soc = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.max_ess_accell_buff_dischg_kw = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.ess_accel_regen_dischg_kw = np.zeros(
            self.cyc.len, dtype=np.float64)
        self.mc_elec_in_kw_for_max_fc_eff = np.zeros(
            self.cyc.len, dtype=np.float64)
        self.elec_kw_req_4ae = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.can_pwr_all_elec = np.array(  # oddball
            [False] * self.cyc.len, dtype=np.bool_)
        self.desired_ess_kw_out_for_ae = np.zeros(
            self.cyc.len, dtype=np.float64)
        self.ess_ae_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.er_ae_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_desired_kw_4fc_eff = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_kw_if_fc_req = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.cur_max_mc_elec_kw_in = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_kw_gap_fr_eff = np.zeros(self.cyc.len, dtype=np.float64)
        self.er_kw_if_fc_req = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.mc_elec_kw_in_if_fc_req = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.mc_kw_if_fc_req = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.fc_forced_on = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.fc_forced_state = np.zeros(self.cyc.len, dtype=np.int32)
        self.mc_mech_kw_4forced_fc = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_time_on = np.zeros(self.cyc.len, dtype=np.float64)
        self.prev_fc_time_on = np.zeros(self.cyc.len, dtype=np.float64)

        # Additional Variables
        self.mps_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.mph_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.dist_m = np.zeros(self.cyc.len, dtype=np.float64)  # oddbal
        self.dist_mi = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.high_acc_fc_on_tag = np.array(
            [False] * self.cyc.len, dtype=np.bool_)
        self.reached_buff = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.max_trac_mps = np.zeros(self.cyc.len, dtype=np.float64)
        self.add_kwh = np.zeros(self.cyc.len, dtype=np.float64)
        self.dod_cycs = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_perc_dead = np.zeros(
            self.cyc.len, dtype=np.float64)  # oddball
        self.drag_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_loss_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.accel_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ascent_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.rr_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_roadway_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.trace_miss_iters = np.zeros(self.cyc.len, dtype=np.float64)
        self.newton_iters = np.zeros(self.cyc.len, dtype=np.float64)

    @property
    def gap_to_lead_vehicle_m(self):
        "Provides the gap-with lead vehicle from start to finish"
        gaps_m = self.cyc0.dist_v2_m.cumsum() - self.cyc.dist_v2_m.cumsum()
        if self.sim_params.follow_allow:
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

        if init_soc is not None:
            if init_soc > 1.0 or init_soc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                init_soc = None

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
                print("WARNING! Provided aux_in_kw_override "
                      + "is not the right length; needs "
                      + f"{len(self.aux_in_kw)} elements")

        self.cyc_met[0] = True
        self.cur_soc_target[0] = self.veh.max_soc
        self.ess_cur_kwh[0] = init_soc * self.veh.ess_max_kwh
        self.soc[0] = init_soc
        self.mps_ach[0] = self.cyc0.mps[0]
        self.mph_ach[0] = self.cyc0.mph[0]

        if self.sim_params.missed_trace_correction:
            # reset the cycle in case it has been manipulated
            self.cyc = cycle.copy_cycle(self.cyc0)

        self.i = 1  # time step counter
        while self.i < len(self.cyc.time_s):
            self.sim_drive_step()

        if (self.cyc.dt_s > 5).any() and self.sim_params.verbose:
            if self.sim_params.missed_trace_correction:
                print('Max time dilation factor =',
                      (round((self.cyc.dt_s / self.cyc0.dt_s).max(), 3)))
            print("Warning: large time steps affect accuracy significantly.")
            print(
                "To suppress this message, view the doc string for simdrive.SimDriveParams.")
            print('Max time step =', (round(self.cyc.dt_s.max(), 3)))
        
        self.set_post_scalars()

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
            - a:
            - b:
        s = d_lead - d
        dv/dt = a * (1 - (v/v_desired)**delta - (s_desired(v,v-v_lead)/s)**2)
        s_desired(v, dv) = s0 + max(0, v*dt_headway + (v * dv)/(2.0 * sqrt(a*b)))
        REFERENCE:
        Treiber, Martin and Kesting, Arne. 2013. "Chapter 11: Car-Following Models Based on Driving Strategies".
            Traffic Flow Dynamics: Data, Models and Simulation. Springer-Verlag. Springer, Berlin, Heidelberg.
            DOI: https://doi.org/10.1007/978-3-642-32460-4.
        """
        # PARAMETERS
        delta = self.sim_params.idm_delta
        a_m__s2 = self.sim_params.idm_accel_m_per_s2  # acceleration (m/s2)
        b_m__s2 = self.sim_params.idm_decel_m_per_s2  # deceleration (m/s2)
        dt_headway_s = self.sim_params.idm_dt_headway_s
        # we assume vehicle's start out "minimum gap" apart
        s0_m = self.sim_params.idm_minimum_gap_m
        if self.sim_params.idm_v_desired_m_per_s > 0:
            v_desired_m__s = self.sim_params.idm_v_desired_m_per_s
        else:
            v_desired_m__s = self.cyc0.mps.max()
        # DERIVED VALUES
        sqrt_ab = (a_m__s2 * b_m__s2)**0.5
        v0_m__s = self.mps_ach[i-1]
        v0_lead_m__s = self.cyc0.mps[i-1]
        dv0_m__s = v0_m__s - v0_lead_m__s
        d0_lead_m = self.cyc0.dist_v2_m[:i].sum() + s0_m
        d0_m = self.cyc.dist_v2_m[:i].sum()
        s_m = max(d0_lead_m - d0_m, 0.01)
        # IDM EQUATIONS
        s_target_m = s0_m + \
            max(0.0, (v0_m__s * dt_headway_s) +
                ((v0_m__s * dv0_m__s)/(2.0 * sqrt_ab)))
        accel_target_m__s2 = a_m__s2 * \
            (1.0 - ((v0_m__s / v_desired_m__s) ** delta) - ((s_target_m / s_m)**2))
        self.cyc.mps[i] = max(
            v0_m__s + (accel_target_m__s2 * self.cyc.dt_s[i]), 0.0)

    def _set_speed_for_target_gap(self, i):
        """
        - i: non-negative integer, the step index
        RETURN: None
        EFFECTS:
        - sets the next speed (m/s)
        """
        self._set_speed_for_target_gap_using_idm(i)

    def sim_drive_step(self):
        """
        Step through 1 time step.
        TODO: create self.set_speed_for_target_gap(self.i):
        TODO: consider implementing for battery SOC dependence
        """
        if self.sim_params.coast_allow:
            self._set_coast_speed(self.i)
        if self.sim_params.follow_allow:
            self._set_speed_for_target_gap(self.i)
        self.solve_step(self.i)
        if self.sim_params.missed_trace_correction and (self.cyc0.dist_m[:self.i].sum() > 0):
            self.set_time_dilation(self.i)
        # TODO: shouldn't this below always get set whether we're coasting or following or not?
        if self.sim_params.coast_allow or self.sim_params.follow_allow:
            self.cyc.mps[self.i] = self.mps_ach[self.i]
            self.cyc.grade[self.i] = self.cyc0.average_grade_over_range(
                self.cyc.dist_m[:self.i].sum(), self.cyc.dt_s[self.i] * self.mps_ach[self.i])

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
            (self.veh.max_trac_mps2 * self.cyc.dt_s[i])

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
                (self.veh.fs_max_kw / self.veh.fs_secs_to_peak_pwr) * (self.cyc.dt_s[i])))
        # maximum fuel storage power output rate of change
        self.fc_trans_lim_kw[i] = self.fc_kw_out_ach[i-1] + (
            self.veh.fc_max_kw / self.veh.fc_sec_to_peak_pwr * self.cyc.dt_s[i]
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
                self.soc[i-1] - self.veh.min_soc) / self.cyc.dt_s[i]
        self.cur_ess_max_kw_out[i] = min(
            self.veh.ess_max_kw, self.ess_cap_lim_dischg_kw[i])

        if self.veh.ess_max_kwh == 0 or self.veh.ess_max_kw == 0:
            self.ess_cap_lim_chg_kw[i] = 0

        else:
            self.ess_cap_lim_chg_kw[i] = max(
                (self.veh.max_soc - self.soc[i-1]) * self.veh.ess_max_kwh * 1 / np.sqrt(self.veh.ess_round_trip_eff) /
                (self.cyc.dt_s[i] * 1 / 3.6e3),
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
            self.mc_mech_kw_out_ach[i-1]) + self.veh.mc_max_kw / self.veh.mc_sec_to_peak_pwr * self.cyc.dt_s[i]

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
            / (1 + self.veh.veh_cg_m * self.veh.wheel_coef_of_fric / self.veh.wheel_base_m) / 1_000 * self.max_trac_mps[i]
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
            mpsAch = self.mps_ach[i]
        else:
            mpsAch = self.cyc.mps[i]

        # TODO: use of self.cyc.mph[i] in regenContrLimKwPerc[i] calculation seems wrong. Shouldn't it be mpsAch or self.cyc0.mph[i]?

        # self.cyc.dist_v2_m.cumsum()[i-1]
        dist_traveled_m = self.cyc.dist_v2_m[:i].sum()
        grade = self.cyc0.average_grade_over_range(
            dist_traveled_m, mpsAch * self.cyc.dt_s[i])
        self.drag_kw[i] = 0.5 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * self.veh.frontal_area_m2 * (
            (self.mps_ach[i-1] + mpsAch) / 2.0) ** 3 / 1_000
        self.accel_kw[i] = self.veh.veh_kg / \
            (2.0 * self.cyc.dt_s[i]) * \
            (mpsAch ** 2 - self.mps_ach[i-1] ** 2) / 1_000
        self.ascent_kw[i] = self.props.a_grav_mps2 * np.sin(np.arctan(
            grade)) * self.veh.veh_kg * ((self.mps_ach[i-1] + mpsAch) / 2.0) / 1_000
        self.cyc_trac_kw_req[i] = self.drag_kw[i] + \
            self.accel_kw[i] + self.ascent_kw[i]
        self.spare_trac_kw[i] = self.cur_max_trac_kw[i] - \
            self.cyc_trac_kw_req[i]
        self.rr_kw[i] = self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef * np.cos(
            np.arctan(grade)) * (self.mps_ach[i-1] + mpsAch) / 2.0 / 1_000
        self.cyc_whl_rad_per_sec[i] = mpsAch / self.veh.wheel_radius_m
        self.cyc_tire_inertia_kw[i] = (
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * self.cyc_whl_rad_per_sec[i] ** 2.0 / self.cyc.dt_s[i] -
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels *
            (self.mps_ach[i-1] /
             self.veh.wheel_radius_m) ** 2.0 / self.cyc.dt_s[i]
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

            dist_traveled_m = self.cyc.dist_m[:i].sum()
            grade_estimate = self.cyc0.average_grade_over_range(
                dist_traveled_m, 0.0)
            grade_tol = 1e-6
            grade_diff = grade_tol + 1.0
            max_grade_iter = 3
            grade_iter = 0
            while grade_diff > grade_tol and grade_iter < max_grade_iter:
                grade_iter += 1
                grade = grade_estimate

                drag3 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * \
                    self.veh.drag_coef * self.veh.frontal_area_m2
                accel2 = 0.5 * self.veh.veh_kg / self.cyc.dt_s[i]
                drag2 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * \
                    self.veh.drag_coef * \
                    self.veh.frontal_area_m2 * self.mps_ach[i-1]
                wheel2 = 0.5 * self.veh.wheel_inertia_kg_m2 * \
                    self.veh.num_wheels / \
                    (self.cyc.dt_s[i] * self.veh.wheel_radius_m ** 2)
                drag1 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * \
                    self.veh.frontal_area_m2 * self.mps_ach[i-1] ** 2
                roll1 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef \
                    * np.cos(np.arctan(grade))
                ascent1 = 0.5 * self.props.a_grav_mps2 * \
                    np.sin(np.arctan(grade)) * self.veh.veh_kg
                accel0 = -0.5 * self.veh.veh_kg * \
                    self.mps_ach[i-1] ** 2 / self.cyc.dt_s[i]
                drag0 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * \
                    self.veh.frontal_area_m2 * self.mps_ach[i-1] ** 3
                roll0 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * \
                    self.veh.wheel_rr_coef * np.cos(np.arctan(grade)) \
                    * self.mps_ach[i-1]
                ascent0 = 0.5 * self.props.a_grav_mps2 * np.sin(np.arctan(grade)) \
                    * self.veh.veh_kg * self.mps_ach[i-1]
                wheel0 = -0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * \
                    self.mps_ach[i-1] ** 2 / \
                    (self.cyc.dt_s[i] * self.veh.wheel_radius_m ** 2)

                total3 = drag3 / 1_000
                total2 = (accel2 + drag2 + wheel2) / 1_000
                total1 = (drag1 + roll1 + ascent1) / 1_000
                total0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) / \
                    1_000 - self.cur_max_trans_kw_out[i]

                total = np.array([total3, total2, total1, total0])
                self.mps_ach[i] = newton_mps_estimate(total)
                grade_estimate = self.cyc0.average_grade_over_range(
                    dist_traveled_m, self.cyc.dt_s[i] * self.mps_ach[i])
                grade_diff = np.abs(grade - grade_estimate)
            self.set_power_calcs(i)

        self.mph_ach[i] = self.mps_ach[i] * params.MPH_PER_MPS
        self.dist_m[i] = self.mps_ach[i] * self.cyc.dt_s[i]
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
                0, (self.soc[i-1] - self.regen_buff_soc[i]) * self.veh.ess_max_kwh * 3_600 / self.cyc.dt_s[i]))

            self.max_ess_regen_buff_chg_kw[i] = min(max(
                0,
                (self.regen_buff_soc[i] - self.soc[i-1]) * self.veh.ess_max_kwh * 3.6e3 / self.cyc.dt_s[i]),
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
                0, (self.accel_buff_soc[i] - self.soc[i-1]) * self.veh.ess_max_kwh * 3.6e3 / self.cyc.dt_s[i])
            self.max_ess_accell_buff_dischg_kw[i] = min(
                max(
                    0,
                    (self.soc[i-1] - self.accel_buff_soc[i]) * self.veh.ess_max_kwh * 3_600 / self.cyc.dt_s[i]),
                self.cur_ess_max_kw_out[i]
            )

        if self.regen_buff_soc[i] < self.accel_buff_soc[i]:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(
                    (self.soc[i-1] - (self.regen_buff_soc[i] + self.accel_buff_soc[i]
                                      ) / 2) * self.veh.ess_max_kwh * 3.6e3 / self.cyc.dt_s[i],
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
        if self.prev_fc_time_on[i] > 0 and self.prev_fc_time_on[i] < self.veh.min_fc_time_on - self.cyc.dt_s[i]:
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
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s[i] /\
                3.6e3 * np.sqrt(self.veh.ess_round_trip_eff)

        else:
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s[i] / \
                3.6e3 * (1 / np.sqrt(self.veh.ess_round_trip_eff))

        if self.veh.ess_max_kwh == 0:
            self.soc[i] = 0.0

        else:
            self.soc[i] = self.ess_cur_kwh[i] / self.veh.ess_max_kwh

        if self.can_pwr_all_elec[i] and not(self.fc_forced_on[i]) and self.fc_kw_out_ach[i] == 0.0:
            self.fc_time_on[i] = 0
        else:
            self.fc_time_on[i] = self.fc_time_on[i-1] + self.cyc.dt_s[i]

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
            self.cyc.dt_s[i] * (1 / 3.6e3)

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
                    d_short[-1] / self.cyc0.dt_s[i] / self.mps_ach[i],
                    self.sim_params.min_time_dilation
                ),
                    self.sim_params.max_time_dilation
                )
            ]

            # add time dilation factor * step size to current and subsequent times
            self.cyc.time_s[i:] += self.cyc.dt_s[i] * t_dilation[-1]
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
            self.cyc.time_s[i:] += self.cyc.dt_s[i] * t_dilation[-1]

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
        atan_grade = float(np.arctan(grade))
        g = self.props.a_grav_mps2
        M = self.veh.veh_kg
        rho_CDFA = self.props.air_density_kg_per_m3 * \
            self.veh.drag_coef * self.veh.frontal_area_m2
        rrc = self.veh.wheel_rr_coef
        return -1.0 * (
            (g/v) * (np.sin(atan_grade) + rrc * np.cos(atan_grade))
            + (0.5 * rho_CDFA * (1.0/M) * v)
        )

    def _calc_distance_to_stop_coast_v2(self, i):
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
        TOL = 1e-6
        NOT_FOUND = -1.0
        v0 = self.cyc.mps[i-1]
        v_brake = self.sim_params.coast_brake_start_speed_m_per_s
        a_brake = self.sim_params.coast_brake_accel_m_per_s2
        ds = self.cyc0.dist_v2_m.cumsum()
        gs = self.cyc0.grade
        d0 = ds[i-1]
        ds_mask = ds >= d0
        dt_s = self.cyc0.dt_s[i]
        distances_m = ds[ds_mask] - d0
        grade_by_distance = gs[ds_mask]
        veh_mass_kg = self.veh.veh_kg
        air_density_kg__m3 = self.props.air_density_kg_per_m3
        CDFA_m2 = self.veh.drag_coef * self.veh.frontal_area_m2
        rrc = self.veh.wheel_rr_coef
        gravity_m__s2 = self.props.a_grav_mps2
        v = v0
        # distance traveled while stopping via friction-braking (i.e., distance to brake)
        dtb = -0.5 * v_brake * v_brake / a_brake
        if v0 <= v_brake:
            return -0.5 * v0 * v0 / a_brake
        d = 0.0
        d_max = distances_m[-1]
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
        i = 0
        MAX_ITER = 2000
        ITERS_PER_STEP = 2
        while v > v_brake and v >= 0.0 and d <= d_max and i < MAX_ITER:
            gr = unique_grade if unique_grade is not None else self.cyc0.average_grade_over_range(
                d0, d)
            k = self._calc_dvdd(v, gr)
            v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s)
            vavg = 0.5 * (v + v_next)
            for i in range(ITERS_PER_STEP):
                k = self._calc_dvdd(vavg, gr)
                v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s)
                vavg = 0.5 * (v + v_next)
            if k >= 0.0 and unique_grade is not None:
                # there is no solution for coastdown -- speed will never decrease
                return NOT_FOUND
            stop = v_next <= v_brake
            if stop:
                v_next = v_brake
            vavg = 0.5 * (v + v_next)
            dd = vavg * dt_s
            d += dd
            v = v_next
            i += 1
            if stop:
                break
        if np.abs(v - v_brake) < TOL:
            return d + dtb
        return NOT_FOUND

    def _should_impose_coast(self, i):
        """
        - i: non-negative integer, the current position in cyc
        - verbose: Bool, if True, prints out debug information
        RETURN: Bool if vehicle should initiate coasting
        Coast logic is that the vehicle should coast if it is within coasting distance of a stop:
        - if distance to coast from start of step is <= distance to next stop
        - AND distance to coast from end of step (using prescribed speed) is > distance to next stop
        """
        if self.sim_params.coast_start_speed_m_per_s > 0.0:
            return self.cyc.mps[i] >= self.sim_params.coast_start_speed_m_per_s
        d0 = self.cyc0.dist_v2_m[:i].sum()
        # distance to stop by coasting from start of step (i-1)
        #dtsc0 = calc_distance_to_stop_coast(v0, dvdd, brake_start_speed_m__s, brake_accel_m__s2)
        dtsc0 = self._calc_distance_to_stop_coast_v2(i)
        if dtsc0 < 0.0:
            return False
        # distance to next stop (m)
        dts0 = self.cyc0.calc_distance_to_next_stop_from(d0)
        return dtsc0 >= dts0

    def _calc_next_rendezvous_trajectory(self, i, min_accel_m__s2, max_accel_m__s2):
        """
        Calculate next rendezvous trajectory.
        - i: non-negative integer, the index into cyc for the start-of-step
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
        brake_start_speed_m__s = self.sim_params.coast_brake_start_speed_m_per_s
        brake_accel_m__s2 = self.sim_params.coast_brake_accel_m_per_s2
        time_horizon_s = self.sim_params.coast_time_horizon_for_adjustment_s
        # distance_horizon_m = 1000.0
        not_found_n = 0
        not_found_jerk_m__s3 = 0.0
        not_found_accel_m__s2 = 0.0
        not_found = (False, not_found_n, not_found_jerk_m__s3,
                     not_found_accel_m__s2)
        if v0 < (brake_start_speed_m__s + 1e-6):
            # don't process braking
            return not_found
        if min_accel_m__s2 > max_accel_m__s2:
            min_accel_m__s2, max_accel_m__s2 = max_accel_m__s2, min_accel_m__s2
        num_samples = len(self.cyc.mps)
        d0 = self.cyc.dist_v2_m[:i].sum()
        # a_proposed = (v1 - v0) / dt
        # distance to stop from start of time-step
        dts0 = self.cyc0.calc_distance_to_next_stop_from(d0)
        if dts0 < 1e-6:
            # no stop to coast towards or we're there...
            return not_found
        v1 = self.cyc.mps[i]
        dt = self.cyc.dt_s[i]
        # distance to brake from the brake start speed (m/s)
        dtb = -0.5 * brake_start_speed_m__s * brake_start_speed_m__s / brake_accel_m__s2
        # distance to brake initiation from start of time-step (m)
        dtbi0 = dts0 - dtb
        cyc0_distances_m = self.cyc0.dist_v2_m.cumsum()
        # Now, check rendezvous trajectories
        if time_horizon_s > 0.0:
            step_idx = i
            dt_plan = 0.0
            r_best_found = False
            r_best_n = 0
            r_best_jerk_m__s3 = 0.0
            r_best_accel_m__s2 = 0.0
            r_best_accel_spread_m__s2 = 0.0
            while dt_plan <= time_horizon_s and step_idx < num_samples:
                dt_plan += self.cyc0.dt_s[step_idx]
                step_ahead = step_idx - (i - 1)
                if step_ahead == 1:
                    # for brake init rendezvous
                    accel = (brake_start_speed_m__s - v0) / dt
                    v1 = max(0.0, v0 + accel * dt)
                    dd_proposed = ((v0 + v1) / 2.0) * dt
                    if np.abs(v1 - brake_start_speed_m__s) < 1e-6 and np.abs(dtbi0 - dd_proposed) < 1e-6:
                        r_best_found = True
                        r_best_n = 1
                        r_best_accel_m__s2 = accel
                        r_best_jerk_m__s3 = 0.0
                        r_best_accel_spread_m__s2 = 0.0
                        break
                else:
                    if dtbi0 > 1e-6:
                        # rendezvous trajectory for brake-start -- assumes fixed time-steps
                        r_bi_jerk_m__s3, r_bi_accel_m__s2 = cycle.calc_constant_jerk_trajectory(
                            step_ahead, 0.0, v0, dtbi0, brake_start_speed_m__s, dt)
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

    def _set_coast_speed(self, i):
        """
        Placeholder for method to impose coasting.
        Might be good to include logic for deciding when to coast.
        Solve for the next-step speed that will yield a zero roadload
        """
        TOL = 1e-6
        v0 = self.mps_ach[i-1]
        if v0 < TOL:
            # TODO: need to determine how to leave coast and rejoin shadow trace
            #self.impose_coast[i] = False
            pass
        else:
            self.impose_coast[i] = self.impose_coast[i -
                                                     1] or self._should_impose_coast(i)

        if not self.impose_coast[i]:
            return
        v1_traj = self.cyc.mps[i]
        if self.cyc.mps[i] == self.cyc0.mps[i] and v0 > self.sim_params.coast_brake_start_speed_m_per_s:
            if self.sim_params.coast_allow_passing:
                # we could be coasting downhill so could in theory go to a higher speed
                # since we can pass, allow vehicle to go up to max coasting speed (m/s)
                # the solver will show us what we can actually achieve
                self.cyc.mps[i] = self.sim_params.coast_max_speed_m_per_s
            else:
                # distances of lead vehicle (m)
                ds_lv = self.cyc0.dist_m.cumsum()
                # current distance traveled at start of step
                d0 = self.cyc.dist_v2_m[:i].sum()
                d1_lv = ds_lv[i]
                max_step_distance_m = d1_lv - d0
                max_avg_speed_m__s = max_step_distance_m / self.cyc0.dt_s[i]
                max_next_speed_m__s = 2 * max_avg_speed_m__s - v0
                self.cyc.mps[i] = max(
                    0, min(max_next_speed_m__s, self.sim_params.coast_max_speed_m_per_s))
        # Solve for the actual coasting speed
        self.solve_step(i)
        self.newton_iters[i] = 0  # reset newton iters
        self.cyc.mps[i] = self.mps_ach[i]
        accel_proposed = (self.cyc.mps[i] -
                          self.cyc.mps[i-1]) / self.cyc.dt_s[i]
        if self.cyc.mps[i] < TOL:
            self.cyc.mps[i] = 0.0
            return
        if np.abs(self.cyc.mps[i] - v1_traj) > TOL:
            adjusted_current_speed = False
            if self.cyc.mps[i] < (self.sim_params.coast_brake_start_speed_m_per_s + TOL):
                v1_before = self.cyc.mps[i]
                self.cyc.modify_with_braking_trajectory(
                    self.sim_params.coast_brake_accel_m_per_s2, i)
                v1_after = self.cyc.mps[i]
                assert v1_before != v1_after
                adjusted_current_speed = True
            else:
                traj_found, traj_n, traj_jerk_m__s3, traj_accel_m__s2 = self._calc_next_rendezvous_trajectory(
                    i,
                    min_accel_m__s2=self.sim_params.coast_brake_accel_m_per_s2,
                    max_accel_m__s2=min(accel_proposed, 0.0)
                )
                if traj_found:
                    # adjust cyc to perform the trajectory
                    final_speed_m__s = self.cyc.modify_by_const_jerk_trajectory(
                        i, traj_n, traj_jerk_m__s3, traj_accel_m__s2)
                    adjusted_current_speed = True
                    if np.abs(final_speed_m__s - self.sim_params.coast_brake_start_speed_m_per_s) < 0.1:
                        i_for_brake = i + traj_n
                        self.cyc.modify_with_braking_trajectory(
                            self.sim_params.coast_brake_accel_m_per_s2,
                            i_for_brake,
                        )
                        adjusted_current_speed = True
            if adjusted_current_speed:
                # TODO: should we have an iterate=True/False, flag on solve_step to
                # ensure newton iters is reset? vs having to call manually?
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
        self.battery_kwh_per_mi = (
            self.ess_dischg_kj / 3.6e3) / self.dist_mi.sum()
        self.electric_kwh_per_mi = (
            (self.roadway_chg_kj + self.ess_dischg_kj) / 3.6e3) / self.dist_mi.sum()
        self.fuel_kj = (self.fs_kw_out_ach * self.cyc.dt_s).sum()

        if (self.fuel_kj + self.roadway_chg_kj) == 0:
            self.ess2fuel_kwh = 1.0

        else:
            self.ess2fuel_kwh = self.ess_dischg_kj / \
                (self.fuel_kj + self.roadway_chg_kj)

        # energy audit calcs
        self.drag_kw = self.drag_kw
        self.drag_kj = (self.drag_kw * self.cyc.dt_s).sum()
        self.ascent_kw = self.ascent_kw
        self.ascent_kj = (self.ascent_kw * self.cyc.dt_s).sum()
        self.rr_kw = self.rr_kw
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

        if (np.abs(self.energy_audit_error) > self.sim_params.energy_audit_error_tol) and \
                self.sim_params.verbose:
            print('Warning: There is a problem with conservation of energy.')
            print('Energy Audit Error:', np.round(self.energy_audit_error, 5))

        self.accel_kw[1:] = (self.veh.veh_kg / (2.0 * (self.cyc.dt_s[1:]))) * (
            self.mps_ach[1:] ** 2 - self.mps_ach[:-1] ** 2) / 1_000

        self.trace_miss = False
        self.trace_miss_dist_frac = abs(
            self.dist_m.sum() - self.cyc0.dist_m.sum()) / self.cyc0.dist_m.sum()
        self.trace_miss_time_frac = abs(
            self.cyc.time_s[-1] - self.cyc0.time_s[-1]) / self.cyc0.time_s[-1]

        if not(self.sim_params.missed_trace_correction):
            if self.trace_miss_dist_frac > self.sim_params.trace_miss_dist_tol:
                self.trace_miss = True
                if self.sim_params.verbose:
                    print('Warning: Trace miss distance fraction:',
                          np.round(self.trace_miss_dist_frac, 5))
                    print('exceeds tolerance of: ', np.round(
                        self.sim_params.trace_miss_dist_tol, 5))
        else:
            if self.trace_miss_time_frac > self.sim_params.trace_miss_time_tol:
                self.trace_miss = True
                if self.sim_params.verbose:
                    print('Warning: Trace miss time fraction:',
                          np.round(self.trace_miss_time_frac, 5))
                    print('exceeds tolerance of: ', np.round(
                        self.sim_params.trace_miss_time_tol, 5))

        # NOTE: I believe this should be accessing self.cyc0.mps[i] instead of self.cyc.mps[i]; self.cyc may be modified...
        self.trace_miss_speed_mps = max([
            abs(self.mps_ach[i] - self.cyc.mps[i]) for i in range(len(self.cyc.time_s))
        ])
        if self.trace_miss_speed_mps > self.sim_params.trace_miss_speed_mps_tol:
            self.trace_miss = True
            if self.sim_params.verbose:
                print('Warning: Trace miss speed [m/s]:',
                      np.round(self.trace_miss_speed_mps, 5))
                print('exceeds tolerance of: ', np.round(
                    self.sim_params.trace_miss_speed_mps_tol, 5))

    def to_rust(self):
        "Create a rust version of SimDrive"
        return copy_sim_drive(self, 'rust', True)


if RUST_AVAILABLE:

    def RustSimDrive(cyc: fsr.RustCycle, veh: fsr.RustVehicle) -> SimDrive:
        """
        Wrapper function to make SimDriveRust look like SimDrive for language server.
        Arguments:
        ----------
        cyc: cycle.Cycle instance
        veh: vehicle.Vehicle instance"""
        return fsr.RustSimDrive(cyc, veh)

else:

    def RustSimDrive(cyc: cycle.Cycle, veh: vehicle.Vehicle) -> SimDrive:
        """
        Wrapper function to make SimDriveRust look like SimDrive for language server.
        Arguments:
        ----------
        cyc: cycle.Cycle instance
        veh: vehicle.Vehicle instance"""
        raise ImportError(
            "FASTSimRust does not seem to be available. Cannot instantiate RustSimDrive..."
        )
        return SimDrive(cyc, veh)


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
        'sim_drive': Cycle 
        'legacy': LegacyCycle
        'rust': RustCycle
    deep: if True, uses deepcopy on everything
    """

    # TODO: if the rust version is input, make sure to copy lists to numpy arrays
    # TODO: no need to implement dict for copy_sim_drive, but please do for the subordinate classes

    if return_type is None:
        # if type(sd) == fsr.RustSimDrive:
        #    return_type = 'rust'
        if type(sd) == SimDrive:
            return_type = 'sim_drive'
        elif type(sd) == fsr.RustSimDrive:
            return_type = 'rust'
        elif type(sd) == LegacySimDrive:
            return_type = "legacy"
        else:
            raise NotImplementedError(
                "Only implemented for rust, sim_drive, or legacy.")

    cyc_return_type = 'cycle' if return_type == 'sim_drive' else return_type
    veh_return_type = 'vehicle' if return_type == 'sim_drive' else return_type
    cyc = cycle.copy_cycle(sd.cyc0, cyc_return_type, deep)
    veh = vehicle.copy_vehicle(sd.veh, veh_return_type, deep)

    if return_type == 'rust':
        return fsr.RustSimDrive(cyc, veh)

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
            sp_return_type = 'sim_params' if (
                return_type == 'sim_drive' or return_type == 'legacy') else return_type
            sd_copy.sim_params = copy_sim_params(sd.sim_params, sp_return_type)
        elif key == 'props':
            pp_return_type = 'physical_properties' if (
                return_type == 'sim_drive' or return_type == 'legacy') else return_type
            sd_copy.props = params.copy_physical_properties(
                sd.props, pp_return_type)
        else:
            # should be ok to deep copy
            val = sd.__getattribute__(key)
            sd_copy.__setattr__(key, copy.deepcopy(val) if deep else val)

    return sd_copy


def sim_drive_equal(a: SimDrive, b: SimDrive, verbose=False) -> bool:
    ""
    if a is b:
        return True
    for k in ref_sim_drive.__dict__.keys():
        a_val = a.__getattribute__(k)
        b_val = b.__getattribute__(k)
        if k == 'cyc' or k == 'cyc0':
            if not cycle.cyc_equal(a_val, b_val):
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif k == 'veh':
            if not vehicle.veh_equal(a_val, b_val):
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif k == 'props':
            if not params.physical_properties_equal(a_val, b_val):
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif k == 'sim_params':
            if not sim_params_equal(a_val, b_val):
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif 'to_list' in a_val.__dir__() + b_val.__dir__():
            if 'to_list' in a_val.__dir__():
                a_val = np.array(a_val.to_list())
            if 'to_list' in b_val.__dir__():
                b_val = np.array(b_val.to_list())
            if not (a_val == b_val).all():
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif type(a_val) == np.ndarray or type(b_val) == np.ndarray:
            if not (a_val == b_val).all():
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                print('got here')
                return False
        elif type(a_val) == list and type(b_val) == list:
            if not a_val == b_val:
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif a_val != b_val:
            if verbose:
                print(f"unequal at key {k}: {a_val} != {b_val}")
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
            '\w*_kw(?!h)\w*', var)]
        # find all vars containing 'Kw' but not 'Kwh'

        prog = re.compile('(\w*)_kw(?!h)(\w*)')
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
                   '_pos'] = np.trapz(tempvars[var + '_pos'], self.cyc.time_s)
            output[search[1] + '_kj' + search[2] +
                   '_neg'] = np.trapz(tempvars[var + '_neg'], self.cyc.time_s)

        output['dist_miles_final'] = sum(self.dist_mi)
        if sum(self.fs_kwh_out_ach) > 0:
            output['mpgge'] = sum(
                self.dist_mi) / sum(self.fs_kwh_out_ach) * self.props.kwh_per_gge
        else:
            output['mpgge'] = 0

        return output

    def set_battery_wear(self):
        """Battery wear calcs"""

        self.add_kwh[1:] = np.array([
            (self.ess_cur_kwh[i] - self.ess_cur_kwh[i-1]) + self.add_kwh[i-1]
            if self.ess_cur_kwh[i] > self.ess_cur_kwh[i-1]
            else 0
            for i in range(1, len(self.ess_cur_kwh))])

        if self.veh.ess_max_kwh == 0:
            self.dod_cycs[1:] = np.array(
                [0.0 for i in range(1, len(self.ess_cur_kwh))])
        else:
            self.dod_cycs[1:] = np.array([
                self.add_kwh[i-1] / self.veh.ess_max_kwh if self.add_kwh[i] == 0
                else 0
                for i in range(1, len(self.ess_cur_kwh))])

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
                   sd.mc_elec_kw_in_ach, sd.mc_mech_kw_out_ach, sd.veh.mc_peak_eff)
    mc_dis_eff = f(np.array(sd.mc_mech_kw_out_ach) > 0.0,
                   sd.mc_mech_kw_out_ach, sd.mc_elec_kw_in_ach, mc_chg_eff)
    ess_traction_frac = f(np.array(sd.mc_elec_kw_in_ach)
                          > 0.0, sd.mc_elec_kw_in_ach, sd.ess_kw_out_ach, 1.0)
    fc_eff = f(
        np.array(sd.trans_kw_in_ach) > 0.0,
        sd.fc_kw_out_ach,
        sd.fc_kw_in_ach,
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
