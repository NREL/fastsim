"""Module containing classes and methods for simulating vehicle drive
cycle. For example usage, see ../README.md"""

### Import necessary python modules
from logging import debug
import numpy as np
import re
import copy

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
        self.max_trace_miss_iters = 5 # number of iterations to achieve time dilation correction
        self.trace_miss_speed_mps_tol = 1.0 # threshold of error in speed [m/s] that triggers warning
        self.trace_miss_time_tol = 1e-3 # threshold for printing warning when time dilation is active
        self.trace_miss_dist_tol = 1e-3 # threshold of fractional eror in distance that triggers warning
        self.sim_count_max = 30  # max allowable number of HEV SOC iterations
        self.verbose = True  # show warning and other messages
        self.newton_gain = 0.9 # newton solver gain
        self.newton_max_iter = 100 # newton solver max iterations
        self.newton_xtol = 1e-9 # newton solver tolerance
        self.energy_audit_error_tol = 0.002 # tolerance for energy audit error warning, i.e. 0.1%
                
        # EPA fuel economy adjustment parameters
        self.max_epa_adj = 0.3 # maximum EPA adjustment factor

    def to_rust(self):
        """Change to the Rust version"""
        return copy_sim_params(self, 'rust')

ref_sim_drive_params = SimDriveParams()

def copy_sim_params(sdp: SimDriveParams, return_type:str=None):
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
        if type(sdp) == fsr.RustSimDriveParams:
            return_type = 'rust'
        elif type(sdp) == SimDriveParams:
            return_type = 'sim_params'
        #elif type(cyc) == LegacyCycle:
        #    return_type = "legacy"
        else:
            raise NotImplementedError(
                "Only implemented for rust, cycle, or legacy.")

    #if return_type == 'dict':
    #    return sdp_dict
    #elif return_type == 'sim_params':
    #    return SimDriveParams.from_dict(sdp_dict)
    #elif return_type == 'legacy':
    #    return LegacyCycle(cyc_dict)
    if return_type == 'dict':
        return sdp_dict
    elif return_type == 'sim_params':
        return SimDriveParams.from_dict(sdp_dict)
    elif return_type == 'rust':
        return fsr.RustSimDriveParams(**sdp_dict)
    else:
        raise ValueError(f"Invalid return_type: '{return_type}'")

def sim_params_equal(a:SimDriveParams, b:SimDriveParams):
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
        return False
    for k in a_dict.keys():
        if a_dict[k] != b_dict[k]:
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
        self.cyc = cycle.copy_cycle(cyc) # this cycle may be manipulated
        self.cyc0 = cycle.copy_cycle(cyc) # this cycle is not to be manipulated
        self.sim_params = SimDriveParams()
        self.props = params.PhysicalProperties()

    def init_arrays(self):
        self.i = 1 # initialize step counter for possible use outside sim_drive_walk()

        # Component Limits -- calculated dynamically
        self.cur_max_fs_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_trans_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_fs_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_max_kw_in = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_fc_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_cap_lim_dischg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_ess_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_avail_elec_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_cap_lim_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_ess_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_elec_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mc_elec_in_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mc_transi_lim_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_mc_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_lim_mc_regen_perc_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_lim_mc_regen_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_mech_mc_kw_in = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_trans_kw_out = np.zeros(self.cyc.len, dtype=np.float64)

        ### Drive Train
        self.cyc_drag_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_accel_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_ascent_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_trac_kw_req = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_trac_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.spare_trac_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_rr_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_whl_rad_per_sec = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.cyc_tire_inertia_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cyc_whl_kw_req = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.regen_contrl_lim_kw_perc = np.zeros(self.cyc.len, dtype=np.float64)
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
        self.regen_buff_soc = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.ess_regen_buff_dischg_kw = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.max_ess_regen_buff_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.ess_accel_buff_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.accel_buff_soc = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.max_ess_accell_buff_dischg_kw = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.ess_accel_regen_dischg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.mc_elec_in_kw_for_max_fc_eff = np.zeros(self.cyc.len, dtype=np.float64)
        self.elec_kw_req_4ae = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.can_pwr_all_elec = np.array(  # oddball
            [False] * self.cyc.len, dtype=np.bool_)
        self.desired_ess_kw_out_for_ae = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_ae_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.er_ae_kw_out = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_desired_kw_4fc_eff = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_kw_if_fc_req = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.cur_max_mc_elec_kw_in = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_kw_gap_fr_eff = np.zeros(self.cyc.len, dtype=np.float64)
        self.er_kw_if_fc_req = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.mc_elec_kw_in_if_fc_req = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.mc_kw_if_fc_req = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.fc_forced_on = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.fc_forced_state = np.zeros(self.cyc.len, dtype=np.int32)
        self.mc_mech_kw_4forced_fc = np.zeros(self.cyc.len, dtype=np.float64)
        self.fc_time_on = np.zeros(self.cyc.len, dtype=np.float64)
        self.prev_fc_time_on = np.zeros(self.cyc.len, dtype=np.float64)

        ### Additional Variables
        self.mps_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.mph_ach = np.zeros(self.cyc.len, dtype=np.float64)
        self.dist_m = np.zeros(self.cyc.len, dtype=np.float64)  # oddbal
        self.dist_mi = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.high_acc_fc_on_tag = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.reached_buff = np.array([False] * self.cyc.len, dtype=np.bool_)
        self.max_trac_mps = np.zeros(self.cyc.len, dtype=np.float64)
        self.add_kwh = np.zeros(self.cyc.len, dtype=np.float64)
        self.dod_cycs = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_perc_dead = np.zeros(self.cyc.len, dtype=np.float64)  # oddball
        self.drag_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ess_loss_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.accel_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.ascent_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.rr_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.cur_max_roadway_chg_kw = np.zeros(self.cyc.len, dtype=np.float64)
        self.trace_miss_iters = np.zeros(self.cyc.len, dtype=np.float64)
        self.newton_iters = np.zeros(self.cyc.len, dtype=np.float64)

    def sim_drive(self, init_soc=-1, aux_in_kw_override=np.zeros(1, dtype=np.float64)):
        """
        Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        Arguments
        ------------
        init_soc: initial SOC for electrified vehicles.  
        aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
            Default of np.zeros(1) causes veh.aux_kw to be used.
            If zero is actually desired as an override, either set
            veh.aux_kw = 0 before instantiaton of SimDrive*, or use
            `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting the
            final value to non-zero prevents override mechanism.  
        """

        if (aux_in_kw_override == 0).all():
            aux_in_kw_override = self.aux_in_kw
        self.hev_sim_count = 0

        if init_soc != -1:
            if init_soc > 1.0 or init_soc < 0.0:
                print('Must enter a valid initial SOC between 0.0 and 1.0')
                print('Running standard initial SOC controls')
                init_soc = None

        elif self.veh.veh_pt_type == CONV:  # Conventional
            # If no EV / Hybrid components, no SOC considerations.
            init_soc = (self.veh.max_soc + self.veh.min_soc) / 2.0

        elif self.veh.veh_pt_type == HEV and init_soc == -1:  # HEV
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
                roadway_chg_kj = np.sum(self.roadway_chg_kw_out_ach * self.cyc.dt_s)
                if (fuel_kj + roadway_chg_kj) > 0:
                    ess_2fuel_kwh = np.abs(
                        (self.soc[0] - self.soc[-1]) * self.veh.max_ess_kwh * 3_600 / (fuel_kj + roadway_chg_kj)
                    )
                else:
                    ess_2fuel_kwh = 0.0
                init_soc = min(1.0, max(0.0, self.soc[-1]))

        elif (self.veh.veh_pt_type == PHEV and init_soc == -1) or (self.veh.veh_pt_type == BEV and init_soc == -1):  # PHEV and BEV
            # If EV, initializing initial SOC to maximum SOC.
            init_soc = self.veh.max_soc

        self.sim_drive_walk(init_soc, aux_in_kw_override)

        self.set_post_scalars()

    def sim_drive_walk(self, init_soc, aux_in_kw_override=np.zeros(1, dtype=np.float64)):
        """
        Receives second-by-second cycle information, vehicle properties, 
        and an initial state of charge and runs sim_drive_step to perform a 
        backward facing powertrain simulation. Method 'sim_drive' runs this
        iteratively to achieve correct SOC initial and final conditions, as 
        needed.

        Arguments
        ------------
        init_soc (optional): initial battery state-of-charge (SOC) for electrified vehicles
        auxInKw: auxInKw override.  Array of same length as cyc.time_s.  
                Default of np.zeros(1) causes veh.aux_kw to be used. If zero is actually
                desired as an override, either set veh.aux_kw = 0 before instantiaton of
                SimDrive*, or use `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting
                the final value to non-zero prevents override mechanism.  
        """
        
        ############################
        ###   Loop Through Time  ###
        ############################

        ###  Assign First Values  ###
        ### Drive Train
        self.init_arrays() # reinitialize arrays for each new run
        if not((aux_in_kw_override == 0).all()):
            self.aux_in_kw = aux_in_kw_override
        
        self.cyc_met[0] = True
        self.cur_soc_target[0] = self.veh.max_soc
        self.ess_cur_kwh[0] = init_soc * self.veh.max_ess_kwh
        self.soc[0] = init_soc
        self.mps_ach[0] = self.cyc0.mps[0]
        self.mph_ach[0] = self.cyc0.mph[0]

        if self.sim_params.missed_trace_correction:
            self.cyc = cycle.copy_cycle(self.cyc0) # reset the cycle in case it has been manipulated

        self.i = 1 # time step counter
        while self.i < len(self.cyc.time_s):
            self.sim_drive_step()

        if (self.cyc.dt_s > 5).any() and self.sim_params.verbose:
            if self.sim_params.missed_trace_correction:
                print('Max time dilation factor =', (round((self.cyc.dt_s / self.cyc0.dt_s).max(), 3)))
            print("Warning: large time steps affect accuracy significantly.") 
            print("To suppress this message, view the doc string for simdrive.SimDriveParams.")
            print('Max time step =', (round(self.cyc.dt_s.max(), 3)))

    def sim_drive_step(self):
        """Step through 1 time step."""
        self.solve_step(self.i)
        if self.sim_params.missed_trace_correction and (self.cyc0.dist_m[:self.i].sum() > 0):
            self.set_time_dilation(self.i)
        # TODO: implement something for coasting here
        # if self.impose_coast[i] == True
            # self.set_coast_speeed(i)

        self.i += 1 # increment time step counter
    
    def solve_step(self, i):
        """Perform all the calculations to solve 1 time step."""
        self.set_misc_calcs(i)
        self.set_comp_lims(i)
        self.set_power_calcs(i)
        self.set_ach_speed(i)
        self.set_hybrid_cont_calcs(i)
        self.set_fc_forced_state(i) # can probably be *mostly* done with list comprehension in post processing
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
        self.max_trac_mps[i] = self.mps_ach[i-1] + (self.veh.max_trac_mps2 * self.cyc.dt_s[i])

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
            self.veh.max_fuel_stor_kw, 
            self.fs_kw_out_ach[i-1] + (
                (self.veh.max_fuel_stor_kw / self.veh.fuel_stor_secs_to_peak_pwr) * (self.cyc.dt_s[i])))
        # maximum fuel storage power output rate of change
        self.fc_trans_lim_kw[i] = self.fc_kw_out_ach[i-1] + (
            self.veh.max_fuel_conv_kw / self.veh.fuel_conv_secs_to_peak_pwr * self.cyc.dt_s[i]
        )

        self.fc_max_kw_in[i] = min(self.cur_max_fs_kw_out[i], self.veh.max_fuel_stor_kw)
        self.fc_fs_lim_kw[i] = self.veh.fc_max_out_kw
        self.cur_max_fc_kw_out[i] = min(
            self.veh.max_fuel_conv_kw, self.fc_fs_lim_kw[i], self.fc_trans_lim_kw[i])

        if self.veh.max_ess_kwh == 0 or self.soc[i-1] < self.veh.min_soc:
            self.ess_cap_lim_dischg_kw[i] = 0.0

        else:
            self.ess_cap_lim_dischg_kw[i] = self.veh.max_ess_kwh * np.sqrt(self.veh.ess_round_trip_eff) * 3.6e3 * (
                self.soc[i-1] - self.veh.min_soc) / self.cyc.dt_s[i]
        self.cur_max_ess_kw_out[i] = min(
            self.veh.max_ess_kw, self.ess_cap_lim_dischg_kw[i])

        if self.veh.max_ess_kwh == 0 or self.veh.max_ess_kw == 0:
            self.ess_cap_lim_chg_kw[i] = 0

        else:
            self.ess_cap_lim_chg_kw[i] = max(
                (self.veh.max_soc - self.soc[i-1]) * self.veh.max_ess_kwh * 1 / np.sqrt(self.veh.ess_round_trip_eff) / 
                (self.cyc.dt_s[i] * 1 / 3.6e3), 
                0
            )

        self.cur_max_ess_chg_kw[i] = min(self.ess_cap_lim_chg_kw[i], self.veh.max_ess_kw)

        # Current maximum electrical power that can go toward propulsion, not including motor limitations
        if self.veh.fc_eff_type == H2FC:
            self.cur_max_elec_kw[i] = self.cur_max_fc_kw_out[i] + self.cur_max_roadway_chg_kw[i] + self.cur_max_ess_kw_out[i] - self.aux_in_kw[i]

        else:
            self.cur_max_elec_kw[i] = self.cur_max_roadway_chg_kw[i] + self.cur_max_ess_kw_out[i] - self.aux_in_kw[i]

        # Current maximum electrical power that can go toward propulsion, including motor limitations
        self.cur_max_avail_elec_kw[i] = min(self.cur_max_elec_kw[i], self.veh.mc_max_elec_in_kw)

        if self.cur_max_elec_kw[i] > 0:
            # limit power going into e-machine controller to
            if self.cur_max_avail_elec_kw[i] == max(self.veh.mc_kw_in_array):
                self.mc_elec_in_lim_kw[i] = min(self.veh.mc_kw_out_array[-1], self.veh.max_motor_kw)
            else:
                self.mc_elec_in_lim_kw[i] = min(
                    self.veh.mc_kw_out_array[
                        np.argmax(self.veh.mc_kw_in_array > min(
                            max(self.veh.mc_kw_in_array) - 0.01, 
                            self.cur_max_avail_elec_kw[i]
                        )) - 1],
                    self.veh.max_motor_kw)
        else:
            self.mc_elec_in_lim_kw[i] = 0.0

        # Motor transient power limit
        self.mc_transi_lim_kw[i] = abs(
            self.mc_mech_kw_out_ach[i-1]) + self.veh.max_motor_kw / self.veh.motor_secs_to_peak_pwr * self.cyc.dt_s[i]

        self.cur_max_mc_kw_out[i] = max(
            min(
                self.mc_elec_in_lim_kw[i], 
                self.mc_transi_lim_kw[i], 
                np.float64(0 if self.veh.stop_start else 1) * self.veh.max_motor_kw),
            -self.veh.max_motor_kw
        )

        if self.cur_max_mc_kw_out[i] == 0:
            self.cur_max_mc_elec_kw_in[i] = 0
        else:
            if self.cur_max_mc_kw_out[i] == self.veh.max_motor_kw:
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i] / \
                    self.veh.mc_full_eff_array[-1]
            else:
                self.cur_max_mc_elec_kw_in[i] = (self.cur_max_mc_kw_out[i] / self.veh.mc_full_eff_array[
                        max(1, np.argmax(
                            self.veh.mc_kw_out_array > min(self.veh.max_motor_kw - 0.01, self.cur_max_mc_kw_out[i])
                            ) - 1
                        )
                    ]
                )

        if self.veh.max_motor_kw == 0:
            self.ess_lim_mc_regen_perc_kw[i] = 0.0

        else:
            self.ess_lim_mc_regen_perc_kw[i] = min(
                (self.cur_max_ess_chg_kw[i] + self.aux_in_kw[i]) / self.veh.max_motor_kw, 1)
        if self.cur_max_ess_chg_kw[i] == 0:
            self.ess_lim_mc_regen_kw[i] = 0.0

        else:
            if self.veh.max_motor_kw == self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i]:
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.max_motor_kw, self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[-1])
            else:
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.max_motor_kw, 
                    self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[
                        max(1, 
                            np.argmax(
                                self.veh.mc_kw_out_array > min(
                                    self.veh.max_motor_kw - 0.01, 
                                    self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i]
                                )
                            ) - 1
                        )
                    ]
                )

        self.cur_max_mech_mc_kw_in[i] = min(
            self.ess_lim_mc_regen_kw[i], self.veh.max_motor_kw)
        self.cur_max_trac_kw[i] = (
            self.veh.wheel_coef_of_fric * self.veh.drive_axle_weight_frac * self.veh.veh_kg * self.props.a_grav_mps2
            / (1 + self.veh.veh_cg_m * self.veh.wheel_coef_of_fric / self.veh.wheel_base_m) / 1_000 * self.max_trac_mps[i]
        )

        if self.veh.fc_eff_type == H2FC:

            if self.veh.no_elec_sys or self.veh.no_elec_aux or self.high_acc_fc_on_tag[i]:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] - self.aux_in_kw[i]) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )

            else:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] - min(self.cur_max_elec_kw[i], 0)) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )

        else:

            if self.veh.no_elec_sys or self.veh.no_elec_aux or self.high_acc_fc_on_tag[i]:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i] - self.aux_in_kw[i]) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )

            else:
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i] - min(self.cur_max_elec_kw[i], 0)) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                )
        
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

        self.cyc_drag_kw[i] = 0.5 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * self.veh.frontal_area_m2 * (
            (self.mps_ach[i-1] + mpsAch) / 2.0) ** 3 / 1_000
        self.cyc_accel_kw[i] = self.veh.veh_kg / (2.0 * self.cyc.dt_s[i]) * (mpsAch ** 2 - self.mps_ach[i-1] ** 2) / 1_000
        self.cyc_ascent_kw[i] = self.props.a_grav_mps2 * np.sin(np.arctan(
            self.cyc.grade[i])) * self.veh.veh_kg * ((self.mps_ach[i-1] + mpsAch) / 2.0) / 1_000
        self.cyc_trac_kw_req[i] = self.cyc_drag_kw[i] + \
            self.cyc_accel_kw[i] + self.cyc_ascent_kw[i]
        self.spare_trac_kw[i] = self.cur_max_trac_kw[i] - self.cyc_trac_kw_req[i]
        self.cyc_rr_kw[i] = self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef * np.cos(
            np.arctan(self.cyc.grade[i])) * (self.mps_ach[i-1] + mpsAch) / 2.0 / 1_000
        self.cyc_whl_rad_per_sec[i] = mpsAch / self.veh.wheel_radius_m
        self.cyc_tire_inertia_kw[i] = (
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * self.cyc_whl_rad_per_sec[i] ** 2.0 / self.cyc.dt_s[i] -
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * (self.mps_ach[i-1] / self.veh.wheel_radius_m) ** 2.0 / self.cyc.dt_s[i]
        ) / 1_000

        self.cyc_whl_kw_req[i] = self.cyc_trac_kw_req[i] + self.cyc_rr_kw[i] + self.cyc_tire_inertia_kw[i]
        self.regen_contrl_lim_kw_perc[i] = self.veh.max_regen / (1 + self.veh.regen_a * np.exp(-self.veh.regen_b * (
            (self.cyc.mph[i] + self.mps_ach[i-1] * params.MPH_PER_MPS) / 2.0 + 1.0)))
        self.cyc_regen_brake_kw[i] = max(min(
                self.cur_max_mech_mc_kw_in[i] * self.veh.trans_eff, 
                self.regen_contrl_lim_kw_perc[i] * -self.cyc_whl_kw_req[i]), 
            0
        )
        self.cyc_fric_brake_kw[i] = -min(self.cyc_regen_brake_kw[i] + self.cyc_whl_kw_req[i], 0)
        self.cyc_trans_kw_out_req[i] = self.cyc_whl_kw_req[i] + self.cyc_fric_brake_kw[i]

        if self.cyc_trans_kw_out_req[i] <= self.cur_max_trans_kw_out[i]:
            self.cyc_met[i] = True
            self.trans_kw_out_ach[i] = self.cyc_trans_kw_out_req[i]

        else:
            self.cyc_met[i] = False
            self.trans_kw_out_ach[i] = self.cur_max_trans_kw_out[i]

        if self.trans_kw_out_ach[i] > 0:
            self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] / self.veh.trans_eff
        else:
            self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] * self.veh.trans_eff

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
                return xs[_ys.index(min(_ys))]

            drag3 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * \
                self.veh.drag_coef * self.veh.frontal_area_m2
            accel2 = 0.5 * self.veh.veh_kg / self.cyc.dt_s[i]
            drag2 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * \
                self.veh.drag_coef * self.veh.frontal_area_m2 * self.mps_ach[i-1]
            wheel2 = 0.5 * self.veh.wheel_inertia_kg_m2 * \
                self.veh.num_wheels / (self.cyc.dt_s[i] * self.veh.wheel_radius_m ** 2)
            drag1 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * \
                self.veh.frontal_area_m2 * self.mps_ach[i-1] ** 2
            roll1 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef \
                * np.cos(np.arctan(self.cyc.grade[i])) 
            ascent1 = 0.5 * self.props.a_grav_mps2 * \
                np.sin(np.arctan(self.cyc.grade[i])) * self.veh.veh_kg 
            accel0 = -0.5 * self.veh.veh_kg * self.mps_ach[i-1] ** 2 / self.cyc.dt_s[i]
            drag0 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * \
                self.veh.frontal_area_m2 * self.mps_ach[i-1] ** 3
            roll0 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * \
                self.veh.wheel_rr_coef * np.cos(np.arctan(self.cyc.grade[i])) \
                * self.mps_ach[i-1]
            ascent0 = 0.5 * self.props.a_grav_mps2 * np.sin(np.arctan(self.cyc.grade[i])) \
                * self.veh.veh_kg * self.mps_ach[i-1] 
            wheel0 = -0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * \
                self.mps_ach[i-1] ** 2 / (self.cyc.dt_s[i] * self.veh.wheel_radius_m ** 2)

            total3 = drag3 / 1_000
            total2 = (accel2 + drag2 + wheel2) / 1_000
            total1 = (drag1 + roll1 + ascent1) / 1_000
            total0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) / 1_000 - self.cur_max_trans_kw_out[i]

            total = np.array([total3, total2, total1, total0])
            self.mps_ach[i] = newton_mps_estimate(total)
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
                self.veh.max_soc - (self.veh.max_regen_kwh / self.veh.max_ess_kwh), (self.veh.max_soc + self.veh.min_soc) / 2)

        else:
            self.regen_buff_soc[i] = max(
                (self.veh.max_ess_kwh * self.veh.max_soc - 
                    0.5 * self.veh.veh_kg * (self.cyc.mps[i] ** 2) * (1.0 / 1_000) * (1.0 / 3_600) * 
                    self.veh.mc_peak_eff * self.veh.max_regen) / self.veh.max_ess_kwh, 
                self.veh.min_soc
            )

            self.ess_regen_buff_dischg_kw[i] = min(self.cur_max_ess_kw_out[i], max(
                0, (self.soc[i-1] - self.regen_buff_soc[i]) * self.veh.max_ess_kwh * 3_600 / self.cyc.dt_s[i]))

            self.max_ess_regen_buff_chg_kw[i] = min(max(
                    0, 
                    (self.regen_buff_soc[i] - self.soc[i-1]) * self.veh.max_ess_kwh * 3.6e3 / self.cyc.dt_s[i]), 
                self.cur_max_ess_chg_kw[i]
            )

        if self.veh.no_elec_sys:
            self.accel_buff_soc[i] = 0

        else:
            self.accel_buff_soc[i] = min(
                max(
                    ((self.veh.max_accel_buffer_mph / params.MPH_PER_MPS) ** 2 - self.cyc.mps[i] ** 2) / 
                    (self.veh.max_accel_buffer_mph / params.MPH_PER_MPS) ** 2 * min(
                        self.veh.max_accel_buffer_perc_of_useable_soc * (self.veh.max_soc - self.veh.min_soc), 
                        self.veh.max_regen_kwh / self.veh.max_ess_kwh
                    ) * self.veh.max_ess_kwh / self.veh.max_ess_kwh + self.veh.min_soc, 
                    self.veh.min_soc
                ), 
                self.veh.max_soc
                )

            self.ess_accel_buff_chg_kw[i] = max(
                0, (self.accel_buff_soc[i] - self.soc[i-1]) * self.veh.max_ess_kwh * 3.6e3 / self.cyc.dt_s[i])
            self.max_ess_accell_buff_dischg_kw[i] = min(
                max(
                    0, 
                    (self.soc[i-1] - self.accel_buff_soc[i]) * self.veh.max_ess_kwh * 3_600 / self.cyc.dt_s[i]), 
                self.cur_max_ess_kw_out[i]
            )

        if self.regen_buff_soc[i] < self.accel_buff_soc[i]:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(
                    (self.soc[i-1] - (self.regen_buff_soc[i] + self.accel_buff_soc[i]) / 2) * self.veh.max_ess_kwh * 3.6e3 / self.cyc.dt_s[i], 
                    self.cur_max_ess_kw_out[i]
                ), 
                -self.cur_max_ess_chg_kw[i]
            )

        elif self.soc[i-1] > self.regen_buff_soc[i]:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(
                    self.ess_regen_buff_dischg_kw[i], 
                    self.cur_max_ess_kw_out[i]), 
                -self.cur_max_ess_chg_kw[i]
            )

        elif self.soc[i-1] < self.accel_buff_soc[i]:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(-1.0 * self.ess_accel_buff_chg_kw[i], self.cur_max_ess_kw_out[i]), -self.cur_max_ess_chg_kw[i])

        else:
            self.ess_accel_regen_dischg_kw[i] = max(
                min(0, self.cur_max_ess_kw_out[i]), -self.cur_max_ess_chg_kw[i])

        self.fc_kw_gap_fr_eff[i] = abs(self.trans_kw_out_ach[i] - self.veh.max_fc_eff_kw)

        if self.veh.no_elec_sys:
            self.mc_elec_in_kw_for_max_fc_eff[i] = 0

        elif self.trans_kw_out_ach[i] < self.veh.max_fc_eff_kw:
            if self.fc_kw_gap_fr_eff[i] == self.veh.max_motor_kw:
                self.mc_elec_in_kw_for_max_fc_eff[i] = -self.fc_kw_gap_fr_eff[i] / self.veh.mc_full_eff_array[-1]
            else:
                self.mc_elec_in_kw_for_max_fc_eff[i] = (-self.fc_kw_gap_fr_eff[i] / 
                    self.veh.mc_full_eff_array[max(1, 
                        np.argmax(self.veh.mc_kw_out_array > min(self.veh.max_motor_kw - 0.01, self.fc_kw_gap_fr_eff[i])) - 1)]
                )

        else:
            if self.fc_kw_gap_fr_eff[i] == self.veh.max_motor_kw:
                self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[len(
                    self.veh.mc_kw_in_array) - 1]
            else:
                self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[np.argmax(
                    self.veh.mc_kw_out_array > min(self.veh.max_motor_kw - 0.01, self.fc_kw_gap_fr_eff[i])) - 1]

        if self.veh.no_elec_sys:
            self.elec_kw_req_4ae[i] = 0

        elif self.trans_kw_in_ach[i] > 0:
            if self.trans_kw_in_ach[i] == self.veh.max_motor_kw:
                self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i] / self.veh.mc_full_eff_array[-1] + self.aux_in_kw[i]
            else:
                self.elec_kw_req_4ae[i] = (self.trans_kw_in_ach[i] / 
                    self.veh.mc_full_eff_array[max(1, np.argmax(
                        self.veh.mc_kw_out_array > min(self.veh.max_motor_kw - 0.01, self.trans_kw_in_ach[i])) - 1)] + self.aux_in_kw[i]
                )

        else:
            self.elec_kw_req_4ae[i] = 0

        self.prev_fc_time_on[i] = self.fc_time_on[i-1]

        # some conditions in the following if statement have a buffer of 1e-6 to prevent false positives/negatives because these have been encountered in practice.   
        if self.veh.max_fuel_conv_kw == 0:
            self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i-1] and  \
                (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i] and \
                (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i] or self.veh.max_fuel_conv_kw == 0)

        else:
            self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i-1] and \
                (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i] and \
                (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i] or self.veh.max_fuel_conv_kw == 0) \
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
                self.desired_ess_kw_out_for_ae[i] = -self.ess_accel_buff_chg_kw[i]

            else:
                self.desired_ess_kw_out_for_ae[i] = self.trans_kw_in_ach[i] + \
                    self.aux_in_kw[i] - self.cur_max_roadway_chg_kw[i]

        else:   
            self.desired_ess_kw_out_for_ae[i] = 0

        if self.can_pwr_all_elec[i]:
            self.ess_ae_kw_out[i] = max(
                -self.cur_max_ess_chg_kw[i], 
                -self.max_ess_regen_buff_chg_kw[i], 
                min(0, self.cur_max_roadway_chg_kw[i] - self.trans_kw_in_ach[i] + self.aux_in_kw[i]), 
                min(self.cur_max_ess_kw_out[i], self.desired_ess_kw_out_for_ae[i])
            )

        else:
            self.ess_ae_kw_out[i] = 0

        self.er_ae_kw_out[i] = min(
            max(0, self.trans_kw_in_ach[i] + self.aux_in_kw[i] - self.ess_ae_kw_out[i]), 
            self.cur_max_roadway_chg_kw[i])
    
    def set_fc_forced_state(self, i):
        """Calculate control variables related to engine on/off state
        Arguments       
        ------------
        i: index of time step"""
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

        elif self.veh.idle_fc_kw > self.trans_kw_in_ach[i] and self.cyc_accel_kw[i] >= 0:
            self.fc_forced_state[i] = 4
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - self.veh.idle_fc_kw

        elif self.veh.max_fc_eff_kw > self.trans_kw_in_ach[i]:
            self.fc_forced_state[i] = 5
            self.mc_mech_kw_4forced_fc[i] = 0

        else:
            self.fc_forced_state[i] = 6
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - \
                self.veh.max_fc_eff_kw

    def set_hybrid_cont_decisions(self, i):
        """Hybrid control decisions.
        Arguments
        ------------
        i: index of time step"""

        if (-self.mc_elec_in_kw_for_max_fc_eff[i] - self.cur_max_roadway_chg_kw[i]) > 0:
            self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i] -
                                            self.cur_max_roadway_chg_kw[i]) * self.veh.ess_dischg_to_fc_max_eff_perc

        else:
            self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i] - \
                                            self.cur_max_roadway_chg_kw[i]) * self.veh.ess_chg_to_fc_max_eff_perc

        if self.accel_buff_soc[i] > self.regen_buff_soc[i]:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i], 
                self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i], self.ess_accel_regen_dischg_kw[i]))

        elif self.ess_regen_buff_dischg_kw[i] > 0:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i], 
                self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i], 
                    min(self.ess_accel_regen_dischg_kw[i], 
                        self.mc_elec_in_lim_kw[i] + self.aux_in_kw[i], 
                        max(self.ess_regen_buff_dischg_kw[i], self.ess_desired_kw_4fc_eff[i])
                    )
                )
            )

        elif self.ess_accel_buff_chg_kw[i] > 0:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i], 
                self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i], 
                    max(-1 * self.max_ess_regen_buff_chg_kw[i], 
                        min(-self.ess_accel_buff_chg_kw[i], self.ess_desired_kw_4fc_eff[i])
                    )
                )
            )

        elif self.ess_desired_kw_4fc_eff[i] > 0:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i], 
                self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], 
                self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i], 
                    min(self.ess_desired_kw_4fc_eff[i], self.max_ess_accell_buff_dischg_kw[i])
                )
            )

        else:
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i], 
                self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], 
                self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                max(-self.cur_max_ess_chg_kw[i], 
                    max(self.ess_desired_kw_4fc_eff[i], -self.max_ess_regen_buff_chg_kw[i])
                )
            )

        self.er_kw_if_fc_req[i] = max(0, 
            min(
                self.cur_max_roadway_chg_kw[i], self.cur_max_mech_mc_kw_in[i],
                self.ess_kw_if_fc_req[i] - self.mc_elec_in_lim_kw[i] + self.aux_in_kw[i]
            )
        )

        self.mc_elec_kw_in_if_fc_req[i] = self.ess_kw_if_fc_req[i] + self.er_kw_if_fc_req[i] - self.aux_in_kw[i]

        if self.veh.no_elec_sys:
            self.mc_kw_if_fc_req[i] = 0

        elif self.mc_elec_kw_in_if_fc_req[i] == 0:
            self.mc_kw_if_fc_req[i] = 0

        elif self.mc_elec_kw_in_if_fc_req[i] > 0:

            if self.mc_elec_kw_in_if_fc_req[i] == max(self.veh.mc_kw_in_array):
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] * self.veh.mc_full_eff_array[-1]
            else:
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] * self.veh.mc_full_eff_array[
                    max(1, np.argmax(
                            self.veh.mc_kw_in_array > min(max(self.veh.mc_kw_in_array) - 0.01, self.mc_elec_kw_in_if_fc_req[i])
                        ) - 1
                    )
                ]

        else:
            if self.mc_elec_kw_in_if_fc_req[i] * -1 == max(self.veh.mc_kw_in_array):
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] / self.veh.mc_full_eff_array[-1]
            else:
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] / (self.veh.mc_full_eff_array[
                        max(1, np.argmax(
                            self.veh.mc_kw_in_array > min(max(self.veh.mc_kw_in_array) - 0.01, self.mc_elec_kw_in_if_fc_req[i] * -1)) - 1
                        )
                    ]
                )

        if self.veh.max_motor_kw == 0:
            self.mc_mech_kw_out_ach[i] = 0

        elif self.fc_forced_on[i] and self.can_pwr_all_elec[i] and (self.veh.veh_pt_type == HEV or 
            self.veh.veh_pt_type == PHEV) and (self.veh.fc_eff_type != H2FC):
            self.mc_mech_kw_out_ach[i] = self.mc_mech_kw_4forced_fc[i]

        elif self.trans_kw_in_ach[i] <= 0:

            if self.veh.fc_eff_type !=H2FC and self.veh.max_fuel_conv_kw > 0:
                if self.can_pwr_all_elec[i] == 1:
                    self.mc_mech_kw_out_ach[i] = - \
                        min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i])
                else:
                    self.mc_mech_kw_out_ach[i] = min(
                        -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]),
                        max(-self.cur_max_fc_kw_out[i], self.mc_kw_if_fc_req[i])
                    )
            else:
                self.mc_mech_kw_out_ach[i] = min(
                    -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]), 
                    -self.trans_kw_in_ach[i]
                )

        elif self.can_pwr_all_elec[i] == 1:
            self.mc_mech_kw_out_ach[i] = self.trans_kw_in_ach[i]

        else:
            self.mc_mech_kw_out_ach[i] = max(self.min_mc_kw_2help_fc[i], self.mc_kw_if_fc_req[i])

        if self.mc_mech_kw_out_ach[i] == 0:
            self.mc_elec_kw_in_ach[i] = 0.0

        elif self.mc_mech_kw_out_ach[i] < 0:

            if self.mc_mech_kw_out_ach[i] * -1 == max(self.veh.mc_kw_in_array):
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] * self.veh.mc_full_eff_array[-1]
            else:
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] * self.veh.mc_full_eff_array[
                    max(1, np.argmax(self.veh.mc_kw_in_array > min(
                        max(self.veh.mc_kw_in_array) - 0.01, 
                        self.mc_mech_kw_out_ach[i] * -1)) - 1
                    )
                ]

        else:
            if self.veh.max_motor_kw == self.mc_mech_kw_out_ach[i]:
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] / self.veh.mc_full_eff_array[-1]
            else:
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] / self.veh.mc_full_eff_array[
                    max(1, np.argmax(self.veh.mc_kw_out_array > min(
                        self.veh.max_motor_kw - 0.01, 
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

        if self.veh.max_ess_kw == 0 or self.veh.max_ess_kwh == 0:
            self.ess_kw_out_ach[i] = 0

        elif self.veh.fc_eff_type == H2FC:

            if self.trans_kw_out_ach[i] >=0:
                self.ess_kw_out_ach[i] = min(max(
                        self.min_ess_kw_2help_fc[i], 
                        self.ess_desired_kw_4fc_eff[i], 
                        self.ess_accel_regen_dischg_kw[i]),
                    self.cur_max_ess_kw_out[i], 
                    self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i]
                )

            else:
                self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] + \
                    self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i]

        elif self.high_acc_fc_on_tag[i] or self.veh.no_elec_aux:
            self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] - self.roadway_chg_kw_out_ach[i]

        else:
            self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i]

        if self.veh.no_elec_sys:
            self.ess_cur_kwh[i] = 0

        elif self.ess_kw_out_ach[i] < 0:
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s[i] /\
                3.6e3 * np.sqrt(self.veh.ess_round_trip_eff)

        else:
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s[i] / \
                3.6e3 * (1 / np.sqrt(self.veh.ess_round_trip_eff))

        if self.veh.max_ess_kwh == 0:
            self.soc[i] = 0.0

        else:
            self.soc[i] = self.ess_cur_kwh[i] / self.veh.max_ess_kwh

        if self.can_pwr_all_elec[i] and not(self.fc_forced_on[i]) and self.fc_kw_out_ach[i] == 0.0:
            self.fc_time_on[i] = 0
        else:
            self.fc_time_on[i] = self.fc_time_on[i-1] + self.cyc.dt_s[i]
    
    def set_fc_power(self, i):
        """Sets fcKwOutAch and fcKwInAch.
        Arguments
        ------------
        i: index of time step"""

        if self.veh.max_fuel_conv_kw == 0:
            self.fc_kw_out_ach[i] = 0

        elif self.veh.fc_eff_type == H2FC:
            self.fc_kw_out_ach[i] = min(
                self.cur_max_fc_kw_out[i], 
                max(0, 
                    self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.ess_kw_out_ach[i] - self.roadway_chg_kw_out_ach[i]
                )
            )

        elif self.veh.no_elec_sys or self.veh.no_elec_aux or self.high_acc_fc_on_tag[i]:
            self.fc_kw_out_ach[i] = min(
                self.cur_max_fc_kw_out[i], 
                max(
                    0, 
                    self.trans_kw_in_ach[i] - self.mc_mech_kw_out_ach[i] + self.aux_in_kw[i]
                )
            )

        else:
            self.fc_kw_out_ach[i] = min(self.cur_max_fc_kw_out[i], max(
                0, self.trans_kw_in_ach[i] - self.mc_mech_kw_out_ach[i]))

        if self.veh.max_fuel_conv_kw == 0:
            self.fc_kw_out_ach_pct[i] = 0
        else:
            self.fc_kw_out_ach_pct[i] = self.fc_kw_out_ach[i] / self.veh.max_fuel_conv_kw

        if self.fc_kw_out_ach[i] == 0:
            self.fc_kw_in_ach[i] = 0
            self.fc_kw_out_ach_pct[i] = 0

        else:
            self.fc_kw_in_ach[i] = (
                self.fc_kw_out_ach[i] / (self.veh.fc_eff_array[np.argmax(
                    self.veh.fc_kw_out_array > min(self.fc_kw_out_ach[i], self.veh.fc_max_out_kw)) - 1]) 
                if self.veh.fc_eff_array[np.argmax(
                    self.veh.fc_kw_out_array > min(self.fc_kw_out_ach[i], self.veh.fc_max_out_kw)) - 1] != 0
                else 0)

        self.fs_kw_out_ach[i] = self.fc_kw_in_ach[i]

        self.fs_kwh_out_ach[i] = self.fs_kw_out_ach[i] * \
            self.cyc.dt_s[i] * (1 / 3.6e3)

    def set_time_dilation(self, i):
        trace_met = (
            ((abs(self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum()) / self.cyc0.dist_m[:i+1].sum()
            ) < self.sim_params.time_dilation_tol) or 
            (self.cyc.mps[i] == 0) # if prescribed speed is zero, trace is met to avoid div-by-zero errors and other possible wackiness
        )

        if not(trace_met):
            self.trace_miss_iters[i] += 1

            d_short = [self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum()] # positive if behind trace
            t_dilation = [
                0.0, # no time dilation initially
                min(max(
                        d_short[-1] / self.cyc0.dt_s[i] / self.mps_ach[i], # initial guess, speed that needed to be achived per speed that was achieved
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
            d_short.append(self.cyc0.dist_m[:i+1].sum() - self.dist_m[:i+1].sum())
            t_dilation.append(
                min(
                    max(
                        t_dilation[-1] - (t_dilation[-1] - t_dilation[-2]) / (d_short[-1] - d_short[-2]) * d_short[-1],
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
    
    def set_coast_speed(self, i):
        """
        Placeholder for method to impose coasting.
        Might be good to include logic for deciding when to coast.
        """
        pass

    def set_post_scalars(self):
        """Sets scalar variables that can be calculated after a cycle is run. 
        This includes mpgge, various energy metrics, and others"""
        
        self.fs_cumu_mj_out_ach = (self.fs_kw_out_ach * self.cyc.dt_s).cumsum() * 1e-3

        if self.fs_kwh_out_ach.sum() == 0:
            self.mpgge = 0.0

        else:
            self.mpgge = self.dist_mi.sum() / (self.fs_kwh_out_ach.sum() / self.props.kwh_per_gge)

        self.roadwayChgKj = (self.roadway_chg_kw_out_ach * self.cyc.dt_s).sum()
        self.essDischgKj = - \
            (self.soc[-1] - self.soc[0]) * self.veh.max_ess_kwh * 3.6e3
        self.battery_kWh_per_mi  = (
            self.essDischgKj / 3.6e3) / self.dist_mi.sum()
        self.electric_kWh_per_mi  = (
            (self.roadwayChgKj + self.essDischgKj) / 3.6e3) / self.dist_mi.sum()
        self.fuelKj = (self.fs_kw_out_ach * self.cyc.dt_s).sum()

        if (self.fuelKj + self.roadwayChgKj) == 0:
            self.ess2fuelKwh  = 1.0

        else:
            self.ess2fuelKwh  = self.essDischgKj / (self.fuelKj + self.roadwayChgKj)

        if self.mpgge == 0:
            # hardcoded conversion
            self.Gallons_gas_equivalent_per_mile = self.electric_kWh_per_mi / self.props.kwh_per_gge
            grid_Gallons_gas_equivalent_per_mile = self.electric_kWh_per_mi / self.props.kwh_per_gge / \
                self.veh.chg_eff

        else:
            self.Gallons_gas_equivalent_per_mile = 1 / \
                self.mpgge + self.electric_kWh_per_mi  / self.props.kwh_per_gge
            grid_Gallons_gas_equivalent_per_mile = 1 / self.mpgge + \
                self.electric_kWh_per_mi / self.props.kwh_per_gge / self.veh.chg_eff

        self.grid_mpgge_elec = 1 / grid_Gallons_gas_equivalent_per_mile
        self.mpgge_elec = 1 / self.Gallons_gas_equivalent_per_mile

        # energy audit calcs
        self.drag_kw = self.cyc_drag_kw 
        self.dragKj = (self.drag_kw * self.cyc.dt_s).sum()
        self.ascent_kw = self.cyc_ascent_kw
        self.ascentKj = (self.ascent_kw * self.cyc.dt_s).sum()
        self.rr_kw = self.cyc_rr_kw
        self.rrKj = (self.rr_kw * self.cyc.dt_s).sum()

        self.ess_loss_kw[1:] = np.array(
            [0 if (self.veh.max_ess_kw == 0 or self.veh.max_ess_kwh == 0)
            else -self.ess_kw_out_ach[i] - (-self.ess_kw_out_ach[i] * np.sqrt(self.veh.ess_round_trip_eff))
                if self.ess_kw_out_ach[i] < 0
            else self.ess_kw_out_ach[i] * (1.0 / np.sqrt(self.veh.ess_round_trip_eff)) - self.ess_kw_out_ach[i]
            for i in range(1, len(self.cyc.time_s))]
        )
        
        self.brakeKj = (self.cyc_fric_brake_kw * self.cyc.dt_s).sum()
        self.transKj = ((self.trans_kw_in_ach - self.trans_kw_out_ach) * self.cyc.dt_s).sum()
        self.mcKj = ((self.mc_elec_kw_in_ach - self.mc_mech_kw_out_ach) * self.cyc.dt_s).sum()
        self.essEffKj = (self.ess_loss_kw * self.cyc.dt_s).sum()
        self.auxKj = (self.aux_in_kw * self.cyc.dt_s).sum()
        self.fcKj = ((self.fc_kw_in_ach - self.fc_kw_out_ach) * self.cyc.dt_s).sum()
        
        self.netKj = self.dragKj + self.ascentKj + self.rrKj + self.brakeKj + self.transKj \
            + self.mcKj + self.essEffKj + self.auxKj + self.fcKj

        self.keKj = 0.5 * self.veh.veh_kg * \
            (self.mps_ach[0] ** 2 - self.mps_ach[-1] ** 2) / 1_000
        
        self.energyAuditError = ((self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj) - self.netKj
            ) / (self.roadwayChgKj + self.essDischgKj + self.fuelKj + self.keKj)

        if (np.abs(self.energyAuditError) > self.sim_params.energy_audit_error_tol) and \
            self.sim_params.verbose:
            print('Warning: There is a problem with conservation of energy.')
            print('Energy Audit Error:', np.round(self.energyAuditError, 5))

        self.accel_kw[1:] = (self.veh.veh_kg / (2.0 * (self.cyc.dt_s[1:]))) * (
            self.mps_ach[1:] ** 2 - self.mps_ach[:-1] ** 2) / 1_000

        self.trace_miss = False
        self.trace_miss_dist_frac = abs(self.dist_m.sum() - self.cyc0.dist_m.sum()) / self.cyc0.dist_m.sum()
        self.trace_miss_time_frac = abs(self.cyc.time_s[-1] - self.cyc0.time_s[-1]) / self.cyc0.time_s[-1]

        if not(self.sim_params.missed_trace_correction):
            if self.trace_miss_dist_frac > self.sim_params.trace_miss_dist_tol:
                self.trace_miss = True
                if self.sim_params.verbose:
                    print('Warning: Trace miss distance fraction:', np.round(self.trace_miss_dist_frac, 5))
                    print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_dist_tol, 5))
        else:
            if self.trace_miss_time_frac > self.sim_params.trace_miss_time_tol:
                self.trace_miss = True
                if self.sim_params.verbose:
                    print('Warning: Trace miss time fraction:', np.round(self.trace_miss_time_frac, 5))
                    print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_time_tol, 5))

        self.trace_miss_speed_mps = max([
            abs(self.mps_ach[i] - self.cyc.mps[i]) for i in range(len(self.cyc.time_s))
        ])
        if self.trace_miss_speed_mps > self.sim_params.trace_miss_speed_mps_tol:
            self.trace_miss = True
            if self.sim_params.verbose:
                print('Warning: Trace miss speed [m/s]:', np.round(self.trace_miss_speed_mps, 5))
                print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_speed_mps_tol, 5))
    
    def to_rust(self):
        "Create a rust version of SimDrive"
        return copy_sim_drive(self, 'rust', True)

class LegacySimDrive(object):
    pass


ref_sim_drive = SimDrive(cycle.ref_cyc, vehicle.ref_veh)
sd_params = inspect_utils.get_attrs(ref_sim_drive)

def copy_sim_drive(sd:SimDrive, return_type:str=None, deep:bool=True) -> SimDrive:
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
        #if type(sd) == fsr.RustSimDrive:
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
    sd_copy = SimDrive(cyc, veh)

    if return_type == 'rust':
        return fsr.RustSimDrive(cyc, veh)

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
            sp_return_type = 'sim_params' if (return_type == 'sim_drive' or return_type == 'legacy') else return_type
            sd_copy.sim_params = copy_sim_params(sd.sim_params, sp_return_type)
        elif key == 'props':
            pp_return_type = 'physical_properties' if (return_type == 'sim_drive' or return_type == 'legacy') else return_type
            sd_copy.props = params.copy_physical_properties(sd.props, pp_return_type)
        else:
            # should be ok to deep copy
            val = sd.__getattribute__(key)
            sd_copy.__setattr__(key, copy.deepcopy(val) if deep else val)
        
    return sd_copy                

def sim_drive_equal(a:SimDrive, b:SimDrive, verbose=False) -> bool:
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
        elif type(a_val) == np.ndarray or type(b_val) == np.ndarray:
            if not (a_val == b_val).all():
                if verbose:
                    print(f"unequal at key {k}: {a_val} != {b_val}")
                return False
        elif a_val != b_val:
            if verbose:
                print(f"unequal at key {k}: {a_val} != {b_val}")
            return False
    return True

class SimAccelTest(SimDrive):
    """Class for running FASTSim vehicle acceleration simulation."""

    def sim_drive(self):
        """Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType."""

        if self.veh.veh_pt_type == CONV:  # Conventional

            # If no EV / Hybrid components, no SOC considerations.

            init_soc = (self.veh.max_soc + self.veh.min_soc) / 2.0
            self.sim_drive_walk(init_soc)

        elif self.veh.veh_pt_type == HEV:  # HEV

            init_soc = (self.veh.max_soc + self.veh.min_soc) / 2.0
            self.sim_drive_walk(init_soc)

        else:

            # If EV, initializing initial SOC to maximum SOC.
            init_soc = self.veh.max_soc
            self.sim_drive_walk(init_soc)

        self.set_post_scalars()


class SimDrivePost(object):
    """Class for post-processing of SimDrive instance.  Requires already-run 
    SimDriveJit or SimDriveClassic instance."""
    
    def __init__(self, sim_drive:SimDrive):
        """Arguments:
        ---------------
        sim_drive: solved sim_drive object"""
        
        for item in inspect_utils.get_attrs(sim_drive):
            self.__setattr__(item, sim_drive.__getattribute__(item))

    def get_output(self):
        """Calculate finalized results
        Arguments
        ------------
        init_soc: initial SOC for electrified vehicles
        
        Returns
        ------------
        output: dict of summary output variables"""

        output = {}

        output['mpgge'] = self.mpgge
        output['battery_kWh_per_mi'] = self.battery_kWh_per_mi
        output['electric_kWh_per_mi'] = self.electric_kWh_per_mi
        output['maxTraceMissMph'] = params.MPH_PER_MPS * max(abs(self.cyc.mps - self.mpsAch))
        self.maxTraceMissMph = output['maxTraceMissMph']

        output['ess2fuelKwh'] = self.ess2fuelKwh

        output['initial_soc'] = self.soc[0]
        output['final_soc'] = self.soc[-1]

        output['mpgge_elec'] = self.mpgge_elec
        output['soc'] = self.soc
        output['distance_mi'] = sum(self.distMiles)
        duration_sec = self.cyc.time_s[-1] - self.cyc.time_s[0]
        output['avg_speed_mph'] = sum(
            self.distMiles) / (duration_sec / 3.6e3)
        self.avg_speed_mph = output['avg_speed_mph']
        self.accel = np.diff(self.mphAch) / np.diff(self.cyc.time_s)
        output['avg_accel_mphps'] = np.mean(self.accel[self.accel > 0])
        self.avg_accel_mphps = output['avg_accel_mphps']

        if max(self.mphAch) > 60:
            output['ZeroToSixtyTime_secs'] = np.interp(60, self.mphAch, self.cyc.time_s)

        else:
            output['ZeroToSixtyTime_secs'] = 0.0

        output['fcKwOutAch'] = np.asarray(self.fcKwOutAch)
        output['fsKwhOutAch'] = np.asarray(self.fsKwhOutAch)
        output['fcKwInAch'] = np.asarray(self.fcKwInAch)
        output['time'] = np.asarray(self.cyc.time_s)

        return output

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
        tempvars = {} # dict for contaning intermediate variables
        output = {}
        for var in pw_var_list:
            tempvars[var + '_pos'] = [x if x >= 0 
                                        else 0 
                                        for x in self.__getattribute__(var)]
            tempvars[var + '_neg'] = [x if x < 0 
                                        else 0 
                                        for x in self.__getattribute__(var)]    
                        
            # Assign values to output dict for positive and negative energy variable names
            search = prog.search(var)
            output[search[1] + '_kj' + search[2] + '_pos'] = np.trapz(tempvars[var + '_pos'], self.cyc.time_s)
            output[search[1] + '_kj' + search[2] + '_neg'] = np.trapz(tempvars[var + '_neg'], self.cyc.time_s)
        
        output['dist_miles_final'] = sum(self.dist_mi)
        if sum(self.fs_kwh_out_ach) > 0:
            output['mpgge'] = sum(self.dist_mi) / sum(self.fs_kwh_out_ach) * self.props.kwh_per_gge
        else:
            output['mpgge'] = 0
    
        return output

    def set_battery_wear(self):
        """Battery wear calcs"""

        self.addKwh[1:] = np.array([
            (self.essCurKwh[i] - self.essCurKwh[i-1]) + self.addKwh[i-1]
            if self.essCurKwh[i] > self.essCurKwh[i-1]
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.dodCycs[1:] = np.array([
            self.addKwh[i-1] / self.veh.max_ess_kwh if self.addKwh[i] == 0
            else 0 
            for i in range(1, len(self.essCurKwh))])
        
        self.essPercDeadArray = np.array([
            np.power(self.veh.ess_life_coef_a, 1.0 / self.veh.ess_life_coef_b) / np.power(self.dodCycs[i], 
            1.0 / self.veh.ess_life_coef_b)
            if self.dodCycs[i] != 0
            else 0
            for i in range(0, len(self.essCurKwh))])


def SimDriveJit(cyc_jit, veh_jit):
    """
    deprecated
    """
    raise NotImplementedError("This function has been deprecated.")

def SimAccelTestJit(cyc_jit, veh_jit):
    """
    deprecated
    """
    raise NotImplementedError("This function has been deprecated")