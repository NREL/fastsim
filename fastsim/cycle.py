"""Module containing classes and methods for for loading 
cycle data. For example usage, see ../README.md"""

### Import necessary python modules
import cmath
import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import re
import sys
from pathlib import Path
from copy import deepcopy
import types
from numba import njit

# local modules
from . import parameters as params

THIS_DIR = Path(__file__).parent
CYCLES_DIR = THIS_DIR / 'resources' / 'cycles'
STANDARD_CYCLE_KEYS = [
    'cycSecs', 'cycMps', 'cycGrade', 'cycRoadType', 'name',]


class Cycle(object):
    """Object for containing time, speed, road grade, and road charging vectors 
    for drive cycle."""

    def __init__(self, std_cyc_name=None, cyc_dict=None, cyc_file_path=None):
        """Runs other methods, depending on provided keyword argument.
        Only one keyword argument should be provided.          
        
        Keyword Arguments:
        ------------------
        std_cyc_name : string for name of standard cycle in resources/cycles/.  
            Must match filename minus '.csv' extension.
        cyc_dict : dictionary with 'cycSecs' and 'cycMps' keys (at minimum) 
            and corresponding arrays.  
        cyc_file_path : string for path to custom cycle file, which much have
            same format as cycles in resources/cycles/
        """
        
        if std_cyc_name:
            self.set_standard_cycle(std_cyc_name)
        if cyc_dict:
            self.set_from_dict(cyc_dict)
        if cyc_file_path:
            self.set_from_file(cyc_file_path)
        
    def get_numba_cyc(self):
        """Returns numba jitclass version of Cycle object."""
        from .cyclejit import CycleJit
        numba_cyc = CycleJit(len(self.cycSecs))
        for key in STANDARD_CYCLE_KEYS:
            if pd.api.types.is_list_like(self.__getattribute__(key)):
                # type should already be np.float64 but astype explicitly enforces this
                numba_cyc.__setattr__(key, self.__getattribute__(key).astype(np.float64))
            elif type(self.__getattribute__(key)) == str:
                numba_cyc.__setattr__(key, self.__getattribute__(key))
        return numba_cyc

    def set_standard_cycle(self, std_cyc_name):
        """Load time trace of speed, grade, and road type in a pandas
        dataframe.
        Argument:
        ---------
        std_cyc_name: cycle name string (e.g. 'udds', 'us06', 'hwfet')"""
        csv_path = Path(CYCLES_DIR) / (std_cyc_name + '.csv')
        self.set_from_file(csv_path)

    def set_from_file(self, cyc_file_path):
        """Load time trace of speed, grade, and road type from user-provided csv
        file in a pandas dataframe.
        Argument:
        ---------
        cyc_file_path: path to file containing cycle data"""
        cyc = pd.read_csv(Path(cyc_file_path))
        self.name = Path(cyc_file_path).stem
        for column in cyc.columns:
            if column in STANDARD_CYCLE_KEYS:
                self.__setattr__(column, cyc[column].to_numpy(dtype=np.float64))
        # fill in unspecified optional values with zeros
        if 'cycGrade' not in cyc.columns:
            self.__setattr__('cycGrade', np.zeros(
                len(self.cycMps), dtype=np.float64))
        if 'cycRoadType' not in cyc.columns:
            self.__setattr__('cycRoadType', np.zeros(
                len(self.cycMps), dtype=np.float64))

    def set_from_dict(self, cyc_dict):
        """Set cycle attributes from dict with keys 'cycSecs', 'cycMps',
        'cycGrade' (optional), 'cycRoadType' (optional) and numpy arrays of
        equal length for values.
        Arguments
        ---------
        cyc_dict: dict containing cycle data
        """
        for key in cyc_dict.keys():
            if key in STANDARD_CYCLE_KEYS:
                if pd.api.types.is_list_like(cyc_dict[key]):
                    self.__setattr__(key, np.array(cyc_dict[key], dtype=np.float64))
                if key == 'name':
                    self.__setattr__(key, cyc_dict[key])
        # fill in unspecified optional values with zeros
        if 'cycGrade' not in cyc_dict.keys():
            self.__setattr__('cycGrade', np.zeros(
                len(self.cycMps), dtype=np.float64))
        if 'cycRoadType' not in cyc_dict.keys():
            self.__setattr__('cycRoadType', np.zeros(
                len(self.cycMps), dtype=np.float64))
        if 'name' not in cyc_dict.keys():
            self.__setattr__('name', 'from_dict')
    

    ### Properties

    def get_cycMph(self) -> np.ndarray:
        return self.cycMps * params.mphPerMps

    def set_cycMph(self, new_value):
        self.cycMps = new_value / params.mphPerMps

    cycMph = property(get_cycMph, set_cycMph)

    def get_time_s(self) -> np.ndarray:
        return self.cycSecs

    def set_time_s(self, new_value):
        self.cycSecs = new_value

    time_s = property(get_time_s, set_time_s)

    # time step deltas
    @property
    def secs(self) -> np.ndarray:
        return np.append(0.0, self.cycSecs[1:] - self.cycSecs[:-1]) 

    @property
    def dt_s(self) -> np.ndarray:
        return self.secs
    
    # distance at each time step
    @property
    def cycDistMeters(self) -> np.ndarray:
        return self.cycMps * self.secs
    
    @property
    def cycAvgMps(self):
        return np.append(0.0, 0.5 * (self.cycMps[1:] + self.cycMps[:-1]))
    
    @property
    def cycDistMeters_v2(self):
        return self.cycAvgMps * self.secs

    @property
    def delta_elev_m(self) -> np.ndarray:
        """
        Cumulative elevation change w.r.t. to initial
        """
        return (self.cycDistMeters * self.cycGrade).cumsum()
    
    def get_cyc_dict(self) -> dict:
        """Returns cycle as dict rather than class instance."""
        keys = STANDARD_CYCLE_KEYS
        
        cyc = {}
        for key in keys:
            cyc[key] = self.__getattribute__(key)
        
        return cyc
    
    def copy(self):
        """Return copy of Cycle instance."""
        cyc_dict = {'cycSecs': self.cycSecs,
                    'cycMps': self.cycMps,
                    'cycGrade': self.cycGrade,
                    'cycRoadType': self.cycRoadType,
                    'name': self.name
                    }
        cyc = Cycle(cyc_dict=cyc_dict)
        return cyc
    
    def calc_distance_to_next_stop_from(self, distance_m):
        """
        Calculate the distance to next stop from `distance_m`
        - distance_m: non-negative-number, the current distance from start (m)
        RETURN: -1 or non-negative-integer
        - if there are no more stops ahead, return -1
        - else returns the distance to the next stop from distance_m
        """
        distances_of_stops_m = np.unique(self.cycDistMeters_v2.cumsum()[self.cycMps < 1e-6])
        remaining_stops_m = distances_of_stops_m[distances_of_stops_m > (distance_m + 1e-6)]
        if len(remaining_stops_m) > 0:
            return remaining_stops_m[0] - distance_m
        return -1.0

    def modify_by_const_jerk_trajectory(self, idx, n, jerk_m__s3, accel0_m__s2):
        """
        Modifies the cycle using the given constant-jerk trajectory parameters
        - idx: non-negative integer, the point in the cycle to initiate
        modification (note: THIS point is modified since trajectory should be calculated from idx-1)
        - jerk_m__s3: number, the "Jerk" associated with the trajectory (m/s3)
        - accel0_m__s2: number, the initial acceleration (m/s2)
        NOTE:
        - modifies cyc in place to hit any critical rendezvous_points by a trajectory adjustment
        - CAUTION: NOT ROBUST AGAINST VARIABLE DURATION TIME-STEPS
        RETURN: Number, final modified speed (m/s)
        """
        num_samples = len(self.cycSecs)
        v0 = self.cycMps[idx-1]
        dt = self.secs[idx]
        v = v0
        for ni in range(1, int(n)+1):
            idx_to_set = (int(idx) - 1) + ni
            if idx_to_set >= num_samples:
                break
            v = speed_for_constant_jerk(ni, v0, accel0_m__s2, jerk_m__s3, dt)
            self.cycMps[idx_to_set] = v
        return v

    def modify_with_braking_trajectory(self, brake_accel_m__s2, idx):
        """
        Add a braking trajectory that would cover the same distance as the given constant brake deceleration
        - brake_accel_m__s2: negative number, the braking acceleration (m/s2)
        - idx: non-negative integer, the index where to initiate the stop trajectory, start of the step (i in FASTSim)
        RETURN: non-negative-number, the final speed of the modified trajectory (m/s) 
        - modifies the cycle in place for braking
        """
        assert brake_accel_m__s2 < 0.0
        i = int(idx)
        v0 = self.cycMps[i-1]
        dt = self.secs[i]
        # distance-to-stop (m)
        dts_m = -0.5 * v0 * v0 / brake_accel_m__s2
        # time-to-stop (s)
        tts_s = -v0 / brake_accel_m__s2
        # number of steps to take
        n = int(np.round(tts_s / dt))
        if n < 2:
            # need at least 2 steps
            n = 2
        jerk_m__s3, accel_m__s2 = calc_constant_jerk_trajectory(n, 0.0, v0, dts_m, 0.0, dt)
        return self.modify_by_const_jerk_trajectory(i, n, jerk_m__s3, accel_m__s2)


def copy_cycle(cyc:Cycle, return_dict=False, use_jit=None) -> Cycle:
    """
    Returns copy of Cycle or CycleJit.
    Arguments:
    cyc: instantianed Cycle or CycleJit
    return_dict: (Boolean) if True, returns cycle as dict. 
        Otherwise, returns exact copy.
    use_jit: (Boolean)
        default -- infer from cycle
        True -- use numba
        False -- don't use numba
    """

    cyc_dict = {}

    from . import cyclejit
    for keytup in cyclejit.cyc_spec:
        key = keytup[0]
        cyc_dict[key] = deepcopy(cyc.__getattribute__(key))
        
    if return_dict:
        return cyc_dict
    
    if use_jit is None:
        use_jit = "Jit" in str(type(cyc))
    
    cyc = Cycle(cyc_dict=cyc_dict)
    
    if use_jit:
        cyc = Cycle(cyc_dict=cyc_dict).get_numba_cyc()
        
    return cyc                

def to_microtrips(cycle, stop_speed_m__s=1e-6, keep_name=False):
    """
    Split a cycle into an array of microtrips with one microtrip being a start
    to subsequent stop plus any idle (stopped time).

    Arguments:
    ----------
    cycle: drive cycle converted to dictionary by cycle.get_cyc_dict()
    stop_speed_m__s: speed at which vehicle is considered stopped for trip
        separation
    keep_name: (optional) bool, if True and cycle contains "name", adds
        that name to all microtrips
    """
    microtrips = []
    ts = np.array(cycle['cycSecs'])
    vs = np.array(cycle['cycMps'])
    gs = np.array(cycle['cycGrade'])
    rs = np.array(cycle['cycRoadType'])
    mt = make_cycle([], [])
    moving = False
    for idx, (t, v, g, r) in enumerate(zip(ts, vs, gs, rs)):
        if v > stop_speed_m__s and not moving:
            if len(mt['cycSecs']) > 1:
                temp = make_cycle(
                    [mt['cycSecs'][-1]],
                    [mt['cycMps'][-1]],
                    [mt['cycGrade'][-1]],
                    [mt['cycRoadType'][-1]])
                mt['cycSecs'] = mt['cycSecs'] - mt['cycSecs'][0]
                for k in mt:
                    mt[k] = np.array(mt[k])
                microtrips.append(mt)
                mt = temp
        mt['cycSecs'] = np.append(mt['cycSecs'], [t])
        mt['cycMps'] = np.append(mt['cycMps'], [v])
        mt['cycGrade'] = np.append(mt['cycGrade'], [g])
        mt['cycRoadType'] = np.append(mt['cycRoadType'], [r])
        moving = v > stop_speed_m__s
    if len(mt['cycSecs']) > 0:
        mt['cycSecs'] = mt['cycSecs'] - mt['cycSecs'][0]
        microtrips.append(mt)
    if keep_name and "name" in cycle:
        for m in microtrips:
            m["name"] = cycle["name"]
    return microtrips


def make_cycle(ts, vs, gs=None, rs=None) -> dict:
    """
    (Array Num) (Array Num) (Array Num)? -> Dict
    Create a cycle from times, speeds, and grades. If grades is not
    specified, it is set to zero.
    Arguments:
    ----------
    ts: array of times [s]
    vs: array of vehicle speeds [mps]
    gs: array of grades
    rs: array of road types (charging or not)
    """
    assert len(ts) == len(vs)
    if gs is None:
        gs = np.zeros(len(ts))
    else:
        assert len(ts) == len(gs)
    if rs is None:
        rs = np.zeros(len(ts))
    else:
        assert len(ts) == len(rs)
    return {'cycSecs': np.array(ts),
            'cycMps': np.array(vs),
            'cycGrade': np.array(gs),
            'cycRoadType': np.array(rs)}


def equals(c1, c2, verbose=True) -> bool:
    """
    Dict Dict -> Bool
    Returns true if the two cycles are equal, false otherwise
    Arguments:
    ----------
    c1: cycle as dictionary from get_cyc_dict()
    c2: cycle as dictionary from get_cyc_dict()
    verbose: Bool, optional (default: True), if True, prints why not equal
    """
    if c1.keys() != c2.keys():
        if verbose:
            c2missing = set(c1.keys()) - set(c2.keys())
            c1missing = set(c2.keys()) - set(c1.keys())
            if len(c1missing) > 0:
                print('c2 keys not contained in c1: {}'.format(c1missing))
            if len(c2missing) > 0:
                print('c1 keys not contained in c2: {}'.format(c2missing))
        return False
    for k in c1.keys():
        if len(c1[k]) != len(c2[k]):
            if verbose:
                print(k + ' has a length discrepancy.')
            return False
        if np.any(np.array(c1[k]) != np.array(c2[k])):
            if verbose:
                print(k + ' has a value discrepancy.')
            return False
    return True


def concat(cycles, name=None):
    """
    Concatenates cycles together one after another into a single dictionary
    (Array Dict) String -> Dict
    Arguments:
    ----------
    cycles: (Array Dict)
    name: (optional) string or None, if a string, adds the "name" key to the output
    """
    final_cycle = {'cycSecs': np.array([]),
                   'cycMps': np.array([]),
                   'cycGrade': np.array([]),
                   'cycRoadType': np.array([])}
    keys = [k for k in final_cycle.keys()]
    first = True
    for cycle in cycles:
        if first:
            for k in keys:
                final_cycle[k] = np.array(cycle[k])
            first = False
        # if len(final_cycle['cycSecs']) == 0: # ???
        #     t0 = 0.0
        else:
            for k in keys:
                if k == 'cycSecs':
                    t0 = final_cycle[k][-1]
                    final_cycle[k] = np.concatenate([
                        final_cycle[k], np.array(cycle[k][1:]) + t0])
                else:
                    final_cycle[k] = np.concatenate([
                        final_cycle[k], np.array(cycle[k][1:])])
    if name is not None:
        final_cycle["name"] = name
    return final_cycle


def resample(cycle:Cycle, new_dt=None, start_time=None, end_time=None,
             hold_keys=None, rate_keys=None):
    """
    Cycle new_dt=?Real start_time=?Real end_time=?Real -> Cycle
    Resample a cycle with a new delta time from start time to end time.

    - cycle: Dict with keys
        'cycSecs': numpy.array Real giving the elapsed time
    - new_dt: Real, optional
        the new delta time of the sampling. Defaults to the
        difference between the first two times of the cycle passed in
    - start_time: Real, optional
        the start time of the sample. Defaults to 0.0 seconds
    - end_time: Real, optional
        the end time of the cycle. Defaults to the last time of the passed in
        cycle.
    - hold_keys: None or (Set String), if specified, yields values that
                 should be interpolated step-wise, holding their value until
                 an explicit change (i.e., NOT interpolated)
    - rate_keys: None or (Set String), if specified, yields values that maintain
                 the interpolated value of the given rate. So, for example,
                 if a speed, will set the speed such that the distance traveled
                 is consistent. Note: using rate keys for cycMps may result in
                 non-zero starting and ending speeds
    Resamples all non-time metrics by the new sample time.
    """
    if new_dt is None:
        new_dt = cycle['cycSecs'][1] - cycle['cycSecs'][0]
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = cycle['cycSecs'][-1]
    new_cycle = {}
    eps = new_dt / 10.0
    new_cycle['cycSecs'] = np.arange(start_time, end_time + eps, step=new_dt)
    for k in cycle:
        if k == 'cycSecs':
            continue
        elif hold_keys is not None and k in hold_keys:
            f = interp1d(cycle['cycSecs'], cycle[k], 0)
            new_cycle[k] = f(new_cycle['cycSecs'])
            continue
        elif rate_keys is not None and k in rate_keys:
            rate_var = np.array(cycle[k])
            # below is same as [(rate_var[i+1] + rate_var[i])/2.0 for i in range(len(rate_var) - 1)]
            avg_rate_var = (rate_var[1:] + rate_var[:-1]) / 2.0
            dts_orig = np.diff(cycle['cycSecs'])
            step_averages = np.diff(
                np.interp(
                    x=new_cycle['cycSecs'],
                    xp=cycle['cycSecs'],
                    fp=np.append(0.0, (avg_rate_var * dts_orig).cumsum())
                ),
            ) / new_dt
            step_averages = np.append(step_averages[0], step_averages)
            step_averages = np.append(step_averages, step_averages[-1])
            midstep_times = np.concatenate(
                (
                    [0.0],
                    np.arange(
                        start_time + (0.5 * new_dt), end_time - (0.5 * new_dt) + eps, step=new_dt),
                    [end_time],
                ))
            new_cycle[k] = np.interp(
                x=new_cycle['cycSecs'],
                xp=midstep_times,
                fp=step_averages
            )
            continue
        try:
            new_cycle[k] = np.interp(
                new_cycle['cycSecs'], cycle['cycSecs'], cycle[k])
        except:
            # if the value can't be interpolated, it must not be a numerical
            # array. Just add it back in as is.
            new_cycle[k] = deepcopy(cycle[k])
    return new_cycle


def clip_by_times(cycle, t_end, t_start=0):
    """
    Cycle Number Number -> Cycle
    INPUT:
    - cycle: Dict, a legitimate driving cycle
    - t_start: Number, time to start
    - t_end: Number, time to end
    RETURNS: Dict, the cycle with fields snipped
        to times >= t_start and <= t_end
    Clip the cycle to the given times and return
    """
    idx = np.logical_and(cycle['cycSecs'] >= t_start,
                         cycle['cycSecs'] <= t_end)
    new_cycle = {}
    for k in cycle:
        try:
            new_cycle[k] = np.array(cycle[k])[idx]
        except:
            new_cycle[k] = cycle[k]

    new_cycle['cycSecs'] -= new_cycle['cycSecs'][0] # reset time to start at zero
    return new_cycle


def accelerations(cycle):
    """
    Cycle -> Real
    Return the acceleration of the given cycle
    INPUTS:
    - cycle: Dict, a legitimate driving cycle
    OUTPUTS: Real, the maximum acceleration
    """
    accels = (np.diff(np.array(cycle['cycMps']))
              / np.diff(np.array(cycle['cycSecs'])))
    return accels


def peak_acceleration(cycle):
    """
    Cycle -> Real
    Return the maximum acceleration of the given cycle
    INPUTS:
    - cycle: Dict, a legitimate driving cycle
    OUTPUTS: Real, the maximum acceleration
    """
    return np.max(accelerations(cycle))


def peak_deceleration(cycle):
    """
    Cycle -> Real
    Return the minimum acceleration (maximum deceleration) of the given cycle
    INPUTS:
    - cycle: Dict, a legitimate driving cycle
    OUTPUTS: Real, the maximum acceleration
    """
    return np.min(accelerations(cycle))


@njit
def calc_constant_jerk_trajectory(n, D0, v0, Dr, vr, dt):
    """
    Num Num Num Num Num Int -> (Dict 'jerk_m__s3' Num 'accel_m__s2' Num)
    INPUTS:
    - n: Int, number of time-steps away from rendezvous
    - D0: Num, distance of simulated vehicle (m/s)
    - v0: Num, speed of simulated vehicle (m/s)
    - Dr: Num, distance of rendezvous point (m)
    - vr: Num, speed of rendezvous point (m/s)
    - dt: Num, step duration (s)
    RETURNS: (Tuple 'jerk_m__s3': Num, 'accel_m__s2': Num)
    Returns the constant jerk and acceleration for initial time step.
    """
    assert n > 1
    assert Dr > D0
    dDr = Dr - D0
    dvr = vr - v0
    k = (dvr - (2.0 * dDr / (n * dt)) + 2.0 * v0) / (
        0.5 * n * (n - 1) * dt
        - (1.0 / 3) * (n - 1) * (n - 2) * dt
        - 0.5 * (n - 1) * dt * dt
    )
    a0 = (
        (dDr / dt)
        - n * v0
        - ((1.0 / 6) * n * (n - 1) * (n - 2) * dt + 0.25 * n * (n - 1) * dt * dt) * k
    ) / (0.5 * n * n * dt)
    return (k, a0)


@njit
def accel_for_constant_jerk(n, a0, k, dt):
    """
    Calculate the acceleration n timesteps away
    INPUTS:
    - n: Int, number of times steps away to calculate
    - a0: Num, initial acceleration (m/s2)
    - k: Num, constant jerk (m/s3)
    - dt: Num, time-step duration in seconds
    NOTE:
    - this is the constant acceleration over the time-step from sample n to sample n+1
    RETURN: Num, the acceleration n timesteps away (m/s2)
    """
    return a0 + (n * k * dt)


@njit
def speed_for_constant_jerk(n, v0, a0, k, dt):
    """
    Int Num Num Num Num -> Num
    Calculate speed (m/s) n timesteps away
    INPUTS:
    - n: Int, numer of timesteps away to calculate
    - v0: Num, initial speed (m/s)
    - a0: Num, initial acceleration (m/s2)
    - k: Num, constant jerk
    - dt: Num, duration of a timestep (s)
    NOTE:
    - this is the speed at sample n
    - if n == 0, speed is v0
    - if n == 1, speed is v0 + a0*dt, etc.
    RETURN: Num, the speed n timesteps away (m/s)
    """
    return v0 + (n * a0 * dt) + (0.5 * n * (n - 1) * k * dt)


@njit
def dist_for_constant_jerk(n, d0, v0, a0, k, dt):
    """
    Calculate distance (m) after n timesteps
    INPUTS:
    - n: Int, numer of timesteps away to calculate
    - d0: Num, initial distance (m)
    - v0: Num, initial speed (m/s)
    - a0: Num, initial acceleration (m/s2)
    - k: Num, constant jerk
    - dt: Num, duration of a timestep (s)
    NOTE:
    - this is the distance traveled from start (i.e., n=0) measured at sample point n
    RETURN: Num, the distance at n timesteps away (m)
    """
    term1 = dt * (
        (n * v0)
        + (0.5 * n * (n - 1) * a0 * dt)
        + ((1.0 / 6.0) * k * dt * (n - 2) * (n - 1) * n)
    )
    term2 = 0.5 * dt * dt * ((n * a0) + (0.5 * n * (n - 1) * k * dt))
    return d0 + term1 + term2
