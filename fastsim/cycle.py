"""Module containing classes and methods for for loading 
cycle data. For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import re
import sys
from pathlib import Path
import copy
import types

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

    def get_cycMph(self):
        return self.cycMps * params.mphPerMps

    def set_cycMph(self, new_value):
        self.cycMps = new_value / params.mphPerMps

    cycMph = property(get_cycMph, set_cycMph)

    def get_time_s(self):
        return self.cycSecs

    def set_time_s(self, new_value):
        self.cycSecs = new_value

    time_s = property(get_time_s, set_time_s)

    # time step deltas
    @property
    def secs(self):
        return np.append(0.0, self.cycSecs[1:] - self.cycSecs[:-1]) 

    @property
    def dt_s(self):
        return self.secs
    
    # distance at each time step
    @property
    def cycDistMeters(self):
        return self.cycMps * self.secs
    
    @property
    def cycAvgMps(self):
        return np.insert(0.5 * (self.cycMps[1:] + self.cycMps[:-1]), 0, 0.0)
    
    @property
    def cycDistMeters_v2(self):
        return self.cycAvgMps * self.secs

    @property
    def delta_elev_m(self):
        """
        Cumulative elevation change w.r.t. to initial
        """
        return (self.cycDistMeters * self.cycGrade).cumsum()
    
    def get_cyc_dict(self):
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


def copy_cycle(cyc, return_dict=False, use_jit=None):
    """Returns copy of Cycle or CycleJit.
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
        cyc_dict[key] = cyc.__getattribute__(key)
        
    if return_dict:
        return cyc_dict
    
    if use_jit is None:
        if type(cyc) == Cycle:
            cyc = Cycle(cyc_dict=cyc_dict)
        else:
            cyc = Cycle(cyc_dict=cyc_dict).get_numba_cyc()
    elif use_jit:
        cyc = Cycle(cyc_dict=cyc_dict).get_numba_cyc()
    else:
        cyc = Cycle(cyc_dict=cyc_dict)
        
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


def make_cycle(ts, vs, gs=None, rs=None):
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


def equals(c1, c2, verbose=True):
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


def resample(cycle, new_dt=None, start_time=None, end_time=None,
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
                    fp=np.insert((avg_rate_var * dts_orig).cumsum(), 0, 0.0)
                ),
            ) / new_dt
            step_averages = np.insert(step_averages, 0, step_averages[0])
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
            new_cycle[k] = copy.deepcopy(cycle[k])
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


TOL = 1e-6


def bracket(n, nmin, nmax):
    """
    Brackets a number, n, between a low and high limit
    - n: Number, some number
    - nmin: (or Number None), the minimum value for number
    - nmax: (or Number None), the maximum value for number
    RETURN: Number
    TODO: switch to np.clip
    """
    assert (nmin is None) or (nmax is None) or (nmax >= nmin)
    if nmax is not None and n > nmax:
        return nmax
    elif nmin is not None and n < nmin:
        return nmin
    return n


def calc_adjusted_brake_accel(v0, dts0, dt, a_brake_min, a_brake_max):
    """
    Calculate the adjusted brake acceleration. Attempts to
    decelerate such that we cover the given distance to stop
    within the constraints of a_brake_min and a_brake_max.
    - v0: non-negative number, current vehicle speed (m/s)
    - dts0: (or number None), current distance to stop from v0 (m, can be None)
    - dt: positive number, the time-step duration (s)
    - a_brake_min: number, the minimum brake acceleration (m/s2)
    - a_brake_max: number, the maximum brake acceleration (m/s2)
    Note: a_brake_min and a_brake_max should be negative. Also:
        assumes: |a_brake_min| >= |a_brake_max|
    RETURN: number, the braking acceleration (m/s2)
    """
    assert np.abs(a_brake_min) >= np.abs(a_brake_max)
    assert a_brake_min < 0.0
    assert a_brake_max < 0.0
    assert v0 >= 0.0
    assert dt > 0.0
    if v0 == 0.0:
        return 0.0
    if dts0 is not None:
        assert dts0 >= 0.0
        v_avg_brake = v0 / 2.0
        if v_avg_brake > 0.0:
            dt_brake_remaining = dts0 / v_avg_brake
        else:
            dt_brake_remaining = None
        if v0 <= min(a_brake_min, a_brake_max) * dt:
            a_brake_adj = -v0 / dt
        elif dt_brake_remaining is None or dt_brake_remaining <= 0.0:
            a_brake_adj = a_brake_min
        else:
            a_brake_adj = bracket(-v0 / dt_brake_remaining, a_brake_min, a_brake_max)
    else:
        a_brake_adj = a_brake_min
    return a_brake_adj


def next_speed_by_eco_approach(
    v0,
    dt,
    a_brake,
    a_brake_min,
    dts0,
    a_coast,
    v_brake_start,
    dtlv0,
    d0_lv,
    d1_lv,
    min_ttc,
):
    """
    Given the state, parameters, and derived values, return
    the next speed under eco-approach in meters/second.
    K = dv/dd = constant
    a(t) = K * v(t)
    a(t)/v(t) = K
    v1 = v0 + integral(K * v(t), 0, dt) = v0 + K * integral(v, 0, dt)
                                        = v0 + K * delta_dist_0_to_1
    1: (v1 - v0) / delta_dist_0_to_1 = K
    2: delta_dist_0_to_1 = (v0 + v1) * dt / 2.0
    substitute 2 into 1:
        (v1 - v0) / ((v0 + v1) * dt / 2.0) = K
        (v1 - v0) / (v0 + v1) = 0.5 * K * dt
        (v1 - v0) = (v0 + v1) * 0.5 * K * dt
        v1 = v0 + 0.5 * K * (v0 + v1) * dt
    INPUTS
    - v0: non-negative number, vehicle speed at the beginning of the step (i.e., last step's speed achieved) (m/s)
    - dt: positive number, the step duration (s)
    - a_brake: negative number, nominal braking acceleration (m/s2)
    - a_brake_min: negative number, maximum braking acceleration that will be entertained (m/s2)
    - dts0: non-negative number, the distance to the next stop from the start of the time-step (m)
    - a_coast: number, the coasting acceleration (m/s2)
    - v_brake_start: positive number, the speed at which friction braking engages (m/s)
    - dtlv0: non-negative number, distance to lead vehicle from start of time-step (m)
    - d0_lv: non-negative number, distance of the lead vehicle at beginning of step (m)
    - d1_lv: non-negative number, distance of the lead vehicle at end of step (m)
    - min_ttc: positive number, the minimum time allowed for closing the distance to lead vehicle (s)
    NOTE:
    - a_brake >= a_brake_min (i.e., abs(a_brake) <= abs(a_brake_min))
    RETURN: number, the next speed (m/s)
    TODO:
    - investigate: doesn't look like this takes into account positive coasting accelerations (e.g., going down hill and picking up speed)
    """
    dv_coast = a_coast * dt
    v1_coast = bracket(v0 + dv_coast, 0.0, None)
    dv_brake = a_brake * dt
    v1_brake = bracket(v0 + dv_brake, 0.0, None)
    if v0 > v_brake_start and v1_coast > v_brake_start:
        v1 = v1_coast
    elif v0 <= v_brake_start:
        a_brake_adj = calc_adjusted_brake_accel(v0, dts0, dt, a_brake, a_brake_min)
        v1 = bracket(v0 + a_brake_adj * dt, 0.0, None)
    else:
        # treat coasting like constant acceleration
        # TODO: revisit this; doesn't look like it preserves distance integration
        dt_coast = (v0 - v_brake_start) / a_coast if a_coast != 0.0 else dt
        dd_coast = (v0 + v_brake_start) * dt_coast / 2.0
        dt_brake = dt - dt_coast
        a_brake_adj = calc_adjusted_brake_accel(
            v0, dts0 - dd_coast, dt_brake, a_brake, a_brake_min
        )
        v1 = v0 + dt_coast * a_coast + dt_brake * a_brake_adj
    vavg = (v0 + v1) / 2.0
    dd = vavg * dt
    dd_lv = d1_lv - d0_lv
    dtlv1 = dtlv0 + dd_lv - dd
    if dd_lv > dd or v1 < v_brake_start:
        return v1
    if v1 > 0.0:
        ttc1 = dtlv1 / v1
    else:
        ttc1 = None
    if ttc1 is not None and ttc1 >= min_ttc:
        return v1
    # In order to never collide, dd <= dd_lv
    # dd = vavg_never_collide * dt = (v0 + v1_never_collide) * dt / 2.0
    # (v0 + v1_never_collide) / 2.0 = dd_lv / dt
    # v0 + v1_never_collide = 2.0 * (dd_lv / dt)
    # v1_never_collide = 2.0 * (dd_lv / dt) - v0
    # however, we won't go below what we could get from braking nor below 0
    if ttc1 is not None:
        f = bracket(ttc1 / min_ttc, 0.1, 0.9)
    else:
        f = 0.9
    v1_never_collide = bracket(2.0 * (dd_lv / dt) - v0, v1_brake, None)
    return bracket(v1 * f + v1_never_collide * (1.0 - f), 0.0, None)


def limit_next_speed(
    v1_proposed,
    dtlv0,
    v1_lv,
    v0,
    v0_lv,
    dt,
    a_brake,
    step,
    t0,
    d0,
    lv_ds,
    total_cycle_distance_m,
    total_cycle_time_s,
    d1_lv,
    d2_lv,
):
    """
    Limit the next speed due to considerations of crash avoidance with leading vehicle.
    - v1_proposed: non-negative number, proposed next speed at end of step (m/s)
    - dtlv0: non-negative number, distance to lead vehicle from start of time-step (m)
    - v1_lv: non-negative number, the current speed of the lead vehicle (m/s)
    - v0: non-negative number, vehicle speed at start of step (m/s)
    - v0_lv: non-negative number, vehicle speed of lead vehicle at start of step (m/s)
    - dt: positive number, time-step duration (s)
    - a_brake: negative number, nominal braking acceleration (via engaging friction brakes) (m/s2)
    - step: non-negative integer, the step index; corresponds to i-1 in FASTSim
    - t0: non-negative number, elapsed time at step-start (s)
    - d0: non-negative number, distance traveled from start at step-start (m)
    - lv_ds: (array non-negative-number), lead-vehicle distances traveled from start (m)
    - total_cycle_distance_m: non-negative number, total distance over the whole cycle (i.e., total distance traveled over the cycle) (m)
    - total_cycle_time_s: non-negative number, total cycle elapsed time (s)
    - d1_lv: non-negative number, distance of lead vehicle at end of this time-step (m)
    - d2_lv: non-negative number, distance of lead vehicle at end of NEXT time-step (m)
    RETURN: non-negative number, limited next speed (m/s)
    """
    if (
        np.abs(dtlv0) < TOL
        and np.abs(v0_lv - v0) < TOL
        and np.abs(v1_lv - v1_proposed) < TOL
    ):
        # If we're in step with lead vehicle, don't attempt to limit speed
        return v1_proposed
    dv_brake = a_brake * dt
    dv_brake_abs = np.abs(dv_brake)
    # if v1_proposed < dv_brake_abs:
    #    # don't regulate speed if we can go to complete stop in one time step
    #    return v1_proposed
    # Let's assume that lead-vehicle data is resampled to dtime_s periods at initialization
    num_steps_to_stop = int(np.ceil(v1_proposed / dv_brake_abs))
    dd_proposed = (v0 + v1_proposed) * dt / 2.0
    d1_proposed = d0 + dd_proposed
    idx0 = step + 1
    idx1 = idx0 + num_steps_to_stop
    lv_ds = lv_ds[idx0:idx1]
    D = total_cycle_distance_m
    T = total_cycle_time_s
    dtd = D - d0
    dtd1 = dtd - dd_proposed
    ttd = T - t0
    ttd1 = ttd - dt
    if ttd > 0.0:
        a_dest_dist0 = (dtd - v0 * ttd) / (0.5 * ttd * ttd)
        if ttd1 > 0.0:
            a_dest_dist1 = (dtd1 - v1_proposed * ttd1) / (0.5 * ttd1 * ttd1)
        else:
            a_dest_dist1 = 0.0
    else:
        a_dest_dist0 = 0.0
        a_dest_dist1 = 0.0
    a_dist0 = min(
        [0.0]
        + [
            (d_lv - d0 - v0 * (idx + 1) * dt) / (0.5 * ((idx + 1) * dt) ** 2)
            for idx, d_lv in enumerate(lv_ds)
        ]
    )
    a_dist1 = min(
        [0.0]
        + [
            (d_lv - d1_proposed - v1_proposed * idx * dt) / (0.5 * (idx * dt) ** 2)
            for idx, d_lv in enumerate(lv_ds)
            if idx > 0
        ]
    )
    a_min0 = min(a_dist0, a_dest_dist0, 0.0)
    a_min1 = min(a_dist1, a_dest_dist1, 0.0)
    if a_min0 > 0.9 * a_brake and a_min1 <= 0.9 * a_brake:
        v1_max = v0 + a_min0 * dt
    elif a_min0 <= 0.9 * a_brake:
        v1_max = v0 + a_brake * dt
    else:
        v1_max = v1_proposed
    dd_nocrash = min(d2_lv, D) - d0
    v1_lim_dist = dd_nocrash / dt - v0 / 2.0
    vavg_nocrash = (min(d1_lv, D) - d0) / dt
    v1_nocrash = bracket(vavg_nocrash * 2.0 - v0, 0.0, None)
    v1 = bracket(v1_proposed, 0.0, max(min(v1_nocrash, v1_lim_dist, v1_max), 0.0))
    # v1 = bracket(v1_proposed, 0.0, max(min(v1_max, v1_nocrash), 0.0))
    return v1


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
    RETURNS: (Dict 'jerk_m__s3' Num 'accel_m__s2' Num)
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
    return {"jerk_m__s3": k, "accel_m__s2": a0}


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
    - this is the speed for the start of the time-step at n
    RETURN: Num, the speed n timesteps away (m/s)
    """
    return v0 + (n * a0 * dt) + (0.5 * n * (n - 1) * k * dt)


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


def should_impose_coast(cyc0, cyc, idx, coasting_factors=None, coast_start_speed_m__s=None):
    """
    - cyc0: Cycle, reference cycle
    - cyc: Cycle, current cycle
    - idx: non-negative integer, the current position in cyc
    - coasting_factors: (dict (tuple speed_m__s, grade_pct) dv__dd), yields
      the change in speed by distance by speed and by grade
      speed_m__s (m/s, >= 0), grade_pct (%), and dv__dd (m/s / m) are all numbers
    - coast_start_speed_m__s: (or None non-negative number), if greater than or equal to the given speed, initiate coast
    RETURN: Bool if vehicle should initiate coasting
    Coast logic is that the vehicle should coast if it is within coasting distance of a stop
    """
    if coast_start_speed_m__s is not None:
        return cyc.cycMps[idx] >= coast_start_speed_m__s
    return False


def calc_next_rendezvous_trajectory(
    cyc,
    idx,
    cyc0,
    time_horizon_s,
    distance_horizon_m,
    min_accel_m__s2,
    max_accel_m__s2,
    brake_start_speed_m__s,
    brake_accel_m__s2,
):
    """
    - cyc: Cycle, the cycle dictionary for the cycle to be modified
    - idx: non-negative integer, the index into cyc for the start-of-step
    - cyc0: Cycle, the lead cycle for reference
    - time_horizon_s: positive number, the time ahead to look for rendezvous opportunities (s)
    - distance_horizon_m: positive number, the distance ahead to detect rendezvous opportunities (m)
    - min_accel_m__s2: number, the minimum acceleration permitted (m/s2)
    - max_accel_m__s2: number, the maximum acceleration permitted (m/s2)
    - brake_start_speed_m__s2: non-negative number, speed at which friction brakes engage (m/s)
    - brake_accel_m__s2: negative number, the brake acceleration (m/s2)
    RETURN: None or {
        "distance_m" non-negative number, the distance from start of cycle to rendezvous at (m)
        "speed_m__s" non-negative number, the speed to rendezvous at,
        "n" positive integer, the number of steps ahead to rendezvous at
        "k" number, the Jerk or first derivative of acceleration m/s3
        "accel_m__s2", number, the initial acceleration of the trajectory
    }
    If no rendezvous exists within the scope, then returns None
    Otherwise, returns the next closest rendezvous in time/space

    # - dts0 <- determine the distance to the next stop from the start of step
    # - dtb <- the distance to brake by braking speed from brake initialization speed (m)
    # - dtbi0 <- distance to brake initiation; dts0 - dtb
    # - Check for rendezvous opportunities with lead vehicle and with brake initiation distance 
    #   for up to or equal to X seconds ahead (or end of cycle) by
    #   - calculating the constant-jerk trajectory to rendezvous with lead vehicle at time-steps up to X seconds ahead (or end of cycle)
    #   - calculating the constant-jerk trajectory to rendezvous with dtbi0 at time-steps up to X seconds ahead (or end of cycle)
    #   - drop trajectories unless all accelerations on trajectory are less than the current acceleration
    #   - for those remaining, pick the one with the highest min-acceleration
    #   - if none match stick with proposed target-speed, else start implementing constant jerk
    """
    TOL = 1e-6
    # v0 is where n=0, i.e., idx-1
    v0 = cyc.cycMps[idx-1]
    if v0 < (brake_start_speed_m__s + TOL):
        # don't process braking
        return None
    if min_accel_m__s2 > max_accel_m__s2:
        min_accel_m__s2, max_accel_m__s2 = max_accel_m__s2, min_accel_m__s2
    num_samples = len(cyc.cycMps)
    d0 = cyc.cycDistMeters_v2[:idx].sum()
    v1 = cyc.cycMps[idx]
    dt = cyc.secs[idx]
    a_proposed = (v1 - v0) / dt
    # distance to stop from start of time-step
    dts0 = calc_distance_to_next_stop(d0, cyc0) 
    if dts0 < TOL:
        # no stop to coast towards or we're there...
        return None
    dt = cyc.secs[idx]
    # distance to brake from the brake start speed (m/s)
    dtb = -0.5 * brake_start_speed_m__s * brake_start_speed_m__s / brake_accel_m__s2
    # distance to brake initiation from start of time-step (m)
    dtbi0 = dts0 - dtb
    cyc0_distances_m = cyc0.cycDistMeters_v2.cumsum()
    # Now, check rendezvous trajectories
    if time_horizon_s > 0.0:
        step_idx = idx
        dt_plan = 0.0
        r_best = None
        while dt_plan <= time_horizon_s and step_idx < num_samples:
            dt_plan += cyc0.secs[step_idx]
            step_ahead = step_idx - (idx - 1)
            if step_ahead == 1:
                # for brake init rendezvous
                accel = (brake_start_speed_m__s - v0) / dt
                r_bi = {
                    'accel_m__s2': accel,
                    'jerk_m__s3': 0.0,
                    'n': 1,
                    'min_accel_m__s2': accel,
                    'max_accel_m__s2': accel,
                    'mean_accel_m__s2': accel,
                }
                v1 = max(0.0, v0 + accel * dt)
                dd_proposed = ((v0 + v1) / 2.0) * dt
                if np.abs(v1 - brake_start_speed_m__s) < TOL and np.abs(dtbi0 - dd_proposed) < TOL:
                    r_best = r_bi
                if False:
                    # lead-vehicle rendezvous
                    dtlv0 = cyc0_distances_m[step_idx] - dist_traveled_m # distance to lead vehicle from step start
                    accel = (self.cyc0.cycMps[step_idx] - v0) / dt
                    r_lv = {
                        'accel_m__s2': accel,
                        'min_accel_m__s2': accel,
                        'max_accel_m__s2': accel,
                    }
                    dd_proposed = ((v0 + v0 + accel * dt) / 2.0) * dt
                    if np.abs(dtlv0 - dd_proposed) < TOL:
                        r_best = r_lv
                if r_best is not None:
                    # return if we have a single-step solution
                    print(f"We have a single-step solution [{idx}]: {str(r_best)}")
                    return r_best
            else:
                if False:
                    # rendezvous trajectory for lead vehicle -- assumes fixed time-steps
                    r_lv = calc_constant_jerk_trajectory(
                        step_ahead, dist_traveled_m, v0, cyc0_distances_m[step_idx], self.cyc0.cycMps[step_idx], dt)
                    if r_lv['accel_m__s2'] < a_proposed:
                        # accelerations by step for the rendezvous with lead vehicle
                        as_r_lv = np.array([
                            accel_for_constant_jerk(n, r_lv['accel_m__s2'], r_lv['jerk_m__s3'], dt)
                            for n in range(1, step_ahead + 1)
                        ])
                        #if r_best is None or as_r_lv.min() > r_best['min_accel_m__s2']:
                        if (as_r_lv <= a_proposed).all() and (r_best is None or as_r_lv.max() < r_best['min_accel_m__s2']):
                            r_best = r_lv
                            r_best['min_accel_m__s2'] = as_r_lv.min()
                            r_best['max_accel_m__s2'] = as_r_lv.max()
                if dtbi0 > TOL:
                    # rendezvous trajectory for brake-start -- assumes fixed time-steps
                    r_bi = calc_constant_jerk_trajectory(
                        step_ahead, 0.0, v0, dtbi0, brake_start_speed_m__s, dt)
                    if r_bi['accel_m__s2'] < max_accel_m__s2 and r_bi['accel_m__s2'] > min_accel_m__s2 and r_bi['jerk_m__s3'] >= 0.0:
                        as_bi = np.array([
                            accel_for_constant_jerk(n, r_bi['accel_m__s2'], r_bi['jerk_m__s3'], dt)
                            for n in range(step_ahead)
                        ])
                        accel_spread = np.abs(as_bi.max() - as_bi.min())
                        flag = (
                            (as_bi.max() < (max_accel_m__s2 + TOL) and as_bi.min() > (min_accel_m__s2 - TOL))
                            and
                            (r_best is None or (accel_spread < r_best['accel_spread_m__s2']))
                        )
                        if flag:
                            r_best = r_bi
                            r_best['n'] = step_ahead
                            r_best['max_accel_m__s2'] = as_bi.max()
                            r_best['min_accel_m__s2'] = as_bi.min()
                            r_best['mean_accel_m__s2'] = as_bi.mean()
                            r_best['accel_spread_m__s2'] = accel_spread
            step_idx += 1
        if r_best is not None:
            return r_best
    return None


def calc_distance_to_next_stop(distance_m, cyc):
    """
    Calculate the distance to the next stop given the current distance position and a cycle
    - distance_m: non-negative number, the current distance from start (m)
    - cyc: Cycle, the cycle to use to evaluate next stop distances
    RETURN: non-negative number, the distance from the current position to the next stop (m)
    NOTE:
    - If there is no next stop or if we are at the stop, 0.0 is returned
    """
    distances_of_stops_m = np.unique(cyc.cycDistMeters_v2.cumsum()[cyc.cycMps < TOL])
    remaining_stops_m = distances_of_stops_m[distances_of_stops_m > (distance_m + TOL)]
    if len(remaining_stops_m) > 0:
        return remaining_stops_m[0] - distance_m
    return 0.0


def modify_cycle_with_trajectory(cyc, idx, n, jerk_m__s3, accel0_m__s2):
    """
    Modifies cyc using the given trajectory parameters
    - cyc: Cycle, the cycle dictionary for the cycle to be modified
    - idx: non-negative integer, the point in the cycle to initiate
      modification (note: THIS point is modified since trajectory should be calculated from idx-1)
    - jerk_m__s3: number, the "Jerk" associated with the trajectory (m/s3)
    - accel0_m__s2: number, the initial acceleration (m/s2)
    NOTE:
    - also resamples cycle grade based on distance by mapping to elevation points
    - modifies cyc in place to hit any critical rendezvous_points by a trajectory adjustment
    - NOT ROBUST AGAINST VARIABLE DURATION TIME-STEPS
    RETURN: Number, final modified speed (m/s)
    TODO: need to update to resample grade by distance
    """
    num_samples = len(cyc.cycSecs)
    v0 = cyc.cycMps[idx-1]
    dt = cyc.secs[idx]
    v = v0
    for ni in range(1, n+1):
        idx_to_set = (idx - 1) + ni
        if idx_to_set >= num_samples:
            break
        v = speed_for_constant_jerk(ni, v0, accel0_m__s2, jerk_m__s3, dt)
        cyc.cycMps[idx_to_set] = v
    return v


def modify_cycle_adding_braking_trajectory(cyc, brake_accel_m__s2, idx):
    """
    - cyc: Cycle, the cycle to modify
    - brake_accel_m__s2: negative number, the braking acceleration (m/s2)
    - idx: non-negative integer, the index where to initiate the stop trajectory, start of the step (i in FASTSim)
    TODO:
    - resample grade to be correct by distance
    """
    assert brake_accel_m__s2 < 0.0
    v0 = cyc.cycMps[idx-1]
    dt = cyc.secs[idx]
    # begin braking deceleration
    dts_m = 0.5 * v0 * v0 / np.abs(brake_accel_m__s2)
    tts_s = -v0 / brake_accel_m__s2
    n = int(np.ceil(tts_s / dt))
    trajectory = calc_constant_jerk_trajectory(n, 0.0, v0, dts_m, 0.0, dt)
    modify_cycle_with_trajectory(cyc, idx, n, trajectory['jerk_m__s3'], trajectory['accel_m__s2'])
