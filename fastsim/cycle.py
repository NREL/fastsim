"""Module containing classes and methods for 
cycle data. For example usage, see ../README.md"""

### Import necessary python modules
from dataclasses import dataclass
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
from . import utils
from fastsimrust import Cycle

THIS_DIR = Path(__file__).parent
CYCLES_DIR = THIS_DIR / 'resources' / 'cycles'

# dicts for switching between legacy and new cycle keys
OLD_TO_NEW = {
    'cycSecs': 'time_s',
    'cycMps': 'mps',
    'cycGrade': 'grade',
    'cycRoadType': 'road_type',
    'name': 'name'
}
NEW_TO_OLD = {val: key for key, val in OLD_TO_NEW.items()}
STANDARD_CYCLE_KEYS = OLD_TO_NEW.values()

@dataclass
class Cycle(object):
    """Object for containing time, speed, road grade, and road charging vectors 
    for drive cycle.  Instantiate with the `from_file` or `from_dict` method.  
    """

    time_s: np.ndarray(1, dtype=float)
    mps: np.ndarray(1, dtype=float)
    grade: np.ndarray(1, dtype=float)
    road_type: np.ndarray(1, dtype=float)
    name: str

    @classmethod
    def from_file(cls, filename:str):
        """
        Load cycle from filename (str).
        Can be absolute or relative path.  If relative, looks in working dir first
        and then in `fastsim/resources/cycles`.  
        
        File must contain columns for:
        -- `cycSecs` or `time_s`
        -- `cycMps` or `mps`
        -- `cycGrade` or `grade` (optional)
        -- `cycRoadType` or `road_type` (optional)
        """
        filename = str(filename)

        if not filename.endswith('.csv'):
            filename += ".csv"
        if not Path(filename).exists() and (CYCLES_DIR / filename).exists():
            filename = CYCLES_DIR / filename
        elif Path(filename).exists():
            filename = Path(filename) 
        else:
            raise ValueError("Invalid cycle filename.")
            
        cyc_df = pd.read_csv(filename)
        cyc_dict = cyc_df.to_dict(orient='list')
        cyc_dict = {key:np.array(val, dtype=float) for key, val in cyc_dict.items()}
        cyc_dict['name'] = filename.stem

        return cls.from_dict(cyc_dict)

    @classmethod
    def from_dict(cls, cyc_dict:dict):
        """
        Load cycle from dict, which must contain keys for:
        -- `cycSecs` or `time_s`
        -- `cycMps` or `mps`
        -- `cycGrade` or `grade` (optional)
        -- `cycRoadType` or `road_type` (optional)
        """
        new_cyc_dict = {}
        for key, val in cyc_dict.items():
            # generate keys from legacy or current mapping
            new_cyc_dict[OLD_TO_NEW.get(key, key)] = val
        new_cyc_dict['name'] = cyc_dict.get('name', '')
        # set zeros if not provided
        new_cyc_dict['grade'] = new_cyc_dict.get(
            'grade', np.zeros(len(new_cyc_dict['time_s'])))
        new_cyc_dict['road_type'] = new_cyc_dict.get(
            'road_type', np.zeros(len(new_cyc_dict['time_s'])))

        # check for invalid keys
        assert len(set(new_cyc_dict.keys()) - set(STANDARD_CYCLE_KEYS)) == 0
        return cls(**new_cyc_dict)

    def get_numba_cyc(self):
        """Deprecated."""
        raise NotImplementedError("This method has been deprecated.  Use get_rust_cyc instead.")

    ### Properties

    def get_mph(self) -> np.ndarray:
        return self.mps * params.MPH_PER_MPS

    def set_mph(self, new_value):
        self.mps = new_value / params.MPH_PER_MPS

    mph = property(get_mph, set_mph)

    # time step deltas
    @property
    def dt_s(self) -> np.ndarray:
        return np.array(np.diff(self.time_s, prepend=0), dtype=float)
    
    # distance at each time step
    @property
    def dist_m(self) -> np.ndarray:
        return self.mps * self.dt_s

    @property
    def delta_elev_m(self):
        """
        Cumulative elevation change w.r.t. to initial
        """
        return (self.dist_m * self.grade).cumsum()

    @property
    def len(self) -> int:
        "return cycle length"
        return len(self.time_s)
    
    def get_cyc_dict(self) -> dict:
        """Returns cycle as dict rather than class instance."""
        keys = STANDARD_CYCLE_KEYS
        
        cyc = {}
        for key in keys:
            cyc[key] = copy.deepcopy(self.__getattribute__(key))
        
        return cyc
    
class LegacyCycle(object):
    """
    Implementation of Cycle with legacy keys.
    """
    def __init__(self, cycle:Cycle):
        """
        Given cycle, returns legacy cycle.
        """
        for key, val in NEW_TO_OLD.items():
            self.__setattr__(val, copy.deepcopy(cycle.__getattribute__(key)))


def copy_cycle(cyc:Cycle, return_type:str='cycle', deep:bool=True):
    """Returns copy of Cycle.
    Arguments:
    cyc: instantianed Cycle or CycleJit
    return_type: 
        'dict': dict
        'cycle': Cycle 
        'legacy_cycle': LegacyCycle
        'rust_cycle': RustCycle
    deep: if True, uses deepcopy on everything
    """

    cyc_dict = {}

    for key in utils.get_attrs(cyc):
        val_to_copy = cyc.__getattribute__(key)
        if type(val_to_copy) == np.ndarray:
            # has to be float or time_s will get converted to int
            cyc_dict[key] = np.array(copy.deepcopy(val_to_copy) if deep else val_to_copy, dtype=float)
        else:
            cyc_dict[key] = copy.deepcopy(val_to_copy) if deep else val_to_copy

    if return_type == 'dict':
        return cyc_dict
    elif return_type == 'cycle':
        return Cycle.from_dict(cyc_dict)
    elif return_type == 'legacy_cycle':
        return LegacyCycle(cyc_dict)
    elif return_type == 'rust_cycle':
        raise NotImplementedError
    else:
        raise ValueError("Invalid return_type.")
        

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
    ts = np.array(cycle['time_s'])
    vs = np.array(cycle['mps'])
    gs = np.array(cycle['grade'])
    rs = np.array(cycle['road_type'])
    mt = make_cycle([], [])
    moving = False
    for idx, (t, v, g, r) in enumerate(zip(ts, vs, gs, rs)):
        if v > stop_speed_m__s and not moving:
            if len(mt['time_s']) > 1:
                temp = make_cycle(
                    [mt['time_s'][-1]],
                    [mt['mps'][-1]],
                    [mt['grade'][-1]],
                    [mt['road_type'][-1]])
                mt['time_s'] = mt['time_s'] - mt['time_s'][0]
                for k in mt:
                    mt[k] = np.array(mt[k])
                microtrips.append(mt)
                mt = temp
        mt['time_s'] = np.append(mt['time_s'], [t])
        mt['mps'] = np.append(mt['mps'], [v])
        mt['grade'] = np.append(mt['grade'], [g])
        mt['road_type'] = np.append(mt['road_type'], [r])
        moving = v > stop_speed_m__s
    if len(mt['time_s']) > 0:
        mt['time_s'] = mt['time_s'] - mt['time_s'][0]
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
    return {'time_s': np.array(ts),
            'mps': np.array(vs),
            'grade': np.array(gs),
            'road_type': np.array(rs)}


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
    final_cycle = {'time_s': np.array([]),
                   'mps': np.array([]),
                   'grade': np.array([]),
                   'road_type': np.array([])}
    keys = [k for k in final_cycle.keys()]
    first = True
    for cycle in cycles:
        if first:
            for k in keys:
                final_cycle[k] = np.array(cycle[k])
            first = False
        # if len(final_cycle['time_s']) == 0: # ???
        #     t0 = 0.0
        else:
            for k in keys:
                if k == 'time_s':
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
        'time_s': numpy.array Real giving the elapsed time
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
                 is consistent. Note: using rate keys for mps may result in
                 non-zero starting and ending speeds
    Resamples all non-time metrics by the new sample time.
    """
    if new_dt is None:
        new_dt = cycle['time_s'][1] - cycle['time_s'][0]
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = cycle['time_s'][-1]
    new_cycle = {}
    eps = new_dt / 10.0
    new_cycle['time_s'] = np.arange(start_time, end_time + eps, step=new_dt)
    for k in cycle:
        if k == 'time_s':
            continue
        elif hold_keys is not None and k in hold_keys:
            f = interp1d(cycle['time_s'], cycle[k], 0)
            new_cycle[k] = f(new_cycle['time_s'])
            continue
        elif rate_keys is not None and k in rate_keys:
            rate_var = np.array(cycle[k])
            # below is same as [(rate_var[i+1] + rate_var[i])/2.0 for i in range(len(rate_var) - 1)]
            avg_rate_var = (rate_var[1:] + rate_var[:-1]) / 2.0
            dts_orig = np.diff(cycle['time_s'])
            step_averages = np.diff(
                np.interp(
                    x=new_cycle['time_s'],
                    xp=cycle['time_s'],
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
                x=new_cycle['time_s'],
                xp=midstep_times,
                fp=step_averages
            )
            continue
        try:
            new_cycle[k] = np.interp(
                new_cycle['time_s'], cycle['time_s'], cycle[k])
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
    idx = np.logical_and(cycle['time_s'] >= t_start,
                         cycle['time_s'] <= t_end)
    new_cycle = {}
    for k in cycle:
        try:
            new_cycle[k] = np.array(cycle[k])[idx]
        except:
            new_cycle[k] = cycle[k]

    new_cycle['time_s'] -= new_cycle['time_s'][0] # reset time to start at zero
    return new_cycle


def accelerations(cycle):
    """
    Cycle -> Real
    Return the acceleration of the given cycle
    INPUTS:
    - cycle: Dict, a legitimate driving cycle
    OUTPUTS: Real, the maximum acceleration
    """
    accels = (np.diff(np.array(cycle['mps']))
              / np.diff(np.array(cycle['time_s'])))
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
