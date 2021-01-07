"""Module containing classes and methods for for loading vehicle and
cycle data. For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import re
import sys
from numba.experimental import jitclass                 # import the decorator
from numba import float64, int32, bool_, types    # import the types
import warnings
warnings.simplefilter('ignore')
from pathlib import Path
import copy

# local modules
from . import parameters as params
from .buildspec import build_spec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CYCLES_DIR = os.path.abspath(
        os.path.join(
            THIS_DIR, 'resources', 'cycles'))
STANDARD_CYCLE_KEYS = ['cycSecs', 'cycMps',
                       'cycGrade', 'cycRoadType', 'cycMph', 'secs', 'cycDistMeters']


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
        numba_cyc = CycleJit(len(self.cycSecs))
        for key in STANDARD_CYCLE_KEYS:
            numba_cyc.__setattr__(key, self.__getattribute__(key).astype(np.float64))
        return numba_cyc

    def set_standard_cycle(self, std_cyc_name):
        """Load time trace of speed, grade, and road type in a pandas
        dataframe.
        Argument:
        ---------
        std_cyc_name: cycle name string (e.g. 'udds', 'us06', 'hwfet')"""
        csv_path = os.path.join(CYCLES_DIR, std_cyc_name + '.csv')
        cyc = pd.read_csv(Path(csv_path))
        for column in cyc.columns:
            self.__setattr__(column, cyc[column].to_numpy())
        self.set_dependents()

    def set_from_file(self, cyc_file_path):
        """Load time trace of speed, grade, and road type from user-provided csv
        file in a pandas dataframe.
        Argument:
        ---------
        cyc_file_path: path to file containing cycle data"""
        cyc = pd.read_csv(cyc_file_path)
        for column in cyc.columns:
            self.__setattr__(column, cyc[column].to_numpy())
        self.set_dependents()

    def set_from_dict(self, cyc_dict):
        """Set cycle attributes from dict with keys 'cycSecs', 'cycMps',
        'cycGrade' (optional), 'cycRoadType' (optional) and numpy arrays of
        equal length for values.
        Arguments
        ---------
        cyc_dict: dict containing cycle data
        """
        for key in cyc_dict.keys():
            self.__setattr__(key, cyc_dict[key])
        # fill in unspecified optional values with zeros
        if 'cycGrade' not in cyc_dict.keys():
            self.__setattr__('cycGrade', np.zeros(len(self.cycMps)))
        if 'cycRoadType' not in cyc_dict.keys():
            self.__setattr__('cycRoadType', np.zeros(len(self.cycMps)))
        self.set_dependents()
    
    def set_dependents(self):
        """Sets values dependent on cycle info loaded from file."""
        self.cycMph = self.cycMps * params.mphPerMps
        self.secs = np.insert(np.diff(self.cycSecs), 0, 0) # time step deltas
        self.cycDistMeters = (self.cycMps * self.secs) 
        for key in self.__dir__():
            try:
                self.__setattr__(key, 
                    np.array(self.__getattribute__(key), dtype=np.float64))
            except:
                pass
    
    def get_cyc_dict(self):
        """Returns cycle as dict rather than class instance."""
        keys = ['cycSecs', 'cycMps', 'cycGrade', 'cycRoadType']
        
        cyc = {}
        for key in keys:
            cyc[key] = self.__getattribute__(key)
        
        return cyc
    
    def copy(self):
        """Return copy of Cycle instance."""
        cyc_dict = {'cycSecs': self.cycSecs,
                    'cycMps': self.cycMps,
                    'cycGrade': self.cycGrade,
                    'cycRoadType': self.cycRoadType
                    }
        cyc = Cycle(cyc_dict=cyc_dict)
        return cyc



# type specifications for attributes of Cycle class
cyc_spec = build_spec(Cycle('udds'))


@jitclass(cyc_spec)
class CycleJit(object):
    """Just-in-time compiled version of Cycle using numba."""
    
    def __init__(self, len_cyc):
        """This method initialized type numpy arrays as required by 
        numba jitclass."""
        self.cycSecs = np.zeros(len_cyc, dtype=np.float64)
        self.cycMps = np.zeros(len_cyc, dtype=np.float64)
        self.cycGrade = np.zeros(len_cyc, dtype=np.float64)
        self.cycRoadType = np.zeros(len_cyc, dtype=np.float64)
        self.cycMph = np.zeros(len_cyc, dtype=np.float64)
        self.secs = np.zeros(len_cyc, dtype=np.float64)
        self.cycDistMeters = np.zeros(len_cyc, dtype=np.float64)

    def copy(self):
        """Return copy of CycleJit instance."""
        cyc = CycleJit(len(self.cycSecs))
        cyc.cycSecs = np.copy(self.cycSecs)
        cyc.cycMps = np.copy(self.cycMps)
        cyc.cycGrade = np.copy(self.cycGrade)
        cyc.cycRoadType = np.copy(self.cycRoadType)
        cyc.cycMph = np.copy(self.cycMph)
        cyc.secs = np.copy(self.secs)
        cyc.cycDistMeters = np.copy(self.cycDistMeters)
        return cyc

    

def to_microtrips(cycle, stop_speed_m__s=1e-6):
    """
    Split a cycle into an array of microtrips with one microtrip being a start
    to subsequent stop plus any idle (stopped time).

    Arguments:
    ----------
    cycle: drive cycle converted to dictionary by cycle.get_cyc_dict()
    stop_speed_m__s: speed at which vehicle is considered stopped for trip
    separation
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


def equals(c1, c2):
    """
    Dict Dict -> Bool
    Returns true if the two cycles are equal, false otherwise
    Arguments:
    ----------
    c1: cycle as dictionary from get_cyc_dict()
    c2: cycle as dictionary from get_cyc_dict()
    """
    if c1.keys() != c2.keys():
        c2missing = set(c1.keys()) - set(c2.keys())
        c1missing = set(c2.keys()) - set(c1.keys())
        if len(c1missing) > 0:
            print('c2 keys not contained in c1: {}'.format(c1missing))
        if len(c2missing) > 0:
            print('c1 keys not contained in c2: {}'.format(c2missing))
        return False
    for k in c1.keys():
        if len(c1[k]) != len(c2[k]):
            print(k + ' has a length discrepancy.')
            return False
        if np.any(np.abs(np.array(c1[k]) - np.array(c2[k])) > 1e-6):
            print(k + ' has a value discrepancy.')
            return False
    return True


def concat(cycles):
    """
    (Array Dict) -> Dict
    Concatenates cycles together one after another
    """
    final_cycle = {'cycSecs': np.array([]),
                   'cycMps': np.array([]),
                   'cycGrade': np.array([]),
                   'cycRoadType': np.array([])}
    first = True
    for cycle in cycles:
        if first:
            final_cycle['cycSecs'] = np.array(cycle['cycSecs'])
            final_cycle['cycMps'] = np.array(cycle['cycMps'])
            final_cycle['cycGrade'] = np.array(cycle['cycGrade'])
            final_cycle['cycRoadType'] = np.array(cycle['cycRoadType'])
            first = False
        # if len(final_cycle['cycSecs']) == 0: # ???
        #     t0 = 0.0
        else:
            t0 = final_cycle['cycSecs'][-1]
            N_pre = len(final_cycle['cycSecs'])
            final_cycle['cycSecs'] = np.concatenate([
                final_cycle['cycSecs'],
                np.array(cycle['cycSecs'][1:]) + t0])
            final_cycle['cycMps'] = np.concatenate([
                final_cycle['cycMps'],
                np.array(cycle['cycMps'][1:])])
            final_cycle['cycGrade'] = np.concatenate([
                final_cycle['cycGrade'],
                np.array(cycle['cycGrade'][1:])])
    return final_cycle


def resample(cycle, new_dt=None, start_time=None, end_time=None,
             hold_keys=None):
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
