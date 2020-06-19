"""Module containing classes and methods for for loading vehicle and cycle data.
For example usage, see ../README.md"""

### Import necessary python modules
import os
import numpy as np
import pandas as pd
import re
import sys
from numba import jitclass                 # import the decorator
from numba import float64, int32, bool_, types    # import the types
import warnings
warnings.simplefilter('ignore')
from pathlib import Path
import ast

# local modules
from . import globalvars as gl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CYCLES_DIR = os.path.abspath(
        os.path.join(
            THIS_DIR, '..', 'cycles'))

class Cycle(object):
    """Object for containing time, speed, road grade, and road charging vectors 
    for drive cycle."""
    def __init__(self, std_cyc_name=None, cyc_dict=None, cyc_file_path=None):
        """Runs other methods, depending on provided keyword argument. Only one keyword
        argument should be provided.  Keyword arguments are identical to 
        arguments required by corresponding methods.  The argument 'std_cyc_name' can be
        optionally passed as a positional argument."""
        super().__init__()
        if std_cyc_name:
            self.set_standard_cycle(std_cyc_name)
        if cyc_dict:
            self.set_from_dict(cyc_dict)
        if cyc_file_path:
            self.set_from_file(cyc_file_path)
        
    def get_numba_cyc(self):
        """Returns numba jitclass version of Cycle object."""
        numba_cyc = TypedCycle(len(self.cycSecs))
        for key in self.__dict__.keys():
            numba_cyc.__setattr__(key, self.__getattribute__(key).astype(np.float64))
        return numba_cyc

    def set_standard_cycle(self, std_cyc_name):
        """Load time trace of speed, grade, and road type in a pandas dataframe.
        Argument:
        ---------
        std_cyc_name: cycle name string (e.g. 'udds', 'us06', 'hwfet')"""
        csv_path = os.path.join(CYCLES_DIR, std_cyc_name + '.csv')
        cyc = pd.read_csv(csv_path)
        for column in cyc.columns:
            self.__setattr__(column, cyc[column].to_numpy())
        self.set_dependents()

    def set_from_file(self, cyc_file_path):
        """Load time trace of speed, grade, and road type from 
        user-provided csv file in a pandas dataframe.
        Argument:
        ---------
        cyc_file_path: path to file containing cycle data"""
        cyc = pd.read_csv(cyc_file_path)
        for column in cyc.columns:
            self.__setattr__(column, cyc[column].to_numpy())
        self.set_dependents()

    def set_from_dict(self, cyc_dict):
        """Set cycle attributes from dict with keys 'cycGrade', 'cycMps', 'cycSecs', 'cycRoadType'
        and numpy arrays of equal length for values.
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
        self.cycMph = self.cycMps * gl.mphPerMps
        self.secs = np.insert(np.diff(self.cycSecs), 0, 0)
    
    def get_cyc_dict(self):
        """Returns cycle as dict rather than class instance."""
        keys = ['cycSecs', 'cycMps', 'cycGrade', 'cycRoadType']
        
        cyc = {}
        for key in keys:
            cyc[key] = self.__getattribute__(key)
        
        return cyc

# type specifications for attributes of Cycle class
cyc_spec = [('cycSecs', float64[:]),
            ('cycMps', float64[:]),
            ('cycGrade', float64[:]),
            ('cycRoadType', float64[:]),
            ('cycMph', float64[:]),
            ('secs', float64[:])
]

@jitclass(cyc_spec)
class TypedCycle(object):
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


def to_microtrips(cycle, stop_speed_m__s=1e-6):
    """
    Split a cycle into an array of microtrips with one microtrip being a start
    to subsequent stop plus any idle (stopped time).

    Arguments:
    ----------
    cycle: drive cycle converted to dictionary by cycle.get_cyc_dict()
    stop_speed_m__s: speed at which vehicle is considered stopped for 
        trip separation
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
    Create a cycle from times, speeds, and grades. If grades is not specified,
    it is set to zero.
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
        return False
    for k in c1.keys():
        if len(c1[k]) != len(c2[k]):
            return False
        if np.any(np.abs(np.array(c1[k]) - np.array(c2[k])) > 1e-6):
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
        # else:
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

# resample(), probably pt2d.make_course(), and clip_by_times()
# this one needs a test/demo
