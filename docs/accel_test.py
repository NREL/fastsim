import sys
if '../src' not in sys.path:
    sys.path.append('../src')
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import importlib
# import seaborn as sns
# sns.set(font_scale=2, style='whitegrid')

# local modules
import SimDrive
importlib.reload(SimDrive)

def create_accel_cyc(length_in_seconds=300, spd_mph=89.48, grade=0.0, hz=10):
    """
    Create a synthetic Drive Cycle for acceleration targeting.
    Defaults to a 15 second acceleration cycle. Should be adjusted based on target acceleration time
    and initial vehicle acceleration time, so that time isn't wasted on cycles that are needlessly long.

    spd_mph @ 89.48 FASTSim XL version mph default speed for acceleration cycles
    grade @ 0 and hz @ 10 also matches XL version settings
    """
    mphPerMps = 2.23694
    cycMps = [(1/mphPerMps)*float(spd_mph)]*(length_in_seconds*hz)
    cycMps[0] = 0.
    cycMps = np.asarray(cycMps)
    cycSecs = np.arange(0, length_in_seconds, 1./hz)
    cycGrade = np.asarray([float(grade)]*(length_in_seconds*hz))
    cycRoadType = np.zeros(length_in_seconds*hz)
    cyc = {'cycMps': cycMps, 'cycSecs': cycSecs, 'cycGrade': cycGrade, 'cycRoadType':cycRoadType}
    return cyc

