import numpy as np

from fastsim import cycle, vehicle, simdrivehot

def get_error_val(model, test, time_steps, der_weight=0.1):
    """Returns time-averaged error for model and test signal.
    Arguments:
    ----------
    model: array of values for signal from model
    model_time_steps: array (or scalar for constant) of values for model time steps [s]
    test: array of values for signal from test
    der_weight: weight of derivative error term
    
    Output: 
    -------
    err: integral of absolute value of difference between model and 
    test per time"""

    err = np.trapz(
        y=abs(model - test),
        x=time_steps) / (time_steps[-1] - time_steps[0]) + \
        der_weight * np.trapz(
        y=abs(np.diff(model) - np.diff(test)) / np.diff(time_steps),
        x=time_steps[1:]) / (time_steps[-1] - time_steps[0])

    return err
