import numpy as np


def get_error_val(model, test, time_steps, normalize=True):
    """Returns time-averaged error for model and test signal.
    Arguments:
    ----------
    model: array of values for signal from model
    test: array of values for signal from test data
    time_steps: array (or scalar for constant) of values for model time steps [s]
    test: array of values for signal from test
    normalize: Boolean, if True, normalizes the data such that all values are >= 0
    
    Output: 
    -------
    err: integral of absolute value of difference between model and 
    test per time"""

    if normalize:
        max_for_norm = max(max(model), max(test))
        min_for_norm = min(min(model), min(test))
        model_norm = (model - min_for_norm) / (max_for_norm - min_for_norm)
        test_norm = (test - min_for_norm) / (max_for_norm - min_for_norm)
        err = np.trapz(y=abs(model_norm - test_norm), x=time_steps) / (time_steps[-1] - time_steps[0])
    
    else:
        err = np.trapz(y=abs(model - test), x=time_steps) / (time_steps[-1] - time_steps[0])

    return err
