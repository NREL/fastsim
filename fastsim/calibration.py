import numpy as np

from fastsim import cycle, vehicle, simdrivehot

def get_error_val(model, test, time_steps):
    """Returns time-averaged error for model and test signal.
    Arguments:
    ----------
    model: array of values for signal from model
    model_time_steps: array (or scalar for constant) of values for model time steps [s]
    test: array of values for signal from test
    
    Output: 
    -------
    err: integral of absolute value of difference between model and 
    test per time"""

    err = np.trapz(
        y=abs(model - test),
        x=time_steps) / time_steps[-1]

    return err


def get_error_for_cycle(param_vals, param_names, test_dfs, veh_jit, teFcInitDegC):
    """Function for running a single cycle and returning the error.
    Arguments:
    ----------
    params : array of parameters values
    param_names : array of parameter names for which params are set
    test_dfs : list of pandas dataframes of test data
    veh_jit : vehicle.VehicleJit instance
    teFcInitDegC : engine initial temperature [°C]
    """

    errors = []
    for test_df in test_dfs:
        cycSecs = test_df['Time[s]'].values
        cycMps = test_df['Dyno_Spd[mps]'].values

        cyc = cycle.Cycle(cyc_dict={'cycSecs': cycSecs, 'cycMps': cycMps})
        cyc_jit = cyc.get_numba_cyc()

        sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit,
            teAmbDegC=test_df['Cell_Temp[C]'],
            teFcInitDegC=teFcInitDegC
        )

        # unpack input parameters
        for param_name, param_val in zip(param_names, param_vals):
            sim_drive.vehthrm.__setattr__(param_name, param_val)

        sim_drive.vehthrm.teTStatFODegC = sim_drive.vehthrm.teTStatSTODegC + \
            sim_drive.vehthrm.teTStatDeltaDegC
        sim_drive.sim_drive()

        # calculate error
        for i in range(len(error_vars)):
            model_err_var = sim_drive.__getattribute__(error_vars[i][0])
            test_err_var = dfdict[cyc_name][error_vars[i][1]].values

            err = cal.get_error_val(model_err_var, test_err_var,
                                    time_steps=cycSecs)

            errors.append(err)

    return tuple(errors)
