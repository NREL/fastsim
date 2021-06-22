import time
import os
import pandas as pd
from collections import ChainMap
import re
import numpy as np
from scipy.integrate import cumtrapz

# load the vehicle test data
def load_test_data(use_cs=True, use_hs=False):
    """Load Fusion test data.
    Keyword Arguments:
    ------------------
    use_cs=True loads cold start data
    use_hs=True loads hot start data"""

    print('Loading test data.')
    test_data_path = 'C:\\Users\\cbaker2\\Documents\\DynoTestData\\FordFusionTestData\\'
    t0 = time.time()
    datadirs = [test_data_path + 'udds_tests\\',
                test_data_path + 'us06_tests\\']
    tests = [[test for test in os.listdir(f) if 
        'csv' in test] for f in datadirs]

    dflist = [{j: pd.read_csv(i+j)
            for j in tests[datadirs.index(i)]} for i in datadirs]
    print('Elapsed time to read data files: {:.3e} s'.format(time.time() - t0))
    dfdict0 = dict(ChainMap(*reversed(dflist)))
    dfdict1 = {}

    for key in dfdict0.keys():
        repkey = key[9:].replace('.xlsx', '').replace(' ', '').replace('_', ' ')
        #     cyclekey = re.match('\S+(?=x\d)', repkey)[0] # grab cycle name without repetition suffix
        # grab cycle name with repetition suffix
        cyclekey = re.match('\S+', repkey)[0]
        tempkey = re.search('\d{1,2}F', repkey)[0]
        startkey = re.search('(?<=_)(c|h)s', key)[0]
        dfdict1[cyclekey + ' ' + tempkey + ' ' + startkey] = dfdict0[key]
    del dfdict0
    times = np.arange(0, 1500, 250)

    # dict for results dataframes
    resdfdict = {}
    dfdict = {}

    cs_keys = [key for key in dfdict1.keys() if re.search('\scs$', key)]
    hs_keys = [key for key in dfdict1.keys() if re.search('\shs$', key)]

    if use_cs and use_hs:
        keys = cs_keys + hs_keys
    elif use_cs:
        keys = cs_keys
    elif use_hs:
        keys = hs_keys
    else:
        print('Passed invalid option.  At least one of `use_cs` or `use_hs` must be True.')

    for k in keys:
        # print(k)

        # Remove rows with wacky time
        steps = dfdict1[k]['DAQ_Time[s]'].diff()
        timejumps = np.where(steps < 0)[0]
        timejump = timejumps[-1]
        dfdict[k] = dfdict1[k].drop(np.arange(timejump), axis=0)
        dfdict[k].index -= dfdict[k].index.min()

        # Remove extra columns
        dfdict[k].insert(int(np.where(dfdict[k].columns == 'Dyno_Spd[mph]')[
                        0][0]), 'Dyno_Spd[mps]', dfdict[k]['Dyno_Spd[mph]'] * 0.44704)

        dfdict[k]['Tractive_Power[W]'] = dfdict[k]['Dyno_TractiveForce[N]'] * \
            dfdict[k]['Dyno_Spd[mps]']
        dfdict[k]['PosTractive_Power[W]'] = dfdict[k]['Tractive_Power[W]']
        # negindices = np.where(dfdict[k]['Tractive_Power[W]'] < 0)[0]
        negindices = dfdict[k]['Tractive_Power[W]'] < 0
        dfdict[k].loc[negindices, 'PosTractive_Power[W]'] = 0

        dfdict[k]['Kinetic_Power_Density[W/kg]'] = dfdict[k]['Dyno_Spd[mps]'] * dfdict[k]['Dyno_Spd[mps]'].diff() \
            / dfdict[k]['DAQ_Time[s]'].diff()
        dfdict[k]['Kinetic_Power_Density[W/kg]'].fillna(0, inplace=True)

        dfdict[k]['PosKinetic_Power_Density[W/kg]'] = dfdict[k]['Kinetic_Power_Density[W/kg]']
        dfdict[k].loc[dfdict[k]['Kinetic_Power_Density[W/kg]']
                    <= 0, 'PosKinetic_Power_Density[W/kg]'] = 0

        dfdict[k]['PosTractive_Energy[J]'] = np.concatenate((np.array([0]), cumtrapz(
            dfdict[k]['PosTractive_Power[W]'], x=dfdict[k]['DAQ_Time[s]'])), axis=0)
        dfdict[k]['Distance[m]'] = np.concatenate((np.array([0]), cumtrapz(
            dfdict[k]['Dyno_Spd[mps]'], x=dfdict[k]['DAQ_Time[s]'])), axis=0)

        dfdict[k]['PosKinetic_Energy_Density[J/kg]'] = np.concatenate((np.array([0]),
                                                                    cumtrapz(dfdict[k]['PosKinetic_Power_Density[W/kg]'].values,
                                                                                x=dfdict[k]['DAQ_Time[s]'].values)), axis=0)


        dfdict[k].index.set_names('time_step', inplace=True)

        # bring temperature into data frame from filenames
        dfdict[k]['Temp_Amb[C]'] = round(
            (int(re.search('\d{1,2}(?=F)', k)[0]) - 32) * 5 / 9, 1)

        # compute deltaT at times
        indices = [np.where(dfdict[k]['DAQ_Time[s]'] >= t)[0][0] for t in times]

        # resdfdict[k].drop(axis=0, index=0, inplace=True) # probably want this initial row
        dfdict[k].set_index([dfdict[k].index, 'Temp_Amb[C]'], inplace=True)

    keys = list(dfdict.keys())
    del dfdict1

    df = pd.concat(dfdict, sort=False)
    df.index.set_names('filename', level=0, inplace=True)
    df = df.reorder_levels(['filename', 'Temp_Amb[C]', 'time_step'])
    cols0 = list(df.columns.values)
    # reorder columns in terms of independent and dependent variables
    cols = [cols0[0]] + cols0[-3:] + cols0[2:-3]
    df = df[cols]
    df['Powertrain Efficiency'] = df['Dyno_Power_Calc[kW]'] / df['Fuel_Power_Calc[kW]']
    df.drop(columns=[col for col in df.columns if re.search(
        'Unnamed', col)], inplace=True)
    
    return df
