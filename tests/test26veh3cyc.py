"""Test script that times runs 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles."""

from math import isclose
import time
import numpy as np
import sys
import importlib
import pandas as pd

sys.path.append('../src')

# local modules
import FASTSim
importlib.reload(FASTSim)

t0 = time.time()

cycles = ['udds', 'hwfet', 'us06']
vehicles = np.arange(1, 27)

iter = 0
for vehno in vehicles:
    print('vehno =', vehno)
    for cycname in cycles:
        cyc = FASTSim.get_standard_cycle(cycname)
        veh = FASTSim.get_veh(vehno)

        output, diagno = FASTSim.sim_drive(cyc, veh)

        if iter > 0:
            dict_diag['vnum'].append(vehno)
            dict_diag['cycle'].append(cycname)
            for key in diagno.keys():
                dict_diag[key].append(diagno[key])

        else:
            dict_diag = {}
            dict_diag['vnum'] = [vehno]
            dict_diag['cycle'] = [cycname]
            for key in diagno.keys():
                dict_diag[key] = [diagno[key]]
            iter += 1

df = pd.DataFrame.from_dict(dict_diag)

t1 = time.time()
print()
print('Elapsed time: ', round(t1 - t0, 2), 's')

df0 = pd.read_csv('../docs/master_benchmark_vars.csv')

# make sure both dataframes have the same columns
new_cols = {col for col in df.columns} - {col for col in df0.columns}
df.drop(columns=new_cols, inplace=True)
old_cols = {col for col in df0.columns} - {col for col in df.columns}
df0.drop(columns=old_cols, inplace=True)


df_err = df.copy()
abs_err = []
for idx in df.index:
    for col in df.columns[2:]:
        if not(isclose(df.loc[idx, col], df0.loc[idx, col], rel_tol=1e-6, abs_tol=1e-9)):
            df_err.loc[idx, col] = (
                df.loc[idx, col] - df0.loc[idx, col]) / df0.loc[idx, col]
            abs_err.append(np.abs(df_err.loc[idx, col]))
            print(str(round(df_err.loc[idx, col] * 100, 5)) + '% for')
            print('New Value: ' + str(round(df.loc[idx, col], 15)))
            print('vnum = ' + str(df.loc[idx, 'vnum']))
            print('cycle = ' + str(df.loc[idx, 'cycle']))
            print('idx =', idx, ', col =', col)
            print()
        else:
            df_err.loc[idx, col] = 0

if len (abs_err) > 0:
    abs_err = np.array(abs_err)
    print('\nmax error =', str(round(abs_err.max() * 100, 4)) + '%')
else:
    print('There is no error.')
