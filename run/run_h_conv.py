
import dedalus.public as d3
import numpy as np
from dedalus.tools import post
import file_tools as flt
import numpy as np
import pandas as pd
import logging
import sys
sys.path.append("../src/")
import os
root = logging.root
for h in root.handlers: h.setLevel("INFO") 
logger = logging.getLogger(__name__)
import horizontal_convection


series = sys.argv[1]
index = int(sys.argv[2])
save_dir = f'data/{series}'

def create_dataframe(param_dic):
    """Convert dictionary of experiment parameters into multiindex of params for each experiment.
    Parameters paired in a tuple will be paired in the multiindex.
    E.g. {'A':[1,2], ('B','C'):([3,4],[-3,-4]),'D':[0]} ->
    A   B   C   D
    1   3  -3   0
    1   4  -4   0
    2   3  -3   0
    2   4  -4   0
    """
    tuples = []
    param_lists = {}
    for key in param_dic:
        if isinstance(key, str):
            param_lists[key] = param_dic[key]
        elif isinstance(key, tuple):
            tuples.append(key)
            param_lists[key[0]] = list(range(len(param_dic[key][0])))
            for keyi in key[1:]:
                param_lists[keyi] = [pd.NA]

    params = pd.MultiIndex.from_product(param_lists.values(), names=param_lists.keys())
    params = pd.DataFrame(index=params).reset_index()

    for tup in tuples:
        for column in tup[1:]:
            params[column] = params[tup[0]]
        for ind, column in enumerate(tup):
            params[column] = params[column].apply(lambda j: param_dic[tup][ind][j])

    return params

import glob

param_list = {
    'Lx': [6],
    'Lz': [1],
    'Tm': [.6],
    'z0': [.8],
    'Ra' : [1e5],#[1e4, 1e5, 1e6, 1e7],
    'Pr' : [1.],
    'S' : [1.],
    'ε' : [2e-2], # Need to explore/read about these. 
    'γ' : [2e-2],
    'δ' : [1e-2],
    'β' : [1.51044385],
    'm' : [0.],
    'n' : [0.],
    'a' : [0.],
    'b' : [0.],
    'timestepper':['SBDF2'],
    'timestep': [1e-4],
    'stop_sim_time':[0.4],
    'snap_time':[1],
    'avg_time': [100],
    'print_step':[50],
    'max_writes':[10000],
    'nx':[128],
    'nz':[128],
    'dealias':[1.5],
    'save_dir': [save_dir],
    'script':[0],
    'adv': [0],
    'restart': [0],
    'chkp_time': [1]
}

params = create_dataframe(param_list)
params['sim_name'] = ['-'.join([series,f'{i:0>3d}']) for i in params.index]
# series_restart = 'ch-3D-comparison-1'
# params['restart_file'] = [last_save_file(f'{series_restart}-{i:0>3d}') for i in range(len(params))]

params.to_csv(f'./parameters/parameters-{series}.csv')

# import europa

horizontal_convection.run_europa_sim(params.loc[index])
