
import dedalus.public as d3
import numpy as np
from dedalus.tools import post
import file_tools as flt
import numpy as np
import pandas as pd
import logging
import sys
sys.path.append("../../../src/")
import os
root = logging.root
for h in root.handlers: h.setLevel("INFO") 
logger = logging.getLogger(__name__)
import horizontal_convection
from optimum_beta import optimum_beta_RaF#, get_eqlb_z
import glob


series = sys.argv[1]
index = int(sys.argv[2])
save_dir = f'data/{series}'

def sort_checkpoint_files(files):
    return sorted(files, key=lambda f: int(f.split("_s")[-1].split(".")[0]))

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
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


Ras = [1e8]#, 1e9]
dts = [1e-7]#, 1e-9]
stop_time = [40.0]

param_list = {
    'Lx': [10],
    'Lz': [1],
    ('Tm', 'z0'): ([0.2, 0.4],[0.8, 0.6]),
    # 'z0': [.8],
    ('Ra', 'timestep', 'stop_sim_time') : (Ras, dts, stop_time), 
    'Pr' : [1.0],
    'S' : [10.0],
    'ε' : [2e-3], # Need to explore/read about these. 
    'γ' : [2e-3],
    'δ' : [1e-2],
    'β' : [1.51044385],
    'm' : [0.],
    'n' : [0.],
    'a' : [0.],
    'b' : [0., 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0],
    'timestepper':['SBDF2'],
    # 'timestep': [1e-4],
    # 'stop_sim_time':[0.4, 0.4],
    'snap_iter_2D':[100],
    'snap_iter_0D':[10],
    'avg_time': [500],
    'print_step':[200],
    'max_writes':[10000],
    'nx':[1024],
    'nz':[1024],
    'dealias':[1.5],
    'save_dir': [save_dir],
    'script':[0],
    'adv': [0],
    'restart': [0],#"/home/hcfch1/scratch/HorizontalConvection/lcl/A5_Ra4/data/Ra4/000/chkp/chkp_s11.h5"],
    'chkp_time': [1000]
}

params = create_dataframe(param_list)


params = params.sort_values(by=["Tm", "b"]).reset_index(drop=True)

params['sim_name'] = ['-'.join([series,f'{i:0>3d}']) for i in params.index]
params['sim_index'] = [f'{i:0>3d}' for i in params.index]
params['sim_suite'] = [series for i in params.index]
params['save_dir'] = ["data/{:s}/{:s}".format(params.loc[i]['sim_suite'], params.loc[i]['sim_index']) for i in params.index]

# series_restart = 'ch-3D-comparison-1'
# params['restart_file'] = [last_save_file(f'{series_restart}-{i:0>3d}') for i in range(len(params))]
restart_root = "/home/hcfch1/scratch/HorizontalConvection/lcl/A10/Ra7/data/Ra7/"
for i in range(len(params)):
    Ra, θm, eps = (params['Ra'][i], params['Tm'][i], params['ε'][i])

    P = 0.001#params["stop_sim_time"][i]
    dt = params["timestep"][i]

    params.loc[i, 'β'] = optimum_beta_RaF(Ra, θm, eps)

    params.loc[i, 'snap_iter_2D'] = round(P / 10 / dt)

    params.loc[i, 'snap_iter_0D'] = round(P / 1000 / dt)

    params.loc[i, 'avg_time'] = round(P / 1 / dt)    # average every 10% of the end time 

    params.loc[i, 'chkp_time'] = round(P / 10 / dt)

    if isinstance(params.loc[i, 'restart'], str):
        params.loc[i, 'stop_sim_time'] += 10.0#params["stop_sim_time"][i] 

    try:
        restart_file = sort_checkpoint_files(glob.glob(restart_root + "{:03d}".format(11) + "/chkp/*.h5"))[-1]
        params.loc[i, 'restart'] = restart_file
    except:
        params.loc[i, 'restart'] = 0
    # print(i, restart_root + "{:03d}".format(i) + "/chkp/*.h5", sorted(glob.glob(restart_root + "{:03d}".format(i) + "/chkp/*.h5")))
    # print(i, restart_file)

params.to_csv(f'./parameters/parameters-{series}.csv')

sim_dir = params.loc[index]['save_dir']


# Do some work
# print(f"Rank {rank} reached the barrier")

# print(f"Rank {rank} passed the barrier")

if rank == 0 and not os.path.isdir(sim_dir):
    os.makedirs(sim_dir)

comm.Barrier()  # Wait here until all ranks arrive
horizontal_convection.run_horizontal_conv_sim(params.loc[index])
