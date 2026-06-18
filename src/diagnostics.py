import dedalus.public as d3
import numpy as np 

def array_diff_1D(*args):
    result = args[0]['g'] - args[1]
    return result

def array_mult(*args):
    result = args[0]['g'] * args[1]
    return result

def array_diff_2D(*args):
    result = args[0]['g']- args[1]
    return np.expand_dims(result, axis=1)

def ice_ocean_interface_extract(*args, vec=False):
    indx = np.argmin(abs(args[1]['g'] - 0.5), axis=1)

    result = args[0]['g'][:, np.arange(0, len(indx)), indx]

    return np.expand_dims(result, axis=2)



