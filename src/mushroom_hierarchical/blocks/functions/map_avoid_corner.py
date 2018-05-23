import numpy as np


def map(ins):
    state = ins[0]
    x = state[0]
    y = state[1]

    A = ((x-75)/75)**2
    B = ((y-75)/75)**2


    return np.array([A, B])
