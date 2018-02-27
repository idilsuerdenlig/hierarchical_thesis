import numpy as np

def G_high(inputs):

    if inputs == 0:
        res = 100
    else:
        res = 0
    return np.array([res])
