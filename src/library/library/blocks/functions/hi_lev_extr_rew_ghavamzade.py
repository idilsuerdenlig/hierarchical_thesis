import numpy as np

def G_high(inputs):

    if inputs == 0:
        res = 100.0
    else:
        res = 0.0
    return np.array([res])
