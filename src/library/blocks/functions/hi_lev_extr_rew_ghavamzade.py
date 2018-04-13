import numpy as np


def G_high(inputs):
    reward = inputs[0]
    if reward is not None and reward[0] == 0:
        res = 100.0
    else:
        res = 0.0
    return np.array([res])
