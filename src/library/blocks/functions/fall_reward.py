import numpy as np


def fall_reward(inputs):
    state = inputs[0]
    if abs(state[1]) > np.pi / 2:
        res = -10000.0
    else:
        res = 0.0
    return np.array([res])
