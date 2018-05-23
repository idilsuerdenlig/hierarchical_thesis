import numpy as np


def saturation(inputs):
    action = inputs[0]
    if action[0] > np.pi / 4:
        res = np.pi/4
    elif action[0] < -np.pi/4:
        res = -np.pi/4
    else:
        res = action[0]
    return np.array([res])
