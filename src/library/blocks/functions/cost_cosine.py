import numpy as np


def cost_cosine(ins):
    del_theta = ins[0]
    states = ins[1]
    set_point = ins[2]
    if np.linalg.norm(states[:2]-set_point) < 2:
        reward = np.array([1])
    else:
        reward = np.cos(del_theta)

    return reward
