import numpy as np


def cost_cosine(ins):
    del_theta = ins[0]
    reward = np.cos(del_theta)-np.cos(np.pi/4)

    return reward
