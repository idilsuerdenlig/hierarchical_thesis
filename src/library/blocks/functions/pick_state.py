import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


def pick_state(inputs):

    states = np.concatenate(inputs)
    indices = [0,1]
    states_needed = np.zeros(shape=(len(indices),))
    pos = 0
    for i in indices:
        states_needed[pos]=states[i]
        pos += 1

    return states_needed



