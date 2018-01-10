import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


def pick_state(inputs):

    indices = [0,1]
    states_needed = np.empty(shape=(len(indices),))
    pos = 0
    for i in indices:
        states_needed[pos]=inputs[i]
        pos += 1
    return states_needed



