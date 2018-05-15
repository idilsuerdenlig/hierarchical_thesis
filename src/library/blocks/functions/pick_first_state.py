import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


def pick_first_state(inputs):
    states = np.concatenate(inputs)
    indices = [0]
    states_needed = np.zeros(len(indices))
    pos = 0
    for i in indices:
        states_needed[pos]=states[i]
        pos += 1

    return states_needed



