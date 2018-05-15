import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


def angle_to_angle_diff_complete_state(inputs):
    alpha_ref = inputs[0]
    states = inputs[1]
    alpha = states[0]
    alpha_dot = states[1]
    beta_dot = states[2]
    delta_alpha = alpha_ref[0] - alpha
    return np.array([delta_alpha, alpha_dot, beta_dot])



