import numpy as np
from mushroom.utils.angles_utils import normalize_angle, shortest_angular_distance


def direction_to_angle(inputs):

    active_direction = inputs[0]
    theta = normalize_angle(inputs[1][2])


    ref_theta = active_direction*(np.pi/14)
    del_theta = shortest_angular_distance(from_angle=theta, to_angle=ref_theta)

    return np.array([del_theta])

