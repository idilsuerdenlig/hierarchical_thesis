import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


def phi(ins):
    x_ref = ins[0][0]
    y_ref = ins[0][1]
    x = ins[1][0]
    y = ins[1][1]
    theta = ins [1][2]
    del_x = x_ref-x
    del_y = y_ref-y
    theta_ref = normalize_angle(np.arctan2(del_y, del_x))
    theta = np.pi/2-theta
    del_theta = shortest_angular_distance(theta_ref,theta)
    return np.array([del_theta])
