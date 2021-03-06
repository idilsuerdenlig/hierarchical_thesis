import numpy as np
from mushroom.utils.angles import shortest_angular_distance, normalize_angle


def pos_ref_angle_difference(ins):
    x_ref = ins[0][0]
    y_ref = ins[0][1]
    x = ins[1][0]
    y = ins[1][1]
    theta = ins[1][2]
    del_x = x_ref-x
    del_y = y_ref-y
    theta_ref = normalize_angle(np.arctan2(del_y, del_x))
    del_theta = shortest_angular_distance(from_angle=theta, to_angle=theta_ref)

    goal_pos = np.array([x_ref, y_ref])
    pos = np.array([x, y])
    del_pos = np.linalg.norm(pos - goal_pos)

    return np.array([del_theta, del_pos])


def angle_ref_angle_difference(ins):
    theta_ref = normalize_angle(ins[0])
    theta = ins[2][2]
    pos_ref = ins[1]
    del_theta = shortest_angular_distance(from_angle=theta, to_angle=theta_ref)
    x = ins[2][0]
    y = ins[2][1]
    pos = np.array([x, y])
    del_pos = np.linalg.norm(pos - pos_ref)

    return np.array([del_theta, del_pos])