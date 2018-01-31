import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


def G_low(inputs):

    plus = 0
    active_direction = inputs[0]
    if active_direction <= 4:
        goal_pos = np.array([14, 7.5])
    else:
        goal_pos = np.array([14, 14])

    pos = np.array([inputs[1][0], inputs[1][1]])

    if np.linalg.norm(pos-goal_pos) <= 1:
        plus = 100
    elif pos[0] > 15 or pos[0] < 0 or pos[1] > 15 or pos[1] < 0:
        plus = -100

    theta_ref = normalize_angle(np.arctan2(pos[1]-goal_pos[1], pos[0]-goal_pos[0]))
    theta = inputs[1][2]
    theta = normalize_angle(np.pi/2-theta)
    del_theta = shortest_angular_distance(from_angle=theta,to_angle=theta_ref)
    power = -del_theta**2/((np.pi/6)*(np.pi/6))
    res = np.expm1(power)+plus

    return np.array([res])
