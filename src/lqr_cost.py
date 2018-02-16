import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


def lqr_cost(ins):
    error_cost = np.zeros(1)
    action_cost = np.zeros(1)

    del_theta = ins[0]
    Q = np.array([del_theta])

    for q in Q:
        error_cost += -q.dot(q)

    if len(ins) == 1:
        action = np.zeros(1)
    else:
        action = ins[1]

    R = np.array([action])

    for r in R:
        action_cost += -r.dot(r)

    lqr_cost = 5*error_cost + action_cost
    return lqr_cost
