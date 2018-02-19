import numpy as np


def lqr_cost(ins):
    q_square = 16
    r_square = 0

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

    normalization_coefficient = 144

    lqr_cost = q_square * error_cost + r_square * normalization_coefficient * action_cost
    return lqr_cost
