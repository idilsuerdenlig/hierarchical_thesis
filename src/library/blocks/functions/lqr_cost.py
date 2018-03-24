import numpy as np


def lqr_cost(ins):
    q_square = 8
    r_square = 1

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
        if r is not None:
            action_cost += -r.dot(r)


    lqr_cost = r_square * action_cost + q_square * error_cost
    return lqr_cost
