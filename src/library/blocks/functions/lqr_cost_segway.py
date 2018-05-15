import numpy as np


def lqr_cost_segway(ins):

    x = np.concatenate(ins)
    Q = np.diag([1.0, 0.0, 0.0])
    J = x.dot(Q).dot(x)
    reward = -J

    return np.array([reward])
