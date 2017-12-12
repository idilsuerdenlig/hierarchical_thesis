import numpy as np


def phi(ins):
    x_ref = ins[0][0]
    y_ref = ins[0][1]
    x = ins[1][0]
    y = ins[1][1]
    theta = ins [1][2]
    del_x = x_ref-x
    del_y = y_ref-y
    theta_ref = np.arctan2(del_y, del_x)
    theta_ref = (theta_ref + np.pi) % (2 * np.pi) - np.pi
    theta = np.pi/2-theta
    del_theta = theta_ref-theta
    del_theta = (del_theta + np.pi) % (2 * np.pi) - np.pi
    return np.array([del_theta])
