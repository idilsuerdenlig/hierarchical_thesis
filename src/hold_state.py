import numpy as np


def hold_state(inputs):

    states = np.concatenate(inputs)
    states = np.array([states[0], states[1]])
    print 'HOLD STATE OUTPUT:   ', states
    return states



