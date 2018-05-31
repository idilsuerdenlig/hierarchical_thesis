import numpy as np


def hi_lev_state(ins):

    x = np.concatenate(ins)
    out = np.zeros(4)

    for i in [4, 5, 6, 7]:
        if x[i] > 0:
            out[i-4] = 1


    return out
