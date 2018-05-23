import numpy as np
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle


class GateToPass():

    def __init__(self, n_gates):
        self.n_gates = int(n_gates)

    def __call__(self, inputs):
        states = np.concatenate(inputs)
        #print(states)
        n_gates_passed = int(states[-1])
        #print(n_gates_passed)
        out = np.zeros(shape=(self.n_gates,))
        if n_gates_passed != self.n_gates:
            out[n_gates_passed] = 1

        return out


