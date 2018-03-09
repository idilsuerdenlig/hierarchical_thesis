from block import Block
import numpy as np

class hold_state(Block):


    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):

        if len(self.last_output) is not 2:
            self.last_output = np.zeros(2)
        if np.any(alarms):
            states = np.concatenate(inputs)
            states = np.array([states[0], states[1]])
            self.last_output = states


    def reset(self, inputs):
        self.last_output = np.zeros(2)

    def init(self):
        self.last_output = None

