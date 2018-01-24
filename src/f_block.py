from block import Block
import numpy as np

class fBlock(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, phi, name=None):

        self.phi = phi
        super(fBlock, self).__init__(name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):

        if np.any(alarms) or absorbing:
        #    if self.name == 'f3':
        #        print self.name, 'STEP', self.clock_counter, absorbing
            self.last_output = self.phi(inputs)
            self.alarm_output = self.last_output
        #    self.clock_counter = 0

        return absorbing, last

    def reset(self, inputs):
        #if self.name == 'f3':
        #    print self.name, 'RESET'
        #self.clock_counter = 0
        self.last_output = self.phi(inputs)
        self.alarm_output = self.last_output


