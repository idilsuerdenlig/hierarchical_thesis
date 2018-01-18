from block import Block
import numpy as np

class fBlock(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, wake_time, phi, name=None):

        self.phi = phi
        super(fBlock, self).__init__(wake_time=wake_time, name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag):

        self.clock_counter += 1
        if self.wake_time == self.clock_counter or absorbing:
            self.last_output = self.phi(inputs)
            self.clock_counter = 0

        return absorbing, last

    def reset(self, inputs):
        self.clock_counter = 0
        self.last_output = self.phi(inputs)


