from block import Block
import numpy as np

class fBlock(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, phi, name=None):

        self.phi = phi
        super(fBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):

        #print self.name, 'STEP', self.clock_counter, absorbing
        self.last_output = self.phi(inputs)
        self.alarm_output = self.last_output


    def reset(self, inputs):
        self.last_output = self.phi(inputs)
        self.alarm_output = self.last_output

    def init(self):
        self.last_output = None
        self.alarm_output = None
