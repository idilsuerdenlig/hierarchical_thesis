from block import Block
import numpy as np

class fBlock(Block):

    def __init__(self, phi, name=None):

        self.phi = phi
        super(fBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):

        self.last_output = self.phi(inputs)
        self.alarm_output = None


    def reset(self, inputs):
        self.last_output = self.phi(inputs)
        self.alarm_output = None

    def init(self):
        self.last_output = None
        self.alarm_output = None
