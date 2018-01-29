from block import Block
import numpy as np

class reward_accumulator_block(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, gamma, name=None):

        self.gamma = gamma
        self.last = None
        super(reward_accumulator_block, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):

        if self.last:
            self.last_output = 0
        else:
            if isinstance(inputs, np.ndarray):
                inputs = inputs[0]
            self.last_output = inputs + self.gamma*self.last_output


    def reset(self):
        self.last_output = None
        self.last = True

