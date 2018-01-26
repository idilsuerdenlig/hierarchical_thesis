from block import Block
import numpy as np

class reward_accumulator_block(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, gamma, name=None):

        self.gamma = gamma
        super(reward_accumulator_block, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):
        if isinstance(inputs, np.ndarray):
            inputs = inputs[0]
        self.last_output = inputs[0] + self.gamma*self.last_output

        return absorbing, last

    def reset(self, inputs):
        self.last_output = 0

