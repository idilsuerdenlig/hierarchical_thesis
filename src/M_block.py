from block import Block
import numpy as np
from mushroom.environments import *

class MBlock(Block):
    """
    This class implements the Model block of a computational graph for hierarchical learning.

    """
    def __init__(self, env, render=False):
        """
        Constructor.

        Args:
            input_connections ([]block): the list of blocks that inputs the object block;
        """
        self.environment = env
        self._render = render
        self._reward = None
        self._last = False
        self._state = None

        super(MBlock, self).__init__(wake_time=1)

    def __call__(self, inputs, reward, absorbing, learn_flag):

        self.clock_counter+=1

        if inputs == None or len(inputs) == 0:
            self.last_output = self.environment.reset()
            absorbing = False

        else:
            self._state = np.concatenate(inputs, axis=0)
            self.last_output, self._reward, absorbing, _ = self.environment.step(self._state)

            if self._render:
                self.environment.render()

        self._last = not(self.clock_counter < self.environment.info.horizon and not absorbing)

        return absorbing, self._last

    def get_reward(self):
        return self._reward

