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
        self.reward = None
        self.last = False
        self._state = None

        super(MBlock, self).__init__(wake_time=1)

    def __call__(self, inputs, reward, absorbing, learn_flag):

        if inputs == None or inputs == list():
            self._state = self.environment.reset()
            absorbing = False
            self.last_output = self._state

        else:
            self._state = inputs[0]
            print(self._state)

            self.last_output, self.reward, absorbing, _ = self.environment.step(self._state)

            if self._render:
                self.environment.render()

        self.last = not(self.clock_counter < self.environment.horizon and not absorbing)


        return absorbing

    def get_reward(self):
        return self.reward

