from block import Block
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
        self.wake_counter = 0
        self._reward = None
        self._absorbing = False
        self._last = False
        self._action = None

        super(MBlock, self).__init__(wake_time=1)

    def __call__(self, inputs, reward, absorbing, last, learn_flag):

        self.clock_counter += 1
        if self.wake_time == self.clock_counter:
            self.wake_counter += 1
            self._action = np.concatenate(inputs, axis=0)
            self.last_output, self._reward, self._absorbing, _ = self.environment.step(self._action)
            if self._render:
                self.environment.render()
            self._last = self.wake_counter >= self.environment.info.horizon or self._absorbing
            self.clock_counter = 0
        return self._absorbing, self._last

    def get_reward(self):
        return self._reward

    def reset(self,inputs):
        self.last_output = self.environment.reset()
        self._reward = None
        self._absorbing = False
        self.clock_counter = 0
        self.wake_counter = 0




