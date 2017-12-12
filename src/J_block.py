from block import Block
import numpy as np
from mushroom.algorithms.agent import Agent

class JBlock(Block):
    """
    This class implements the functions to calculate rewards.

    """
    def __init__(self, wake_time, reward_function):
        """
        Constructor.

        Args:
            policy (object): the policy to use for the agent;
            gamma (float): discount factor;
            params (dict): other parameters of the algorithm.
            n_iterations: number of iterations for the fit of the agent
        """
        self.reward_function = reward_function
        super(JBlock, self).__init__(wake_time=wake_time)

    def __call__(self, inputs, reward, absorbing, last, learn_flag):

        self.clock_counter += 1

        if self.wake_time == self.clock_counter or absorbing:
            self.clock_counter = 0
            if inputs.ndim != 1:
                self.last_output = self.reward_function(np.concatenate(inputs), absorbing)
            else:
                self.last_output = self.reward_function(inputs, absorbing)

            return absorbing, last

    def reset(self, inputs):
        self.clock_counter = 0
        if inputs.ndim != 1:
            self.last_output = self.reward_function(np.concatenate(inputs), False)
        else:
            self.last_output = self.reward_function(inputs, False)

    def get_reward(self):
        return self.last_output
