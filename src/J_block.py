from block import Block
from mushroom.algorithms.agent import Agent

class JBlock(Block):
    """
    This class implements the functions to calculate rewards.

    """
    def __init__(self, wake_time):
        """
        Constructor.

        Args:
            policy (object): the policy to use for the agent;
            gamma (float): discount factor;
            params (dict): other parameters of the algorithm.
            n_iterations: number of iterations for the fit of the agent
        """

        super(JBlock, self).__init__(wake_time=wake_time)

    def __call__(self, inputs, reward, absorbing):

        self.clock_counter+=1

        if self.wake_time == self.clock_counter:
           self.clock_counter = 0
           self.last_output = self.reward_function(inputs)

        return self.last_output

    def reward_function(self):
        return reward

    def get_reward(self):
        return self.last_output
