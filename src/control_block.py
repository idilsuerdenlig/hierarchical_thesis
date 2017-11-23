from block import Block
from J_block import JBlock
from M_block import MBlock
from mushroom.algorithms.agent import *
import numpy as np
class ControlBlock(Block):
    """
    This class implements the functions to initialize and move the agent drawing
    actions from its policy.

    """
    def __init__(self, wake_time, agent, horizon=None, n_eps_per_fit=None, n_steps_per_fit=None):


        self.agent = agent
        self.step_counter = 0
        self.curr_step_counter = 0
        self.curr_episode_counter = 0
        self.n_eps_per_fit = n_eps_per_fit
        self.n_steps_per_fit = n_steps_per_fit
        self.dataset = list()
        if horizon is None:
            self.horizon = self.agent.horizon
        else:
            self.horizon = horizon
        self.last_input = None
        self.last_output = None
        self.last = True

        super(ControlBlock, self).__init__(wake_time=wake_time)


    def __call__(self, inputs, reward, absorbing, learn_flag):


        if absorbing or self.step_counter == self.horizon :
            print 'self.step_counter == horizon'
            self.agent.episode_start()
            state = np.concatenate(inputs, axis = 0)
            sample = self.last_input, self.last_output, reward, state, absorbing, True
            self.dataset.append(sample)
            if absorbing:
                self.last_output = None
                self.last_input = state
                self.clock_counter = self.wake_time-1
            self.last = True
            self.step_counter = 0
            self.curr_step_counter += 1
            self.curr_episode_counter += 1

        elif self.wake_time == self.clock_counter:
            state = np.concatenate(inputs, axis=0)
            if self.last_output is not None:
                sample = self.last_input, self.last_output, reward, state, False, False
                self.dataset.append(sample)
                self.curr_step_counter += 1
                self.step_counter += 1
            self.last_input = state
            self.last_reward = reward
            self.last = False

        if learn_flag and \
            (self.curr_step_counter == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            self.fit(self.dataset)
            self.dataset = list()

        if self.wake_time == self.clock_counter:
            action = self.agent.draw_action(state=state)
            self.last_output = action
            self.clock_counter = 0

        self.clock_counter += 1
        return absorbing

    def fit(self, dataset):
        print dataset
        self.agent.fit(dataset)
        self.curr_episode_counter = 0
        self.curr_step_counter = 0


    def add_reward(self, reward_block):

        assert isinstance(reward_block, JBlock) or isinstance(reward_block, MBlock)
        self.reward_connection = reward_block
