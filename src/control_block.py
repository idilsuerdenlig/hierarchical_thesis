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
    def __init__(self, wake_time, agent, n_eps_per_fit, n_steps_per_fit, horizon):


        self.agent = agent
        self.step_counter = 0
        self.curr_step_counter = 0
        self.curr_episode_counter = 0
        self.n_eps_per_fit = n_eps_per_fit
        self.n_steps_per_fit = n_steps_per_fit
        self.dataset = list()
        self.horizon = horizon
        self.last_input = None
        self.last_reward = None
        self.last_output = None
        self.last_abs = False
        self.last_last = False

        super(ControlBlock, self).__init__(wake_time=wake_time)


    def __call__(self, inputs, reward, absorbing, learn_flag):

        self.clock_counter += 1

        if learn_flag and \
            (self.curr_step_counter == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            self.fit(self.dataset)
            print self.dataset
            self.dataset = list()

        if absorbing or self.step_counter == self.horizon :
            self.agent.episode_start()
            state = np.concatenate(inputs, axis = 0)
            sample = self.last_input, self.last_output, self.last_reward, state, self.last_abs
            self.dataset.append(sample)
            self.last_output = None
            self.last_reward = None
            self.last_input = None
            self.last_abs = False
            self.last_last = False
            self.clock_counter = 0
            self.step_counter = 0
            self.curr_step_counter = len(self.dataset)-1
            self.curr_episode_counter+=1

        elif self.wake_time == self.clock_counter:
             state = np.concatenate(inputs, axis=0)
             action = self.agent.draw_action(state=state)
             self.clock_counter=0
             if self.last_reward is not None:
                sample = self.last_input, self.last_output, self.last_reward, state, self.last_abs
                self.dataset.append(sample)
             self.last_output = action
             self.last_input = state
             self.last_reward = reward
             self.last_abs = absorbing
             self.curr_step_counter = len(self.dataset)
             self.step_counter += 1
             self.last_last = self.last_abs or self.step_counter == self.horizon

        return absorbing, self.last_last

    def fit(self, dataset):

        self.agent.fit(dataset)
        self.curr_episode_counter = 0
        self.curr_step_counter = 0


    def add_reward(self, reward_block):

        assert isinstance(reward_block, JBlock) or isinstance(reward_block, MBlock)
        self.reward_connection = reward_block
