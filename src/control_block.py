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
    def __init__(self, wake_time, agent, n_eps_per_fit=None, n_steps_per_fit=None, callbacks=list()):

        self.agent = agent
        self.step_counter = 0
        self.curr_step_counter = 0
        self.curr_episode_counter = 0
        self.n_eps_per_fit = n_eps_per_fit
        self.n_steps_per_fit = n_steps_per_fit
        self.dataset = list()
        self.horizon = self.agent.mdp_info.horizon
        self.last_input = None
        self.last_output = None
        self.callbacks = callbacks

        super(ControlBlock, self).__init__(wake_time=wake_time)

    def __call__(self, inputs, reward, absorbing, learn_flag):
        self.clock_counter += 1
        last = False

        if absorbing:
            self.curr_episode_counter += 1
            self.step_counter = 0

        if absorbing or self.wake_time == self.clock_counter:
            self.curr_step_counter += 1
            self.step_counter += 1
            last = absorbing or self.step_counter == self.horizon
            if inputs.ndim == 1:
                state = inputs
            else:
                state = np.concatenate(inputs, axis=0)
            sample = self.last_input, self.last_output, reward, state, absorbing, last
            self.dataset.append(sample)

        if learn_flag and \
            (self.curr_step_counter == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            self.fit(self.dataset)
            self.dataset = list()

        if last and not absorbing:
            self.agent.episode_start()
            self.step_counter = 0

        if self.wake_time == self.clock_counter and not absorbing:
            action = self.agent.draw_action(state)
            self.last_output = action
            self.last_input = state
            self.last_reward = reward
            self.clock_counter = 0

        return absorbing

    def fit(self, dataset):
        self.agent.fit(dataset)
        self.curr_episode_counter = 0
        self.curr_step_counter = 0
        for c in self.callbacks:
            callback_pars = dict(dataset=dataset)
            c(**callback_pars)

    def add_reward(self, reward_block):
        assert isinstance(reward_block, JBlock) or isinstance(reward_block, MBlock)
        self.reward_connection = reward_block

    def reset(self, inputs):
        if inputs.ndim == 1:
            state = inputs
        else:
            state = np.concatenate(inputs,axis=0)
        self.agent.episode_start()
        self.clock_counter = 0
        self.step_counter = 0
        action = self.agent.draw_action(state=state)
        self.last_output = action
        self.last_input = state


