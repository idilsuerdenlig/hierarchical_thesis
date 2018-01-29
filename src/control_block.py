from block import Block
import numpy as np
from dataset_manager import DatasetManager
class ControlBlock(Block):
    """
    This class implements the functions to initialize and move the agent drawing
    actions from its policy.

    """
    def __init__(self, name, agent, n_eps_per_fit=None, n_steps_per_fit=None, callbacks=list()):

        self.agent = agent
        self.ep_step_counter = 0
        self.curr_step_counter = 0
        self.curr_episode_counter = 0
        self.n_eps_per_fit = n_eps_per_fit
        self.n_steps_per_fit = n_steps_per_fit
        self.horizon = self.agent.mdp_info.horizon
        self.gamma = self.agent.mdp_info.gamma
        self.last_input = None
        self.last_output = None
        self.last = False
        self.callbacks = callbacks
        self.dataset = DatasetManager()

        super(ControlBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):

        if isinstance(inputs[0], np.float64):
            state = inputs
        else:
            state = np.concatenate(inputs, axis=0)

        if last:
            print 'model last', self.name
            sample = state, None, reward, absorbing, True
            self.dataset.add_sample(sample, False)

        elif self.last:
            print 'start eps', self.name
            self.curr_episode_counter += 1
            self.agent.episode_start()
            self.ep_step_counter = 0
            self.draw_action(state)
            sample = state, self.last_output
            self.dataset.add_first_sample(sample, False)

        else:
            print 'step', self.name
            self.draw_action(state)
            sample = state, self.last_output, reward, absorbing, last or self.last
            self.dataset.add_sample(sample, False)

        if learn_flag and \
            (self.curr_step_counter == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            #print self.curr_episode_counter
            self.fit(self.dataset.get())
            self.dataset.empty_dataset()

        self.last = self.ep_step_counter >= self.horizon or last

        self.alarm_output = self.last

    def draw_action(self, state):
        self.last_input = state
        self.last_output = self.agent.draw_action(state)
        self.curr_step_counter += 1
        self.ep_step_counter += 1

        '''if self.name == 'control block H':
            print '-------------------------------------------------------------------------'
            print self.name, self.last_output
    '''



    def check_no_of_eps(self, dataset):
        i = 0
        size_eps = list()
        for dataset_step in dataset:
            if dataset_step[-1]:
                i += 1
            else:
                i += 1
                size_eps.append(i)
                i = 0
        return size_eps


    def fit(self, dataset):
        for step in dataset:
            print step
            import time
            time.sleep(1)
        self.agent.fit(dataset)
        self.curr_episode_counter = 0
        self.curr_step_counter = 0
        for c in self.callbacks:
            callback_pars = dict(dataset=dataset)
            c(**callback_pars)

    def add_reward(self, reward_block):
        self.reward_connection = reward_block

    def reset(self):

        self.agent.episode_start()
        self.ep_step_counter = 0
        self.last = True
        self.alarm_output = True
        self.dataset.empty_dataset()


