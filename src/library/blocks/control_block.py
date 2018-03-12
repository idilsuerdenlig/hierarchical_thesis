from block import Block
import numpy as np
from library.utils.dataset_manager import DatasetManager

class ControlBlock(Block):
    """
    This class implements the functions to initialize and move the agent drawing
    actions from its policy.

    """
    def __init__(self, name, agent, termination_condition=None, n_eps_per_fit=None, n_steps_per_fit=None, callbacks=list()):

        self.agent = agent
        self.ep_step_counter = 0
        self.curr_episode_counter = 0
        self.n_eps_per_fit = n_eps_per_fit
        self.n_steps_per_fit = n_steps_per_fit
        self.dataset = DatasetManager()
        self.horizon = self.agent.mdp_info.horizon
        self.gamma = self.agent.mdp_info.gamma
        self.last_input = None
        self.last_output = None
        self.last = False
        self.callbacks = callbacks
        self.terminated = False
        if termination_condition is None:
            self.termination_condition = lambda x : False
        else:
            self.termination_condition = termination_condition

        super(ControlBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):

        state = np.concatenate(inputs, axis=0)

        if self.last or last:
            self.curr_episode_counter += 1

        if self.last:
            if not self.terminated:

                self.last_call(inputs, reward, absorbing, learn_flag)
            self.reset(inputs)
        else:
            self.draw_action(state, last)
            sample = state, self.last_output, reward, absorbing, last or self.last
            #if self.name == 'control block 1':
            #    print self.name, 'STEP-----------------------------------------------------'
            #    print sample
            self.dataset.add_sample(sample, False)

            self.last = self.ep_step_counter >= self.horizon or self.termination_condition(state)

        if learn_flag and \
            (len(self.dataset) == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            self.fit(self.dataset.get())
            self.dataset.empty()


        self.alarm_output = self.last

    def last_call(self, inputs, reward, absorbing, learn_flag):

        state = np.concatenate(inputs, axis=0)
        sample = state, None, reward, absorbing, True
        self.dataset.add_sample(sample, False)
        #if self.name == 'control block 1':
        #    print self.name, 'LAST STEP-----------------------------------------------------'
        #    print sample
        self.terminated = True
        if learn_flag and \
            (len(self.dataset) == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            self.fit(self.dataset.get())
            self.dataset.empty()

    def draw_action(self, state, last):
        if not last:
            self.last_input = state
            self.last_output = self.agent.draw_action(state)
            self.ep_step_counter += 1

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
        #if self.name == 'control block 1':
        #print self.name, 'FIT-----------------------------------------------------'
        self.agent.fit(dataset)
        self.curr_episode_counter = 0
        for c in self.callbacks:
            callback_pars = dict(dataset=dataset)
            c(**callback_pars)

    def reset(self, inputs):
        state = np.concatenate(inputs, axis=0)
        #if self.last_input is not None and self.name != 'control block H':
        #    self.build_sample(state, 0, True, True)
        self.agent.episode_start()
        self.ep_step_counter = 0
        self.draw_action(state, False)
        sample = state, self.last_output
        self.dataset.add_first_sample(sample, False)
        self.alarm_output = False
        self.last = False
        self.terminated = False

    def init(self):
        self.dataset.empty()
        self.ep_step_counter = 0
        self.curr_episode_counter = 0
        self.last_output = None
        self.last = False
        self.terminated = False
