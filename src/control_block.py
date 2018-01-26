from block import Block
import numpy as np
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
        self.dataset = list()
        self.horizon = self.agent.mdp_info.horizon
        self.gamma = self.agent.mdp_info.gamma
        self.last_input = None
        self.last_output = None
        self.last = False
        self.callbacks = callbacks

        super(ControlBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):

        self.alarm_output = self.last

        if isinstance(inputs[0], np.float64):
            state = inputs
        else:
            state = np.concatenate(inputs, axis=0)

        self.build_sample(state, reward, absorbing, last)

        if learn_flag and \
            (self.curr_step_counter == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            #print self.curr_episode_counter
            self.fit(self.dataset)
            self.dataset = list()

        self.draw_action(state, last)

        self.last = self.ep_step_counter >= self.horizon

        if self.last or last:
            self.curr_episode_counter += 1

        if self.last:
            self.ep_step_counter = 0
            if not last:
                self.agent.episode_start()



    def draw_action(self, state, last):
        if not last:
            self.last_input = state
            self.last_output = self.agent.draw_action(state)
            self.curr_step_counter += 1
            self.ep_step_counter += 1


    def build_sample(self, next_state, reward, absorbing, last):
        sample = self.last_input, self.last_output, reward, next_state, absorbing, last or self.last
        self.dataset.append(sample)

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

        if dataset[0][0] is None:
            dataset = dataset[1:]


        if self.name == 'control block 1' or self.name == 'control block 2':
            print '-----------------------------------------------------------', self.name
            for step in dataset:
                print step
        if np.any(self.check_no_of_eps(dataset)) > self.horizon:
            exit()


        #print self.name, 'len of dataset   :', len(dataset), 'FITS'
        #if len(dataset)>45 and (self.name == 'control block 1' or 'control block 2'):
        #    for i in dataset:
        #        print i
        #    exit()
        self.agent.fit(dataset)
        self.curr_episode_counter = 0
        self.curr_step_counter = 0
        for c in self.callbacks:
            callback_pars = dict(dataset=dataset)
            c(**callback_pars)

    def add_reward(self, reward_block):
        self.reward_connection = reward_block

    def reset(self, inputs):
        if self.name == 'control block 1' or self.name == 'control block 2':
            print inputs
        if isinstance(inputs[0], np.float64):
            state = inputs
        else:
            state = np.concatenate(inputs,axis=0)
        #if self.last_input is not None and self.name != 'control block H':
        #    self.build_sample(state, 0, True, True)
        self.agent.episode_start()
        self.ep_step_counter = 0
        self.draw_action(state, False)
        self.alarm_output = False
        self.last = False


