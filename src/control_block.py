from block import Block
import numpy as np
class ControlBlock(Block):
    """
    This class implements the functions to initialize and move the agent drawing
    actions from its policy.

    """
    def __init__(self, name, agent, n_eps_per_fit=None, n_steps_per_fit=None, callbacks=list()):

        self.agent = agent
        self.step_counter = 0
        self.curr_step_counter = 0
        self.curr_episode_counter = 0
        self.n_eps_per_fit = n_eps_per_fit
        self.n_steps_per_fit = n_steps_per_fit
        self.dataset = list()
        self.horizon = self.agent.mdp_info.horizon
        self.gamma = self.agent.mdp_info.gamma
        self.last_input = None
        self.last_output = None
        self.callbacks = callbacks
        self.rewardlist = list()
        self.discounted_reward = 0

        super(ControlBlock, self).__init__(name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):
        #self.clock_counter += 1
        self.last = False
        self.rewardlist.append(reward)
        self.alarm_output = False

        print alarms, self.name
        if np.any(alarms):
            print 'wakey wakey', self.name

        if last or np.any(alarms):
            self.curr_step_counter += 1
            self.step_counter += 1
            self.last = last or self.step_counter >= self.horizon
            if isinstance(inputs[0], np.float64):
                state = inputs
            else:
                state = np.concatenate(inputs, axis=0)

            for index, _reward in enumerate(self.rewardlist):
                df = self.gamma**index
                self.discounted_reward += df*_reward
            if isinstance(self.discounted_reward, np.ndarray):
                self.discounted_reward = self.discounted_reward[0]
            print 'adding sample', self.name, 'STEP', self.step_counter
            sample = self.last_input, self.last_output, self.discounted_reward, state, absorbing, self.last
            self.dataset.append(sample)

            self.rewardlist = list()
            self.discounted_reward = 0

        if self.last:
            print self.name, 'EP ENDED'
            self.curr_episode_counter += 1
            self.step_counter = 0

        if learn_flag and \
            (self.curr_step_counter == self.n_steps_per_fit or self.curr_episode_counter == self.n_eps_per_fit):
            #print self.curr_episode_counter
            self.fit(self.dataset)
            self.dataset = list()

        if self.last and not last:
            #print self.name, 'HORIZON REACHED'
            self.agent.episode_start()
            print 'new EPISODE', self.name
            self.step_counter = 0

        if np.any(alarms) and not last:
            #print self.name, 'STEP', self.step_counter
            action = self.agent.draw_action(state)
            self.last_output = action
            self.last_input = state
            #self.clock_counter = 0

        self.alarm_output = self.last

    def fit(self, dataset):
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
        print self.name, 'RESETS'
        if isinstance(inputs[0], np.float64):
            state = inputs
        else:
            state = np.concatenate(inputs,axis=0)
        self.agent.episode_start()
        #self.clock_counter = 0
        self.step_counter = 0
        action = self.agent.draw_action(state=state)
        #print self.name, 'STEP 0'
        self.last_output = action
        self.last_input = state
        self.alarm_output = False


