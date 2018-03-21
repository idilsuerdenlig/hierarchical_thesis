from mushroom.algorithms.agent import Agent
from mushroom.utils.parameters import *
import numpy as np

class GhavamzadeAgent(Agent):

    def __init__(self, policy, mdp_info, learning_rate, features=None):
        self.learning_rate = learning_rate
        self.z = np.zeros(policy.weights_size)


        super(GhavamzadeAgent, self).__init__(policy, mdp_info, features)

    def fit(self, dataset):
        #print len(dataset)
        assert len(dataset) == 1
        print(dataset)


        state, action, reward, next_state, absorbing = self._parse(dataset)
        self._update(state, action, reward, next_state, absorbing)

    def _update(self, state, action, reward, next_state, absorbing):

        if self.phi is not None:
            state = self.phi(state)

        self.z = self.z + self.policy.diff_log(state, action)

        theta = self.policy.get_weights()
        if np.isinf(np.linalg.norm(reward*self.z)):
            print('ERROR DIVERGING PARAMETER')
            exit()

        gradient = reward*self.z
        theta = theta + self.learning_rate(gradient)*gradient
        self.policy.set_weights(theta)
        if absorbing:
            self.z = np.zeros(self.policy.weights_size)



    def _parse(self, dataset):

        sample = dataset[0]
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]

        if self.phi is not None:
           state = self.phi(state)

        return state, action, reward, next_state, absorbing



    def __str__(self):
        return self.__name__
