from mushroom.algorithms.agent import Agent
import numpy as np

class GhavamzadeAgent(Agent):

    def __init__(self, policy, mdp_info, params, features):
        self.learning_rate = params['algorithm_params'].pop('learning_rate')
        self.z = np.zeros(policy.weights_size)


        super(GhavamzadeAgent, self).__init__(policy, mdp_info, params, features)

    def fit(self, dataset):
        assert len(dataset) == 1

        state, action, reward, next_state, absorbing = self._parse(dataset)
        self._update(state, action, reward, next_state, absorbing)


    def _update(self, state, action, reward, next_state, absorbing):
        if absorbing:
            self.z = np.zeros(self.policy.weights_size)

        else:
            self.z = self.z + self.policy.diff_log(state, action)

        theta = self.policy.get_weights()
        theta = theta + self.learning_rate(state, action)*reward*self.z
        self.policy.set_weights(theta)


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
