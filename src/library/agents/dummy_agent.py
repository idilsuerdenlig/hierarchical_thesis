import numpy as np
class SimpleAgent(object):
    """
    This class implements the functions to initialize and move the agent drawing
    actions from its policy.

    """
    def __init__(self, name, mdp_info, params=None, features=None, policy=None):
        """
        Constructor.

        Args:
            policy (object): the policy to use for the agent;
            gamma (float): discount factor;
            params (dict): other parameters of the algorithm;
            features (object, None): features to use for the input of the
                approximator.

        """
        self.mdp_info = mdp_info
        self.name = name
        self.action_space = mdp_info.action_space
        self.observation_space = mdp_info.observation_space
        self.gamma = mdp_info.gamma
        self.horizon = mdp_info.horizon
        self.params = params
        self.phi = features
        self.policy = policy

        self._next_action = None

    def initialize(self, mdp_info):
        """
        Fill the dictionary with information about the MDP.

        Args:
            mdp_info (dict): MDP information.

        """
        self.counter = 0
        for k, v in mdp_info.iteritems():
            self.mdp_info[k] = v

    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (list): the dataset;
            n_iterations (int): number of fit steps of the approximator.

        """
    def draw_action(self, state):
        """
        Return the action to execute. It is the action returned by the policy
        or the action set by the algorithm (e.g. SARSA).

        Args:
            state (np.array): the state where the agent is.

        Returns:
            The action to be executed.

        """
        if self.name == 'HIGH':
            action = np.array([110,110])
        elif self.policy is None:
            action = np.random.choice(self.action_space.n)
            action = np.array([4])
        else:
            action = self.policy.draw_action(state)
        self.counter += 1

        return action

    def episode_start(self):
        """
        Reset some parameters when a new episode starts. It is used only by
        some algorithms (e.g. DQN).

        """
        self.counter = 0


    def parse(self, sample):
        """
        Utility to parse the sample.

        Args:
             sample (list): the current episode step.

        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag. If provided, `state` is preprocessed with the features.

        """
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]
        last = sample[5]

        if self.phi is not None:
            state = self.phi(state)

        return state, action, reward, next_state, absorbing, last
