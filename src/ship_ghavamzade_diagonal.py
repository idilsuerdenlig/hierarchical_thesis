import numpy as np


from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles_utils import normalize_angle, shortest_angular_distance


class ShipGhavamzadeDiagonal(Environment):
    """
    The Ship Steering environment as presented in:
    "Hierarchical Policy Gradient Algorithms". Ghavamzadeh M. and Mahadevan S..
    2013.

    """
    def __init__(self, small=True, hard=False):
        """
        Constructor.

        Args:
             small (bool, True): whether to use a small state space or not.
             hard (bool, False): whether to use -100 as reward for going
                                 outside or -10000. With -100 reward the
                                 environment is considerably harder.

        """
        self.__name__ = 'ShipGhavamzadeDiagonal'

        # MDP parameters
        self.field_size = 150
        low = np.array([0, 0, -np.pi, -np.pi / 12.])
        high = np.array([self.field_size, self.field_size, np.pi, np.pi / 12.])
        self.omega_max = np.array([np.pi / 12.])
        self._v = 3.
        self._T = 5.
        self._dt = .2
        self.goal_pos = np.array([140, 140])
        self._out_reward = -100
        self._success_reward = 100

        # MDP properties
        observation_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-self.omega_max, high=self.omega_max)
        horizon = 5000
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(ShipGhavamzadeDiagonal, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:

            self._state = np.array([40, 40, np.random.uniform(-np.pi, np.pi), 0])
        else:
            self._state = state

        return self._state


    def step(self, action):
        r = np.maximum(-self.omega_max, np.minimum(self.omega_max, action[0]))
        new_state = np.empty(4)
        new_state[0] = self._state[0] + self._v * np.sin(self._state[2]) *\
            self._dt
        new_state[1] = self._state[1] + self._v * np.cos(self._state[2]) *\
            self._dt
        new_state[2] = normalize_angle(self._state[2] + self._state[3] * self._dt)
        new_state[3] = self._state[3] + (r - self._state[3]) * self._dt /\
            self._T

        pos = np.array([new_state[0], new_state[1]])
        if new_state[0] > self.field_size or new_state[1] > self.field_size\
           or new_state[0] < 0 or new_state[1] < 0:
            reward = self._out_reward
            absorbing = True
        elif np.linalg.norm(pos - self.goal_pos) <= 10:
            reward = self._success_reward
            absorbing = True
        else:
            reward = -1
            absorbing = False

        theta_ref = normalize_angle(np.arctan2(pos[1] - self.goal_pos[1], pos[0] - self.goal_pos[0]))
        theta = new_state[2]
        theta = normalize_angle(np.pi / 2 - theta)
        del_theta = shortest_angular_distance(from_angle=theta, to_angle=theta_ref)
        power = -del_theta ** 2 / ((np.pi / 6) * (np.pi / 6))
        reward = reward+np.expm1(power)

        self._state = new_state

        return self._state, reward, absorbing, {}


