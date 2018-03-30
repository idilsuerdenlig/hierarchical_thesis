import numpy as np

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles_utils import normalize_angle
from mushroom.utils.angles_utils import *

class ShipSteering(Environment):
    """
    The Ship Steering environment as presented in:
    "Hierarchical Policy Gradient Algorithms". Ghavamzadeh M. and Mahadevan S..
    2013.

    """
    def __init__(self, small=True, hard=False, n_steps_action=1):
        """
        Constructor.

        Args:
             small (bool, True): whether to use a small state space or not.
             hard (bool, False): whether to use -100 as reward for going
                                 outside or -10000. With -100 reward the
                                 environment is considerably harder.

        """
        self.__name__ = 'ShipSteering'

        # MDP parameters
        self.field_size = 150 if small else 1000
        low = np.array([0, 0, -np.pi, -np.pi / 12.])
        high = np.array([self.field_size, self.field_size, np.pi, np.pi / 12.])
        self.omega_max = np.array([np.pi / 12.])
        self.hard = hard
        self._v = 3.
        self._T = 5.
        self._dt = .2
        self._gate_s = np.empty(2)
        self._gate_e = np.empty(2)
        self._gate_s[0] = 100 if small else 350
        self._gate_s[1] = 120 if small else 400
        self._gate_e[0] = 120 if small else 450
        self._gate_e[1] = 100 if small else 400
        self._out_reward = -100 if self.hard else -10000
        self._success_reward = 0 if self.hard else 100
        self._small = small

        self.n_steps_action = n_steps_action
        # MDP properties
        observation_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-self.omega_max, high=self.omega_max)
        horizon = 5000
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(ShipSteering, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self._small:
                self._state = np.zeros(4)
            else:
                low = self.info.observation_space.low
                high = self.info.observation_space.high
                self._state = (high-low)*np.random.rand(4) + low
        else:
            self._state = state

        return self._state

    def step_low_level(self, action):

        r = np.maximum(-self.omega_max, np.minimum(self.omega_max, action[0]))
        new_state = np.empty(4)
        new_state[0] = self._state[0] + self._v * np.sin(self._state[2]) * \
                       self._dt
        new_state[1] = self._state[1] + self._v * np.cos(self._state[2]) * \
                       self._dt
        new_state[2] = normalize_angle(self._state[2] + self._state[3] * self._dt)
        new_state[3] = self._state[3] + (r - self._state[3]) * self._dt / \
                       self._T

        if new_state[0] > self.field_size or new_state[1] > self.field_size\
           or new_state[0] < 0 or new_state[1] < 0:
            reward = self._out_reward
            absorbing = True

        elif self._through_gate(self._state[:2], new_state[:2]):
            reward = self._success_reward
            absorbing = True
        else:
            reward = -1
            absorbing = False

        self._state = new_state

        return self._state, reward, absorbing, {}

    def step(self, action):

        for _ in range(self.n_steps_action):
            state, reward, absorbing,_ = self.step_low_level(action)
            if absorbing:
                break

        return state, reward, absorbing, {}

    def _through_gate(self, start, end):
        r = self._gate_e - self._gate_s
        s = end - start
        den = self._cross_2d(vecr=r, vecs=s)

        if den == 0:
            return False

        t = self._cross_2d((start - self._gate_s), s) / den
        u = self._cross_2d((start - self._gate_s), r) / den

        return 1 >= u >= 0 and 1 >= t >= 0

    @staticmethod
    def _cross_2d(vecr, vecs):
        return vecr[0] * vecs[1] - vecr[1] * vecs[0]
