import numpy as np

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces


class ShipSteeringMultiGate(Environment):
    """
    The Ship Steering environment as presented in:
    "Hierarchical Policy Gradient Algorithms". Ghavamzadeh M. and Mahadevan S..
    2013.

    """
    def __init__(self, small=True):
        """
        Constructor.

        Args:
             small (bool): whether to use a small state space or not.

        """
        self.__name__ = 'ShipSteeringMultiGate'

        # MDP parameters
        self.nrew = 0
        self.no_of_gates = 4
        self.gates_passed = np.zeros(shape=(self.no_of_gates,))
        self.gate_s = np.empty(shape=(self.no_of_gates,2))
        self.gate_e = np.empty(shape=(self.no_of_gates,2))

        self.field_size = 150 if small else 1000
        low = np.array([0, 0, -np.pi, -np.pi / 12.])
        high = np.array([self.field_size, self.field_size, np.pi, np.pi / 12.])
        self.omega_max = np.array([np.pi / 12.])
        self._v = 3.
        self._T = 5.
        self._dt = .2

        self._gate_s[0][0] = 30 if small else 80
        self._gate_s[0][1] = 50 if small else 100
        self._gate_e[0][0] = 50 if small else 100
        self._gate_e[0][1] = 30 if small else 80

        self._gate_s[1][0] = 30 if small else 80
        self._gate_s[1][1] = 100 if small else 900
        self._gate_e[1][0] = 50 if small else 100
        self._gate_e[1][1] = 120 if small else 920

        self._gate_s[2][0] = 100 if small else 900
        self._gate_s[2][1] = 120 if small else 920
        self._gate_e[2][0] = 120 if small else 920
        self._gate_e[2][1] = 100 if small else 900

        self._gate_s[3][0] = 100 if small else 900
        self._gate_s[3][1] = 30 if small else 80
        self._gate_e[3][0] = 120 if small else 920
        self._gate_e[3][1] = 50 if small else 100

        # MDP properties
        observation_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-self.omega_max, high=self.omega_max)
        horizon = 5000
        gamma = .99
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(ShipSteeringMultiGate, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.zeros(5)
        else:
            self._state = state

        return self._state

    def step(self, action):
        r = np.maximum(-self.omega_max, np.minimum(self.omega_max, action[0]))
        new_state = np.empty(5)
        new_state[0] = self._state[0] + self._v * np.sin(self._state[2]) *\
            self._dt
        new_state[1] = self._state[1] + self._v * np.cos(self._state[2]) *\
            self._dt
        new_state[2] = self._state[2] + self._state[3] * self._dt

        new_state[2] = (new_state[2] + np.pi) % (2 * np.pi) - np.pi

        new_state[3] = self._state[3] + (r - self._state[3]) * self._dt /\
            self._T

        if new_state[0] > self.field_size or new_state[1] > self.field_size\
           or new_state[0] < 0 or new_state[1] < 0:
            reward = -100
            absorbing = True
        else:
            self._through_gate(self._state[:2], new_state[:2])
            if self.gates_passed == [1, 0, 0, 0] and self.nrew == 0:
                reward = 10
                self.new_state[4] = 1
                absorbing = False
            elif self.gates_passed == [1, 1, 0, 0] and self.nrew == 1:
                reward = 20
                self.new_state[4] = 2
                absorbing = False
            elif self.gates_passed == [1, 1, 1, 0] and self.nrew == 2:
                reward = 30
                self.new_state[4] = 3
                absorbing = False
            elif self.gates_passed == [1, 1, 1, 1] and self.nrew == 3:
                reward = 40
                self.new_state[4] = 4
                absorbing = True
            else:
                reward = -1
                absorbing = False

        self._state = new_state

        return self._state, reward, absorbing, {}


    def _through_gate(self, start, end):

        for i in xrange(self.no_of_gates):
            r = self._gate_e[i] - self._gate_s[i]
            s = end - start
            den = self._cross_2d(vecr=r, vecs=s)

            if den is not 0:
                t = self._cross_2d((start - self._gate_s[i]), s) / den
                u = self._cross_2d((start - self._gate_s[i]), r) / den
                if 1 >= u >= 0 and 1 >= t >= 0:
                    self.gates_passed[i] = 1

    @staticmethod
    def _cross_2d(vecr, vecs):
        return vecr[0] * vecs[1] - vecr[1] * vecs[0]
