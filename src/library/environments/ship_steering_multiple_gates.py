import numpy as np

from mushroom.utils.angles_utils import normalize_angle
from mushroom.utils import spaces

from mushroom.environments import Environment, MDPInfo


class ShipSteeringMultiGate(Environment):
    """
    The Ship Steering environment as presented in:
    "Hierarchical Policy Gradient Algorithms". Ghavamzadeh M. and Mahadevan S..
    2013 with multiple gates.

    """
    def __init__(self, n_steps_action=3):

        self.__name__ = 'ShipSteeringMultiGate'
        self.n_steps_action = n_steps_action

        # MDP parameters
        self.no_of_gates = 4

        self.field_size = 1000
        low = np.array([0, 0, -np.pi, -np.pi / 12.])
        high = np.array([self.field_size, self.field_size, np.pi, np.pi / 12.])
        self.omega_max = np.array([np.pi / 12.])
        self._v = 3.
        self._T = 5.
        self._dt = .2

        self._gate1_sx = 800
        self._gate1_sy = 350
        self._gate1_ex = 900
        self._gate1_ey = 350

        gate_1s = np.array([self._gate1_sx, self._gate1_sy])
        gate_1e = np.array([self._gate1_ex, self._gate1_ey])

        gate_1 = np.array([gate_1s, gate_1e])

        self._gate2_sx = 300
        self._gate2_sy = 600
        self._gate2_ex = 400
        self._gate2_ey = 600

        gate_2s = np.array([self._gate2_sx, self._gate2_sy])
        gate_2e = np.array([self._gate2_ex, self._gate2_ey])

        gate_2 = np.array([gate_2s, gate_2e])

        self._gate3_sx = 500
        self._gate3_sy = 700
        self._gate3_ex = 600
        self._gate3_ey = 700

        gate_3s = np.array([self._gate3_sx, self._gate3_sy])
        gate_3e = np.array([self._gate3_ex, self._gate3_ey])

        gate_3 = np.array([gate_3s, gate_3e])

        self._gate4_sx = 300
        self._gate4_sy = 1000
        self._gate4_ex = 400
        self._gate4_ey = 1000

        gate_4s = np.array([self._gate4_sx, self._gate4_sy])
        gate_4e = np.array([self._gate4_ex, self._gate4_ey])

        gate_4 = np.array([gate_4s, gate_4e])

        self._gate_list = gate_1, gate_2, gate_3, gate_4



        # MDP properties
        observation_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-self.omega_max, high=self.omega_max)
        horizon = 5000
        gamma = .99
        self._out_reward = -100

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
        new_state = self._state

        for _ in range(self.n_steps_action):

            state = new_state

            new_state = np.empty(5)

            new_state[0] = state[0] + self._v * np.cos(state[2]) * self._dt
            new_state[1] = state[1] + self._v * np.sin(state[2]) * self._dt
            new_state[2] = normalize_angle(state[2] + state[3] * self._dt)
            new_state[3] = state[3] + (r - state[3]) * self._dt / self._T
            new_state[4] = state[4]

            if new_state[0] > self.field_size \
               or new_state[1] > self.field_size \
               or new_state[0] < 0 or new_state[1] < 0:
                reward = self._out_reward
                absorbing = True
                break
            elif self._through_gate(self._state[:2], new_state[:2], int(state[4])):
                gate_to_pass = int(state[4])
                if gate_to_pass == 0:
                    reward = 10
                    new_state[4] = 1
                    absorbing = False
                elif gate_to_pass == 1:
                    reward = 20
                    new_state[4] = 2
                    absorbing = False
                elif gate_to_pass == 2:
                    reward = 40
                    new_state[4] = 3
                    absorbing = False
                elif gate_to_pass == 3:
                    reward = 100
                    new_state[4] = 4
                    absorbing = True
            else:
                reward = -1
                absorbing = False

        self._state = new_state

        return self._state, reward, absorbing, {}


    def _through_gate(self, start, end, counter):

        gate_ = self._gate_list[counter]
        gate_e = gate_[1]
        gate_s = gate_[0]

        r = gate_e - gate_s
        s = end - start
        den = self._cross_2d(vecr=r, vecs=s)

        if den == 0:
            return False

        t = self._cross_2d((start - gate_s), s) / den
        u = self._cross_2d((start - gate_s), r) / den

        return 1 >= u >= 0 and 1 >= t >= 0

    @staticmethod
    def _cross_2d(vecr, vecs):
        return vecr[0] * vecs[1] - vecr[1] * vecs[0]
