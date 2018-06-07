import numpy as np

from mushroom.utils.angles_utils import normalize_angle
from mushroom.utils import spaces

from mushroom.environments import Environment, MDPInfo
from mushroom.utils.viewer import Viewer


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
        low = np.array([0, 0, -np.pi, -np.pi / 12., 0])
        high = np.array([self.field_size, self.field_size, np.pi, np.pi / 12.,
                         self.no_of_gates])
        self.omega_max = np.array([np.pi / 12.])
        self._v = 3.
        self._T = 5.
        self._dt = .2

        gate_1s = np.array([800, 350])
        gate_1e = np.array([900, 350])

        gate_1 = np.array([gate_1s, gate_1e])

        gate_2s = np.array([300, 600])
        gate_2e = np.array([400, 600])

        gate_2 = np.array([gate_2s, gate_2e])

        gate_3s = np.array([500, 700])
        gate_3e = np.array([600, 700])

        gate_3 = np.array([gate_3s, gate_3e])

        gate_4s = np.array([300, 999])
        gate_4e = np.array([400, 999])

        gate_4 = np.array([gate_4s, gate_4e])

        self._gate_list = gate_1, gate_2, gate_3, gate_4

        self.gate_to_pass = 0

        # MDP properties
        observation_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-self.omega_max, high=self.omega_max)
        horizon = 5000
        gamma = .99
        self._out_reward = -10000
        self.correct_order = False

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = Viewer(self.field_size, self.field_size,
                              background=(66, 131, 237))

        super(ShipSteeringMultiGate, self).__init__(mdp_info)

    def reset(self, state=None):

        if state is None:
            self._state = np.zeros(8)
            self._state[0] = 500
            self._state[1] = 500

        else:
            self._state = state
        #self.gate_to_pass = 0
        #self.correct_order = False
        return self._state

    def step(self, action):

        r = np.maximum(-self.omega_max, np.minimum(self.omega_max, action[0]))
        new_state = self._state

        for _ in range(self.n_steps_action):

            state = new_state

            new_state = np.empty(8)

            new_state[0] = state[0] + self._v * np.cos(state[2]) * self._dt
            new_state[1] = state[1] + self._v * np.sin(state[2]) * self._dt
            new_state[2] = normalize_angle(state[2] + state[3] * self._dt)
            new_state[3] = state[3] + (r - state[3]) * self._dt / self._T
            new_state[4] = state[4]
            new_state[5] = state[5]
            new_state[6] = state[6]
            new_state[7] = state[7]

            if new_state[0] > self.field_size \
               or new_state[1] > self.field_size \
               or new_state[0] < 0 or new_state[1] < 0:
                reward = self._out_reward
                absorbing = True
                break

            elif self._through_gate(self._state[:2], new_state[:2], 0):
                state[4] += 1
                if state[4] == 1:
                    reward = 10
                else:
                    reward = 0
                absorbing = False
            elif self._through_gate(self._state[:2], new_state[:2], 1):
                state[5] += 1
                if state[5] == 1:
                    reward = 10
                else:
                    reward = 0
                absorbing = False
            elif self._through_gate(self._state[:2], new_state[:2], 2):
                state[6] += 1
                if state[6] == 1:
                    reward = 10
                else:
                    reward = 0
                absorbing = False
            elif self._through_gate(self._state[:2], new_state[:2], 3):
                state[7] += 1
                if state[7] == 1:
                    reward = 10
                    absorbing = True
                else:
                    reward = 0
                    absorbing = False

            else:
                reward = 0
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

    def render(self, mode='human'):

        for gate in self._gate_list:
            gate_s = gate[0]
            gate_e = gate[1]
            self._viewer.line(gate_s, gate_e, width=3)

        boat = [
            [-4, -4],
            [-4, 4],
            [4, 4],
            [8, 0.0],
            [4, -4]
        ]
        self._viewer.polygon(self._state[:2], self._state[2], boat,
                             color=(32, 193, 54))

        self._viewer.display(self._dt)
