import numpy as np
from scipy.integrate import odeint

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles_utils import normalize_angle


class Segway(Environment):
    """
    The Segway environment (continuous version) as presented in:
    "Deep Learning for Actor-Critic Reinforcement Learning". Xueli Jia. 2015.
    """
    def __init__(self, random_start=False):
        """
        Constructor.

        Args:
            random_start: whether to start from a random position or from the
                          horizontal one

        """
        # MDP parameters

        self.gamma = 0.99
        self.stateDimensionality = 3
        self.actionDimensionality = 1
        self.rewardDimensionality = 1
        self.statesNumber = 0
        self.actionsNumber = 0
        self.isFiniteHorizon = False
        self.isAverageReward = False
        self.isEpisodic = True
        self.horizon = 300

        self.Mr = 0.3 * 2
        self.Mp = 2.55
        self.Ip = 2.6e-2
        self.Ir = 4.54e-4 * 2
        self.l = 13.8e-2
        self.r = 5.5e-2
        self.dt = 1e-2
        self.g = 9.81
        self.mup = 1e-4
        self.mur = 6.5e-3
        self._max_omega = np.pi*25/180
        self.max_u = np.pi*5/180

        self._random = random_start

        high = np.array([np.pi, self._max_omega, self._max_omega])

        # MDP properties
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-self.max_u]),
                                  high=np.array([self.max_u]))
        horizon = 5000
        mdp_info = MDPInfo(observation_space, action_space, self.gamma, horizon)

        super(Segway, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self._random:
                angle = np.random.uniform(-np.pi, np.pi)
            else:
                angle = -np.pi/8

            self._state = np.array([angle, 0., 0.])
        else:
            self._state = state
            self._state[0] = normalize_angle(self._state[0])
            self._state[1] = np.maximum(-self._max_omega,
                                        np.minimum(self._state[1],
                                                   self._max_omega))
            self._state[2] = np.maximum(-self._max_omega,
                                        np.minimum(self._state[2],
                                                   self._max_omega))

        return self._state

    def step(self, action):

        u = np.maximum(-self.max_u, np.minimum(self.max_u, action[0]))
        new_state = odeint(self._dynamics, self._state, [0, self.dt],
                           (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])
        self._state[1] = np.maximum(-self._max_omega,
                                    np.minimum(self._state[1],
                                               self._max_omega))
        self._state[2] = np.maximum(-self._max_omega,
                                    np.minimum(self._state[1],
                                               self._max_omega))

        Q = np.diag([3.0, 0.1, 0.1])
        R = np.array([0.01])

        x = self._state

        J = np.transpose(x) * Q * x + np.transpose(u) * R * u;

        reward = -J[0];

        return self._state, reward, False, {}

    def _dynamics(self, state, t, u):

        alpha = state[0]
        d_alpha = np.maximum(-self._max_omega,
                             np.minimum(state[1], self._max_omega))

        h1 = (self.Mr+self.Mp)*(self.r**2)+self.Ir
        h2 = self.Mp*self.r*self.l*np.cos(alpha)
        h3 = self.l**2*self.Mp+self.Ip


        omegaP = d_alpha

        #angular acceleration[rad / s ^ 2]

        dOmegaP = -(h2*self.l*self.Mp*self.r*np.sin(alpha)*omegaP**2 -
                    self.g*h1*self.l*self.Mp*np.sin(alpha) + (h2+h1)*u)\
                  / (h1*h3-h2**2)
        dOmegaR = (h3*self.l*self.Mp*self.r*np.sin(alpha)*omegaP**2 -
                   self.g*h2*self.l*self.Mp*np.sin(alpha) + (h3+h2)*u)\
                  / (h1*h3-h2**2)

        dx = list()
        dx.append(omegaP)
        dx.append(dOmegaP)
        dx.append(dOmegaR)

        return dx



