import numpy as np
from scipy.integrate import odeint

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles_utils import normalize_angle
from mushroom.utils.viewer import Viewer


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

        gamma = 0.97

        self.Mr = 0.3 * 2
        self.Mp = 2.55
        self.Ip = 2.6e-2
        self.Ir = 4.54e-4 * 2
        self.l = 13.8e-2
        self.r = 5.5e-2
        self.dt = 1e-2
        self.g = 9.81
        self.max_u = 5

        self._random = random_start

        high = np.array([np.pi, 15, 75])

        # MDP properties
        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([-self.max_u]),
                                  high=np.array([self.max_u]))
        horizon = 300
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        self._viewer = Viewer(5*self.l, 5*self.l)
        self._last_x = 0

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

        self._last_x = 0

        return self._state

    def step(self, action):

        u = np.maximum(-self.max_u, np.minimum(self.max_u, action[0]))
        new_state = odeint(self._dynamics, self._state, [0, self.dt],
                           (u,))

        self._state = np.array(new_state[-1])
        self._state[0] = normalize_angle(self._state[0])

        if abs(self._state[0]) > np.pi / 2:
            absorbing = True
            reward = -10000
        else:
            absorbing = False
            Q = np.diag([3.0, 0.1, 0.1])

            x = self._state

            J = x.dot(Q).dot(x)

            reward = -J

        return self._state, reward, absorbing, {}

    def _dynamics(self, state, t, u):

        alpha = state[0]
        d_alpha = state[1]

        h1 = (self.Mr+self.Mp)*(self.r**2)+self.Ir
        h2 = self.Mp*self.r*self.l*np.cos(alpha)
        h3 = self.l**2*self.Mp+self.Ip


        omegaP = d_alpha

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

    def render(self, mode='human'):
        start = 2.5*self.l*np.ones(2)
        end = 2.5*self.l*np.ones(2)

        dx = -self._state[2] * self.r * self.dt

        self._last_x += dx

        if self._last_x > 2.5*self.l or self._last_x < -2.5*self.l:
            self._last_x = (2.5*self.l+self._last_x)%(5*self.l) - 2.5*self.l

        start[0] += self._last_x
        end[0] += -2*self.l*np.sin(self._state[0]) + self._last_x
        end[1] += 2*self.l*np.cos(self._state[0])

        if (start[0] > 5*self.l and end[0] > 5*self.l) \
                or (start[0] < 0 and end[0] < 0):
            start[0] = start[0] % 5*self.l
            end[0] = end[0] % 5*self.l

        self._viewer.line(start, end)
        self._viewer.circle(start, self.r)

        self._viewer.display(self.dt)


