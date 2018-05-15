import numpy as np

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles_utils import normalize_angle,\
    shortest_angular_distance
from mushroom.utils.viewer import Viewer


class PreyPredator(Environment):
    """
    A prey-predator environment environment. A Predator must catch a faster prey
    in an environment with obstacles.
    """
    def __init__(self):
        self._rotation_radius = 0.6
        self._catch_radius = 0.4

        self._v_prey = 0.11
        self._v_predator = 0.1
        self._dt = 0.1

        self._omega_prey = self._v_prey / self._rotation_radius
        self._omega_predator = self._v_predator / self._rotation_radius

        self._max_x = 5.0
        self._max_y = 5.0

        self._obstacles = [
            (np.array([self._max_x/5,
                       self._max_y - 3.5*self._catch_radius]),
             np.array([self._max_x,
                       self._max_y - 3.5*self._catch_radius])),

            (np.array([-3/5*self._max_x,
                       self._max_y/4]),
             np.array([-3/5*self._max_x,
                       -2/5*self._max_y])),

            (np.array([-3/5*self._max_x + 3.5*self._catch_radius,
                       self._max_y / 4]),
             np.array([-3/5*self._max_x + 3.5*self._catch_radius,
                       -2/5*self._max_y])),

            (np.array([-3/5*self._max_x,
                       self._max_y/4]),
             np.array([-3/5*self._max_x + 3.5*self._catch_radius,
                       self._max_y/4]))
            ]

        # Add bounds of the map
        self._obstacles += [(np.array([-self._max_x, -self._max_y]),
                             np.array([-self._max_x, self._max_y])),

                            (np.array([-self._max_x, -self._max_y]),
                            np.array([self._max_x, -self._max_y])),

                            (np.array([self._max_x, self._max_y]),
                             np.array([-self._max_x, self._max_y])),

                            (np.array([self._max_x, self._max_y]),
                             np.array([self._max_x, -self._max_y]))
                            ]

        high = np.array([self._max_x, self._max_y, np.pi,
                         self._max_x, self._max_y, np.pi])


        # MDP properties
        horizon = 2000
        gamma = 0.99

        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([0,
                                                -self._omega_predator]),
                                  high=np.array([self._v_predator,
                                                 self._omega_predator]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        width = 500
        height = int(width * self._max_y / self._max_x)
        self._viewer = Viewer(2*self._max_x, 2*self._max_y, width, height)

        super(PreyPredator, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.array([0., 0., 0.,
                                    self._max_x/2, self._max_y/2, np.pi/2])
            self._state = np.array([-3., 0., 0.,
                                    -4., -1., 0.])
        else:
            self._state = state
            self._state[2] = normalize_angle(self._state[2])
            self._state[5] = normalize_angle(self._state[5])

        return self._state

    def step(self, action):
        # compute new predator state
        u = self._bound(action,
                        self.info.action_space.low,
                        self.info.action_space.high)

        state_predator = self._state[:3]
        state_predator = self._differential_drive_dynamics(state_predator, u)

        # Compute new prey state
        u_prey = self._prey_controller(self._state)
        state_prey = self._state[3:]
        state_prey = self._differential_drive_dynamics(state_prey, u_prey)

        # Update state
        self._state = np.concatenate([state_predator, state_prey], 0)

        delta_norm_new = np.linalg.norm(self._state[:2]-self._state[3:5])

        if delta_norm_new < self._catch_radius:
            absorbing = True
            reward = 100
        else:
            absorbing = False
            reward = -1


        return self._state, reward, absorbing, {}

    def _prey_controller(self, state):
        delta_norm = np.linalg.norm(state[:2] - state[3:5])

        if delta_norm > 3.0:
            velocity_prey = 0
        elif delta_norm > 1.5:
            velocity_prey = self._v_prey / 2
        else:
            velocity_prey = self._v_prey

        if velocity_prey > 0:
            cos_theta = np.cos(state[5])
            sin_theta = np.sin(state[5])
            increment = self._rotation_radius*np.array([cos_theta, sin_theta])

            collision, i = self._check_collision(state[3:5],
                                                 state[3:5]+increment)

            if collision is None:
                omega_prey = self._avoid_predator(state)
            else:
                obstacle = self._obstacles[i]
                velocity_prey, omega_prey = self._avoid_obstacle(obstacle,
                                                                 collision,
                                                                 state,
                                                                 velocity_prey)
        else:
            omega_prey = self._avoid_predator(state)

        u_prey = np.empty(2)
        u_prey[0] = velocity_prey
        u_prey[1] = self._bound(omega_prey, -self._omega_prey, self._omega_prey)

        return u_prey

    def _avoid_obstacle(self, obstacle, collision, state, velocity_prey):
        dir1 = collision - state[3:5]
        dir2 = obstacle[1] - obstacle[0]
        alpha = self._vector_angle(dir1, dir2)

        # Avoid near obstacles.
        if abs(alpha) < np.pi/4:
            omega_prey = -self._omega_prey
        else:
            omega_prey = self._omega_prey

        collision_distance = np.linalg.norm(state[3:5] - collision)

        if collision_distance < self._rotation_radius:
            velocity_prey = collision_distance * np.abs(omega_prey)

        return velocity_prey, omega_prey

    def _avoid_predator(self, state):
        attack_angle = normalize_angle(np.arctan2(state[4] - state[1],
                                                  state[3] - state[0]))

        angle_current = shortest_angular_distance(attack_angle, state[5])

        if angle_current < 0:
            escape_angle = normalize_angle(attack_angle - np.pi / 4)
        else:
            escape_angle = normalize_angle(attack_angle + np.pi / 4)

        delta_angle = shortest_angular_distance(state[5], escape_angle)

        omega_prey = delta_angle * (2 / np.pi)

        return omega_prey

    @staticmethod
    def _cross_2d(vecr, vecs):
        return vecr[0] * vecs[1] - vecr[1] * vecs[0]

    @staticmethod
    def _vector_angle(x, y):
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        cos_alpha = x.dot(y)/x_norm/y_norm

        return np.arccos(cos_alpha)

    def _check_collision(self, start, end):
        collision = None

        min_u = np.inf

        for i, obstacle in enumerate(self._obstacles):
            r = obstacle[1] - obstacle[0]
            s = end - start
            den = self._cross_2d(vecr=r, vecs=s)

            if den != 0:
                t = self._cross_2d((start - obstacle[0]), s) / den
                u = self._cross_2d((start - obstacle[0]), r) / den

                if 1 >= u >= 0 and 1 >= t >= 0:

                    if u < min_u:
                        collision = start + (u-1e-3)*s
                        min_u = u

        return collision, i

    def _differential_drive_dynamics(self, state, u):
        delta = np.empty(3)

        delta[0] = np.cos(state[2]) * u[0]
        delta[1] = np.sin(state[2]) * u[0]
        delta[2] = u[1]

        new_state = state + delta

        new_state[0] = self._bound(new_state[0], -self._max_x, self._max_x)
        new_state[1] = self._bound(new_state[1], -self._max_y, self._max_y)
        new_state[2] = normalize_angle(new_state[2])

        collision, _ = self._check_collision(state[:2], new_state[:2])

        if collision is not None:
            new_state[:2] = collision

        return new_state

    def render(self, mode='human'):
        center = np.array([self._max_x, self._max_y])

        predator_pos = self._state[:2]
        predator_theta = self._state[2]

        prey_pos = self._state[3:5]
        prey_theta = self._state[5]


        # Predator
        self._viewer.circle(center + predator_pos, self._catch_radius,
                            (255, 255, 255))
        self._viewer.arrow_head(center + predator_pos, self._catch_radius,
                                predator_theta, (255, 0, 0))

        # Prey
        self._viewer.arrow_head(center + prey_pos, self._catch_radius,
                                prey_theta, (0, 0, 255))

        # Obstacles
        for obstacle in self._obstacles:
            start = obstacle[0]
            end = obstacle[1]
            self._viewer.line(center + start, center + end)



        self._viewer.display(self._dt)



