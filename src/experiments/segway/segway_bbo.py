import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.utils.dataset import compute_J
from mushroom.utils.angles_utils import shortest_angular_distance
from mushroom_hierarchical.environments.segway_linear_motion import SegwayLinearMotion

from tqdm import tqdm
tqdm.monitor_interval = 0


class SegwayControlPolicy:
    def __init__(self, weights):
        self._weights = weights

    def __call__(self, state, action):
        policy_action = np.atleast_1d(np.abs(self._weights).dot(state))

        return 1. if np.array_equal(action, policy_action) else 0.

    def draw_action(self, state):
        angle_setpoint = state[0]*self._weights[0]

        new_state = state[1:]

        new_state[0] = shortest_angular_distance(angle_setpoint, new_state[0])

        return np.atleast_1d(np.abs(self._weights[1:]).dot(new_state))

    def diff(self, state, action):
        raise RuntimeError('Deterministic policy is not differentiable')

    def diff_log(self, state, action):
        raise RuntimeError('Deterministic policy is not differentiable')

    def set_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights

    @property
    def weights_size(self):
        return len(self._weights)

    def __str__(self):
        return self.__name__


def experiment(n_epochs, n_iteration, n_ep_per_fit, n_eval_run):
    np.random.seed()

    # MDP
    mdp = SegwayLinearMotion()

    input_dim = mdp.info.observation_space.shape[0]
    mu = np.zeros(input_dim)
    sigma = 2e-0*np.ones(input_dim)
    policy = SegwayControlPolicy(mu)
    dist = GaussianDiagonalDistribution(mu, sigma)

    agent = RWR(dist, policy, mdp.info, 0.01)


    # Train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=n_eval_run, render=False)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start ', np.mean(J))

    for i in range(n_epochs):
        core.learn(n_episodes=n_iteration*n_ep_per_fit,
                   n_episodes_per_fit=n_ep_per_fit, render=False)

        dataset_eval = core.evaluate(n_episodes=n_eval_run, render=True)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)

        p = dist.get_parameters()

        print('mu:    ', p[:input_dim])
        print('sigma: ', p[input_dim:])
        print('J at iteration ' + str(i) + ': ' +
              str(np.mean(J)))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    experiment(n_epochs=20,
               n_iteration=4,
               n_ep_per_fit=25,
               n_eval_run=10)

