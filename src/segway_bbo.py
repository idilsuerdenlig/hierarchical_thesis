import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.policy import DeterministicPolicy
from mushroom.features import Features
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.basis import PolynomialBasis
from mushroom.features import Features
from library.environments.segway_linear_motion import SegwayLinearMotion

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_iteration, n_ep_per_fit, n_eval_run):
    np.random.seed()

    # MDP
    mdp = SegwayLinearMotion()

    basis = PolynomialBasis.generate(1, mdp.info.observation_space.shape[0])
    phi = Features(basis_list=basis[1:])


    # Features
    approximator = Regressor(LinearApproximator,
                   input_shape=(phi.size,),
                   output_shape=mdp.info.action_space.shape)

    n_weights = approximator.weights_size
    mu = np.zeros(n_weights)
    sigma = 2e-0*np.ones(n_weights)
    policy = DeterministicPolicy(approximator)
    dist = GaussianDiagonalDistribution(mu, sigma)

    agent = RWR(dist, policy, mdp.info, 0.01, phi)


    # Train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=n_eval_run, render=True)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start ', np.mean(J))

    for i in range(n_epochs):
        core.learn(n_episodes=n_iteration*n_ep_per_fit,
                   n_episodes_per_fit=n_ep_per_fit, render=False)

        dataset_eval = core.evaluate(n_episodes=n_eval_run, render=False)
        dataset_eval = core.evaluate(n_episodes=n_eval_run, render=False)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)

        p = dist.get_parameters()

        print('mu:    ', p[:n_weights])
        print('sigma: ', p[n_weights:])
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

