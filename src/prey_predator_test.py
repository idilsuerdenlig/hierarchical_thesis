import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.policy import DeterministicPolicy
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.basis import PolynomialBasis
from mushroom.features import Features
from library.environments.prey_predator import PreyPredator

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_episodes, n_ep_per_fit):
    np.random.seed()

    # MDP
    mdp = PreyPredator()

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

    agent = REPS(dist, policy, mdp.info, 0.05, phi)


    # Train
    dataset_callback = CollectDataset()
    core = Core(agent, mdp, callbacks=[dataset_callback])

    for i in range(n_epochs):
        core.learn(n_episodes=n_episodes,
                   n_episodes_per_fit=n_ep_per_fit, render=True)
        J = compute_J(dataset_callback.get(), gamma=mdp.info.gamma)
        dataset_callback.clean()

        p = dist.get_parameters()

        print('mu:    ', p[:n_weights])
        print('sigma: ', p[n_weights:])
        print('Reward at iteration ' + str(i) + ': ' +
              str(np.sum(J)/n_episodes))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)

if __name__ == '__main__':
    n_epochs = 20
    n_episodes = 100
    n_ep_per_fit = 25

    experiment(n_epochs, n_episodes, n_ep_per_fit)
