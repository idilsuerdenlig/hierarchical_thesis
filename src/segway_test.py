import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.policy import DeterministicPolicy
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.parameters import AdaptiveParameter
from mushroom.features.basis import PolynomialBasis
from mushroom.features.tiles import Tiles
from mushroom.features import Features
from library.environments.segway import Segway

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_episodes, n_ep_per_fit):
    np.random.seed()

    # MDP
    mdp = Segway()


    # Features
    basis = PolynomialBasis.generate(1, mdp.info.observation_space.shape[0])
    phi = Features(basis_list=basis)
    # Agent
    approximator = Regressor(LinearApproximator,
                   input_shape=(phi.size,),
                   output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(approximator)

    n_weights = approximator.weights_size

    mu = np.zeros(n_weights)
    sigma = 1e-0*np.ones(n_weights)
    dist = GaussianDiagonalDistribution(mu, sigma)

    lr = AdaptiveParameter(1e-1)
    agent = REPS(dist, policy, mdp.info, 0.5, features=phi)


    # Train
    dataset_callback = CollectDataset()
    core = Core(agent, mdp, callbacks=[dataset_callback])

    for i in range(n_epochs):
        core.learn(n_episodes=n_episodes,
                   n_episodes_per_fit=n_ep_per_fit, render=False)
        J = compute_J(dataset_callback.get(), gamma=1.0)
        dataset_callback.clean()
        print('Reward at iteration ' + str(i) + ': ' +
              str(np.sum(J)/n_episodes))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_steps=5000, render=True)

if __name__ == '__main__':
    n_epochs = 24
    n_episodes = 100
    n_ep_per_fit = 10

    experiment(n_epochs, n_episodes, n_ep_per_fit)
