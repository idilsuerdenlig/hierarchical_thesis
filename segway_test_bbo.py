import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.algorithms.actor_critic import COPDAC_Q
from mushroom.policy import DeterministicPolicy, GaussianPolicy
from mushroom.distributions import GaussianDiagonalDistribution, GaussianDistribution
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.parameters import AdaptiveParameter, Parameter
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
    approximator = Regressor(LinearApproximator,
                   input_shape=mdp.info.observation_space.shape,
                   output_shape=mdp.info.action_space.shape)


    mu = np.zeros(3)
    sigma = 2e-0*np.ones(3)
    policy = DeterministicPolicy(approximator)
    dist = GaussianDiagonalDistribution(mu, sigma)

    agent = REPS(dist, policy, mdp.info, 0.1)


    # Train
    dataset_callback = CollectDataset()
    core = Core(agent, mdp, callbacks=[dataset_callback])

    for i in range(n_epochs):
        core.learn(n_episodes=n_episodes,
                   n_steps_per_fit=1, render=False)
        J = compute_J(dataset_callback.get(), gamma=mdp.info.gamma)
        dataset_callback.clean()
        print('Reward at iteration ' + str(i) + ': ' +
              str(np.sum(J)/n_episodes))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)

if __name__ == '__main__':
    n_epochs = 24
    n_episodes = 100
    n_ep_per_fit = 10

    experiment(n_epochs, n_episodes, n_ep_per_fit)
