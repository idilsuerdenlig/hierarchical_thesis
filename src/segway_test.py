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
from library.environments.segway import Segway

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_episodes):
    np.random.seed()

    # MDP
    mdp = Segway()

    # Agent
    approximator = Regressor(LinearApproximator,
                   input_shape=mdp.info.observation_space.shape,
                   output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(approximator)

    mu = -0.3*np.ones(3)
    sigma = 1e-0*np.ones(3)
    dist = GaussianDiagonalDistribution(mu, sigma)

    agent = REPS(dist, policy, mdp.info, 1.0)


    # Train
    dataset_callback = CollectDataset()
    core = Core(agent, mdp, callbacks=[dataset_callback])

    for i in range(n_epochs):
        core.learn(n_episodes=n_episodes,
                   n_episodes_per_fit=n_episodes, render=True)
        J = compute_J(dataset_callback.get(), gamma=1.0)
        dataset_callback.clean()
        print('Reward at iteration ' + str(i) + ': ' +
              str(np.sum(J)/n_episodes))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_steps=5000, render=True)

if __name__ == '__main__':
    n_epochs = 24
    n_episodes = 10

    experiment(n_epochs, n_episodes)
