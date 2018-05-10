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
    n_tilings = 10

    tilings = Tiles.generate(n_tilings - 1, [15, 15, 15],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high + 1e-3)

    phi = Features(tilings=tilings)


    tilings_v = tilings + Tiles.generate(1, [1, 1, 1],
                                         mdp.info.observation_space.low,
                                         mdp.info.observation_space.high + 1e-3)

    psi = Features(tilings=tilings_v)

    mu = Regressor(LinearApproximator,
                   input_shape=(phi.size,),
                   #input_shape=mdp.info.observation_space.shape,
                   output_shape=mdp.info.action_space.shape)

    sigma = 2e-1*np.eye(1)
    policy = GaussianPolicy(mu, sigma)

    alpha_theta = Parameter(5e-7)
    alpha_omega = Parameter(5e-1/n_tilings)
    alpha_v = Parameter(5e-1/n_tilings)
    agent = COPDAC_Q(policy, mu, mdp.info,
                 alpha_theta, alpha_omega, alpha_v,
                 value_function_features=psi,
                 policy_features=phi)


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
