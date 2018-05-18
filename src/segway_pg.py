import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.policy import GaussianPolicy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.parameters import *
from mushroom.features.basis import PolynomialBasis
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from library.environments.segway_linear_motion import SegwayLinearMotion

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_iteration, n_ep_per_fit, n_eval_run):
    np.random.seed()

    # MDP
    mdp = SegwayLinearMotion()

    #basis = PolynomialBasis.generate(mdp.info.observation_space.shape[0], 1)

    low = mdp.info.observation_space.low
    high = mdp.info.observation_space.high
    tiles = Tiles.generate(1, [10, 10, 10, 10], low, high)
    phi = Features(tilings=tiles)


    # Features
    mu = Regressor(LinearApproximator,
                   input_shape=(phi.size,),
                   output_shape=mdp.info.action_space.shape)

    sigma = 1e-2*np.eye(1)
    policy = GaussianPolicy(mu, sigma)

    lr = AdaptiveParameter(1e-1)
    agent = GPOMDP(policy, mdp.info, lr, phi)


    # Train
    dataset_callback = CollectDataset()
    core = Core(agent, mdp)

    for i in range(n_epochs):
        core.learn(n_episodes=n_iteration*n_ep_per_fit,
                   n_episodes_per_fit=n_ep_per_fit, render=False)

        p = policy.get_weights()

        dataset_eval = core.evaluate(n_episodes=n_eval_run)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)

        print('mu:    ', p[:mu.weights_size])
        #print('sigma: ', p[mu.weights_size:])
        print('Reward at iteration ' + str(i) + ': ' +
              str(np.mean(J)))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    experiment(n_epochs=20,
               n_iteration=4,
               n_ep_per_fit=25,
               n_eval_run=10)

