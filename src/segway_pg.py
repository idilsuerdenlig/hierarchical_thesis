import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.policy import DiagonalGaussianPolicy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.parameters import *
from mushroom.features.basis import PolynomialBasis
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.features.tensors import gaussian_tensor
from library.environments.segway_linear_motion import SegwayLinearMotion

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_iteration, n_ep_per_fit, n_eval_run):
    np.random.seed()

    # MDP
    mdp = SegwayLinearMotion()

    #basis = PolynomialBasis.generate(mdp.info.observation_space.shape[0], 1)

    '''low = mdp.info.observation_space.low
    high = mdp.info.observation_space.high

    tiles = Tiles.generate(1, [10, 10, 10, 10], low, high, uniform=True)
    phi = Features(tilings=tiles)

    basis = gaussian_tensor.generate(4*[5],
                                     [
                                         [-2.0, 2.0],
                                         [-np.pi/2, np.pi/2],
                                         [-15, 15],
                                         [-75, 75]
                                     ])
    phi = Features(tensor_list=basis, name='rbf', input_dim=4)
    
    input_shape=(phi.size,1)
    '''

    phi = None
    input_shape = mdp.info.observation_space.shape


    # Features
    mu = Regressor(LinearApproximator,
                   input_shape=input_shape,
                   output_shape=mdp.info.action_space.shape)

    sigma = 1e-0*np.ones(1)
    policy = DiagonalGaussianPolicy(mu, sigma)

    lr =  Parameter(1e-5)
    agent = GPOMDP(policy, mdp.info, lr, phi)


    # Train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=n_eval_run, render=False)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start ', np.mean(J))

    for i in range(n_epochs):
        core.learn(n_episodes=n_iteration*n_ep_per_fit,
                   n_episodes_per_fit=n_ep_per_fit, render=False)

        p = policy.get_weights()

        dataset_eval = core.evaluate(n_episodes=n_eval_run)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)

        print('mu:    ', p[:mu.weights_size])
        print('sigma: ', p[mu.weights_size:])
        print('J at iteration ', i, ': ', np.mean(J))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    experiment(n_epochs=20,
               n_iteration=10,
               n_ep_per_fit=10,
               n_eval_run=10)

