import numpy as np

from mushroom.core import Core
from mushroom.algorithms.policy_search import *
from mushroom.policy import GaussianPolicy
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.dataset import compute_J
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.parameters import Parameter
from mushroom.features.basis import PolynomialBasis
from mushroom.features import Features
from mushroom_hierarchical.environments.prey_predator import PreyPredator

from keyboard_agent import KeyboardAgent

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, ep_per_epoch_train, ep_per_epoch_eval, n_iterations):
    np.random.seed()

    # MDP
    mdp = PreyPredator()

    basis = PolynomialBasis.generate(1, mdp.info.observation_space.shape[0])
    phi = Features(basis_list=basis[1:])


    # Features
    approximator = Regressor(LinearApproximator,
                   input_shape=(phi.size,),
                   output_shape=mdp.info.action_space.shape)

    sigma = 1e-2*np.eye(mdp.info.action_space.shape[0])
    policy = GaussianPolicy(approximator, sigma)

    lr = Parameter(1e-5)
    #agent = GPOMDP(policy, mdp.info, lr, phi)
    agent = KeyboardAgent()


    # Train
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=ep_per_epoch_eval, render=True)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    print('Reward at start: ', np.mean(J))

    for i in range(n_epochs):
        core.learn(n_episodes=ep_per_epoch_train,
                   n_episodes_per_fit=ep_per_epoch_train//n_iterations,
                   render=False)
        dataset = core.evaluate(n_episodes=ep_per_epoch_eval, render=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)

        p = policy.get_weights()

        print('mu:    ', p)
        print('Reward at iteration ', i, ': ', np.mean(J))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    n_epochs = 20
    ep_per_epoch_train = 100
    ep_per_epoch_eval = 1
    n_iterations = 4

    experiment(n_epochs=20,
               ep_per_epoch_train=100,
               ep_per_epoch_eval=1,
               n_iterations=4)
