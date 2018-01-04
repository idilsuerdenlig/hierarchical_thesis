import numpy as np

from mushroom.algorithms.policy_search import REINFORCE
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core.core import Core
from mushroom.environments import ShipSteering
from mushroom.features.tiles import Tiles
from mushroom.features.features import Features
from mushroom.policy import GaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from visualize_ship_steering import visualizeShipSteering



def experiment(n_iterations, n_runs, ep_per_run):
    np.random.seed()

    # MDP
    mdp = ShipSteering()

    # Policy
    tile = Tiles(x_range=[[0., 150.],
                           [0., 150.],
                           [-np.pi, np.pi],
                           [-np.pi / 12, np.pi / 12]],
                  n_tiles=[20, 20, 36, 5])

    theta = Features(tilings=tile)
    #phi = Features(basis_list=basis)

    input_shape = (theta.size,)
    shape = input_shape + mdp.action_space.shape

    approximator_params = dict(params_shape=shape)
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.action_space.shape,
                             params=approximator_params)
    sigma = Parameter(value=.05)
    policy = GaussianPolicy(mu=approximator, sigma=sigma)

    # Agent
    #learning_rate = Parameter(value=.001)
    learning_rate = AdaptiveParameter(value=.01)

    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = REINFORCE(policy, mdp.gamma, agent_params, theta)

    # Train
    core = Core(agent, mdp)
    for i in xrange(n_runs):
        core.learn(n_iterations=n_iterations, how_many=ep_per_run,
                   n_fit_steps=1, iterate_over='episodes')
        dataset_eval = core.evaluate(how_many=ep_per_run,
                                     iterate_over='episodes')
        J = compute_J(dataset_eval, gamma=mdp.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))
        visualizeShipSteering(datalist_eval=dataset_eval, J=J)


    np.save('ship_steering.npy', dataset_eval)


if __name__ == '__main__':
    experiment(n_iterations=40, n_runs=10, ep_per_run=100)
