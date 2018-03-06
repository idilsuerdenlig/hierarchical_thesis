import numpy as np
from ship_ghavamzade_diagonal import ShipGhavamzadeDiagonal
from ship_ghavamzade_straight import ShipSteeringStraight
from ghavamzade_agent import GhavamzadeAgent
from mushroom.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core import Core
from mushroom.environments import ShipSteering
from mushroom.features.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy import GaussianPolicy, MultivariateGaussianPolicy, MultivariateDiagonalGaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from tqdm import tqdm
from mushroom.utils.angles_utils import shortest_angular_distance, normalize_angle
from CMAC import CMACApproximator

tqdm.monitor_interval = 0

def experiment(n_runs, n_iterations, ep_per_run):
    np.random.seed()

    # MDP
    mdp = ShipGhavamzadeDiagonal()

    # Policy
    high = [150, 150, np.pi, np.pi / 12]
    low = [0, 0, -np.pi, -np.pi / 12]
    n_tiles = [5, 5, 36, 5]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 9


    tilingsL= Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low, high=high)

    input_shape = mdp.info.observation_space.shape

    approximator_params = dict(tiles=tilingsL, input_dim=input_shape[0])
    approximator = Regressor(CMACApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape,
                             **approximator_params)
    sigma = np.array([[1.3e-2]])
    policy = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

    # Agent
    learning_rate = Parameter(value=1e-3)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = GhavamzadeAgent(policy, mdp.info, agent_params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in xrange(n_runs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_steps_per_fit=1)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))

    np.save('ship_steering_diagonal.npy', dataset_eval)

    np.save('success_per_thousand_eps.npy', mdp.success_per_thousand_ep)

if __name__ == '__main__':

    experiment(n_runs=1, n_iterations=10, ep_per_run=100)

