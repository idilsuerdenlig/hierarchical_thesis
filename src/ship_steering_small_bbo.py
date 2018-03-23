import numpy as np

from mushroom.algorithms.policy_search import RWR, PGPE, REPS
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core import Core
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.features.tiles import Tiles
from mushroom.features.features import Features
from mushroom.policy import DeterministicPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import AdaptiveParameter
from library.environments.idilshipsteering import ShipSteering
import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from mushroom.utils.folder import *

tqdm.monitor_interval = 0


def experiment(alg, params, subdir, exp_no):
    np.random.seed()

    # MDP
    mdp = ShipSteering(small=True, hard=True, n_steps_action=3)

    high = [15, 15, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 6]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 9

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high)
    phi = Features(tilings=tilings)

    input_shape = (phi.size,)

    approximator_params = dict(input_dim=input_shape)
    approximator = Regressor(LinearApproximator,
                             input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 4e-2 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # Agent
    agent = alg(distribution, policy, mdp.info, features=phi, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    dataset_eval_all = list()
    for n in range(n_runs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(n) + ': ' + str(np.mean(J)))
        dataset_eval_all += dataset_eval

    mk_dir_recursive('./' + subdir + str(exp_no))
    np.save(subdir+str(exp_no)+'/dataset_eval_file', dataset_eval_all)


if __name__ == '__main__':
    how_many = 1
    n_jobs = 1
    n_runs = 10
    n_iterations = 10
    ep_per_run = 10

    algs = [REPS, RWR, PGPE]
    params = [{'eps': 0.5},
              {'beta': 1},
              {'learning_rate': AdaptiveParameter(value=0.05)}]

    for alg, params in zip(algs, params):
        subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + \
                 '_small_bbo/' + alg.__name__ + '/'
        mk_dir_recursive('./' + subdir)

        np.save(subdir + '/algorithm_params_dictionary', params)
        experiment_params = {'how_many': how_many, 'n_runs': n_runs,
                             'n_iterations': n_iterations,
                             'ep_per_run': ep_per_run}
        np.save(subdir + '/experiment_params_dictionary', experiment_params)
        print('---------------------------------------------------------------')
        print(alg.__name__)
        Parallel(n_jobs=n_jobs)(delayed(experiment)(alg, params, subdir, i)
                                for i in range(how_many))
