import numpy as np

from mushroom.algorithms.policy_search import RWR, PGPE, REPS
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core import Core
from mushroom.distributions import GaussianCholeskyDistribution
from mushroom.environments import LQR
from mushroom.policy import DeterministicPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import AdaptiveParameter
from library.environments.idilshipsteering import ShipSteering
import datetime
from joblib import Parallel, delayed


from tqdm import tqdm

tqdm.monitor_interval = 0


def experiment(alg, params, n_runs, fit_per_run, ep_per_run, exp_no):
    np.random.seed()

    # MDP
    mdp = ShipSteering(small=True, hard=True, n_steps_action=3)

    approximator_params = dict(input_dim=mdp.info.observation_space.shape)
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-3 * np.eye(policy.weights_size)
    distribution = GaussianCholeskyDistribution(mu, sigma)

    # Agent
    agent = alg(distribution, policy, mdp.info, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    dataset_eval_all = list()
    for i in range(n_runs):
        core.learn(n_episodes=fit_per_run * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))
        dataset_eval_all.append(dataset_eval)


    np.save(subdir+str(exp_no)+'/dataset_eval_file', dataset_eval_all)
    if exp_no is 0:
        np.save(subdir+'/algorithm_params_dictionary', params)
        experiment_params = [{'how_many': how_many}, {'n_runs': n_runs},
                             {'n_iterations': fit_per_run},
                             {'ep_per_run': ep_per_run}]
        np.save(subdir+'/experiment_params_dictionary', experiment_params)

if __name__ == '__main__':
    learning_rate = AdaptiveParameter(value=0.05)
    how_many = 1
    algs = [REPS, RWR, PGPE]
    params = [{'eps': 0.5}, {'beta': 1}, {'learning_rate': learning_rate}]
    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_small_bbo/'

    for alg, params in zip(algs, params):
        print(alg.__name__)
        Js = Parallel(n_jobs=1)(delayed(experiment)(alg, params, subdir, exp_no=i,
                                                    n_runs=4, fit_per_run=10,
                                                    ep_per_run=100)for i in range(how_many))
