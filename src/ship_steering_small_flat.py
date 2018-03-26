import numpy as np

from mushroom.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core import Core
from library.environments.idilshipsteering import ShipSteering
from mushroom.features.tiles import Tiles
from mushroom.features.features import Features
from mushroom.features.tensors import gaussian_tensor
from mushroom.policy import GaussianPolicy, MultivariateGaussianPolicy, MultivariateDiagonalGaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from tqdm import tqdm
from library.utils.pick_last_ep_dataset import pick_last_ep
from library.utils.callbacks.collect_policy_parameter import CollectPolicyParameter
from mushroom.utils.folder import *
import datetime
from joblib import Parallel, delayed


"""
This script aims to replicate the experiments on the Ship Steering MDP 
using policy gradient algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, params, experiment_params ,subdir, i):

    np.random.seed()

    # MDP
    mdp = ShipSteering(small=True, hard=True, n_steps_action=3)

    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 6]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 1

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high)

    phi = Features(tilings=tilings)
    input_shape = (phi.size,)

    approximator_params = dict(input_dim=phi.size)
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params)

    sigma = np.array([[1e-4]])
    #std = np.array([1e-2])
    #policy = MultivariateDiagonalGaussianPolicy(mu=approximator, std=std)
    policy = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

    # Agent
    agent = alg(policy, mdp.info, features=phi, **params)

    # Train
    parameter_dataset = CollectPolicyParameter(policy)
    core = Core(agent, mdp, callbacks=[parameter_dataset])


    dataset_eval = list()
    dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
    dataset_eval += dataset_eval_run
    print('J at start : ' + str(np.mean(J)))

    for n in range(n_runs):
        print('ITERATION    :', n)
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
        J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
        print('J at iteration ' + str(n) + ': ' + str(np.mean(J)))
        dataset_eval += dataset_eval_run

    mk_dir_recursive('./' + subdir + str(i))
    np.save(subdir+str(i)+'/dataset_eval_file', dataset_eval)
    np.save(subdir+str(i)+'/parameter_dataset_file', parameter_dataset)




if __name__ == '__main__':

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_small_flat/'
    alg = GPOMDP
    learning_rate = AdaptiveParameter(value=1e-4)
    how_many = 100
    n_runs = 10
    n_iterations = 10
    ep_per_run = 20
    mk_dir_recursive('./' + subdir)
    params = {'learning_rate': learning_rate}
    np.save(subdir + '/algorithm_params_dictionary', params)
    experiment_params = {'how_many': how_many, 'n_runs': n_runs,
                         'n_iterations': n_iterations, 'ep_per_run': ep_per_run}
    np.save(subdir + '/experiment_params_dictionary', experiment_params)

    Js = Parallel(n_jobs=8)(delayed(experiment)(alg=alg, params=params,
                                                experiment_params=experiment_params,
                                                subdir=subdir, i=i) for i in range(how_many))