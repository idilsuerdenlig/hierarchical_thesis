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

def experiment(alg, n_runs, n_iterations, ep_per_run, subdir, i):
    np.random.seed()

    # MDP
    mdp = ShipSteering(small=True, hard=True, n_steps_action=3)

    high = [15, 15, np.pi, np.pi/12]
    low = [0, 0, -np.pi, -np.pi/12]
    n_tiles = [5, 5, 36, 5]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 3

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low, high=high)
    phi = Features(tilings=tilings)


    input_shape = (phi.size,)

    approximator_params = dict(input_dim=phi.size)
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params)

    sigma = np.array([[.05]])
    policy = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

    # Agent
    learning_rate = AdaptiveParameter(value=.01)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = alg(policy, mdp.info, features=phi, **algorithm_params)

    # Train
    parameter_dataset = CollectPolicyParameter(policy)
    core = Core(agent, mdp, callbacks=[parameter_dataset])


    dataset_eval_visual = list()
    dataset_eval = list()


    for n in range(n_runs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
        last_ep_dataset = pick_last_ep(dataset_eval_run)
        dataset_eval_visual += last_ep_dataset
        dataset_eval += dataset_eval_run

    mk_dir_recursive('./' + subdir + str(i))
    np.save(subdir+str(i)+'/dataset_eval_file', dataset_eval)
    np.save(subdir+str(i)+'/parameter_dataset_file', parameter_dataset)
    np.save(subdir+str(i)+'/dataset_eval_visual_file', dataset_eval_visual)

if __name__ == '__main__':

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_small_flat/'
    n_experiment = 1
    Js = Parallel(n_jobs=1)(delayed(experiment)(alg=GPOMDP, n_runs=40, n_iterations=100, ep_per_run=10, subdir=subdir, i=i) for i in range(n_experiment))