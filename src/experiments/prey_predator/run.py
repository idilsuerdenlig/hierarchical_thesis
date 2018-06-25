import datetime
from joblib import Parallel, delayed

from mushroom.algorithms.policy_search import *
from mushroom.utils.parameters import *

from mushroom.utils.folder import *

from mushroom_hierarchical.environments.prey_predator import PreyPredator

from hierarchical import *


if __name__ == '__main__':
    n_jobs = 1

    how_many = 1
    n_epochs = 10
    ep_per_epoch = 800
    ep_per_eval = 50

    n_iterations = 16
    ep_per_fit_low = 10

    n_fit_flat = 16
    ep_per_fit_flat = ep_per_epoch // n_fit_flat

    mdp = PreyPredator()

    # directory
    name = 'prey_predator'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # HIERARCHICAL
    algs_and_params_hier = [
        (GPOMDP, {'learning_rate': AdaptiveParameter(value=1e-3)},
         GPOMDP, {'learning_rate': Parameter(value=1e-2)})
    ]

    std_high = 1e-0*np.ones(1)
    std_low = 1e-1*np.ones(2)
    horizon = 5
    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        agent_h = build_high_level_agent(alg_h, params_h, mdp, std_high)
        agent_l = build_low_level_agent(alg_l, params_l, mdp, horizon, std_low)

        ep_per_fit_high = ep_per_epoch // n_iterations

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        J = Parallel(n_jobs=n_jobs)(delayed(experiment)
                                    (mdp, agent_h, agent_l,
                                     n_epochs,
                                     ep_per_epoch,
                                     ep_per_eval,
                                     ep_per_fit_high,
                                     ep_per_fit_low)
                                    for _ in range(how_many))
        np.save(subdir + '/H_' + alg_h.__name__ + '_' + alg_l.__name__, J)
