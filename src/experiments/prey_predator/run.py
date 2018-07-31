import datetime
from joblib import Parallel, delayed

from mushroom.algorithms.policy_search import *
from mushroom.algorithms.value import *
from mushroom.utils.parameters import *

from mushroom.utils.folder import *

from mushroom_hierarchical.environments.prey_predator import PreyPredator

from hierarchical import *


if __name__ == '__main__':
    n_jobs = 1

    how_many = 1
    n_epochs = 10
    ep_per_epoch = 1000
    ep_per_eval = 100

    ep_per_fit_low = 10

    mdp = PreyPredator()

    # directory
    name = 'prey_predator'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # HIERARCHICAL
    algs_and_params_hier = [
        (TrueOnlineSARSALambda,
         {'learning_rate': Parameter(value=0.9/3), 'lambda_coeff': 0.95},
         GPOMDP, {'learning_rate': AdaptiveParameter(value=1e-3)})
    ]

    eps = ExponentialDecayParameter(1, 0.25)
    std_low = 1e-1*np.ones(1)
    horizon = 10
    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        agent_h = build_high_level_agent(alg_h, params_h, mdp, eps)
        agent_l = build_low_level_agent(alg_l, params_l, mdp, horizon, std_low)

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        J = Parallel(n_jobs=n_jobs)(delayed(experiment)
                                    (mdp, agent_h, agent_l,
                                     n_epochs,
                                     ep_per_epoch,
                                     ep_per_eval,
                                     ep_per_fit_low)
                                    for _ in range(how_many))
        np.save(subdir + '/H_' + alg_h.__name__ + '_' + alg_l.__name__, J)
