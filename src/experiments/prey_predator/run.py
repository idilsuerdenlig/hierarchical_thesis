import datetime
from joblib import Parallel, delayed

from mushroom.algorithms.policy_search import *
from mushroom.utils.parameters import AdaptiveParameter

from mushroom.utils.folder import *

from mushroom_hierarchical.agents.Q_lambda_discrete import QLambdaDiscrete
from mushroom_hierarchical.environments.prey_predator import PreyPredator

from hierarchical import *


if __name__ == '__main__':
    n_jobs = -1

    how_many = 1
    ep_per_eval = 40
    n_epochs = 50
    ep_per_epoch = 800
    ep_per_eval = 50

    n_iterations = 16
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
        (GPOMDP, {'learning_rate': AdaptiveParameter(value=1e-3)},
         GPOMDP, {'learning_rate': AdaptiveParameter(value=1e-3)})
    ]

    std_high = 1e-2*np.ones(1)
    std_low = 1e-3*np.ones(2)
    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        agent_h = build_high_level_agent(alg_h, params_h, mdp, std_high)
        agent_l = build_low_level_agent(alg_l, params_l, mdp, std_low)

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

    # GHAVAMZADEH
    params_high = {'learning_rate': Parameter(value=8e-2), 'lambda_coeff': 0.9}
    agent_high = build_high_level_ghavamzadeh(QLambdaDiscrete, params_high, mdp)

    params_low = {'learning_rate': AdaptiveParameter(value=1e-2)}
    agent_cross = build_low_level_ghavamzadeh(GPOMDP, params_low, mdp)
    agent_plus = build_low_level_ghavamzadeh(GPOMDP, params_low, mdp)

    print('ghavamzadeh')
    J = Parallel(n_jobs=n_jobs)(delayed(ghavamzadeh_experiment)
                                (mdp, agent_plus, agent_cross, agent_high,
                                 n_epochs, ep_per_epoch, ep_per_eval,
                                 ep_per_run_low_ghavamzadeh)
                                for _ in range(how_many))
    np.save(subdir + '/ghavamzadeh', J)
