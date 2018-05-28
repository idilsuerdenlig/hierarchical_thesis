import datetime
from joblib import Parallel, delayed

from mushroom.algorithms.policy_search import *
from mushroom.utils.parameters import AdaptiveParameter
from mushroom.utils.folder import *

from mushroom_hierarchical.environments.ship_steering_multiple_gates import *

from hierarchical import *


if __name__ == '__main__':

    n_jobs = 1

    how_many = 1#00
    ep_per_eval = 40
    n_epochs = 50
    ep_per_epoch = 800
    ep_per_eval = 50

    n_iterations_hier = 10

    ep_per_run_low = 10

    # MDP
    mdp = ShipSteeringMultiGate(n_steps_action=3)

    # directory
    name = 'multigate_ship_steering'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # Hierarchical
    algs_and_params_hier = [
        (GPOMDP, {'learning_rate': AdaptiveParameter(value=50)},
         PGPE, {'learning_rate': AdaptiveParameter(value=5e-4)})
    ]

    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        mu = 500
        sigma = 250
        agent_h = build_high_level_agent(alg_h, params_h, mdp, mu, sigma)
        agent_l = build_low_level_agent(alg_l, params_l, mdp)

        ep_per_run_hier = ep_per_epoch // n_iterations_hier

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        J = Parallel(n_jobs=n_jobs)(delayed(hierarchical_experiment)
                                    (mdp, agent_l, agent_h,
                                     n_epochs, n_iterations_hier,
                                     ep_per_run_hier, ep_per_eval,
                                     ep_per_run_low)
                                    for _ in range(how_many))
        np.save(subdir + '/H_' + alg_h.__name__ + '_' + alg_l.__name__, J)