import datetime
from joblib import Parallel, delayed

from mushroom.algorithms.policy_search import *
from mushroom.algorithms.value.td import *
from mushroom.utils.parameters import AdaptiveParameter, Parameter
from mushroom.utils.folder import *

from mushroom_hierarchical.environments.ship_steering_multiple_gates import *

from hierarchical import *


if __name__ == '__main__':

    n_jobs = 1

    how_many = 1#00
    n_epochs = 50
    ep_per_epoch_train = 100
    ep_per_epoch_eval = 10
    n_iterations = 10

    ep_per_fit_low = 10
    ep_per_fit_mid = 20

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
        (QLearning, {'learning_rate': Parameter(value=10)},
        GPOMDP, {'learning_rate': AdaptiveParameter(value=50)},
        GPOMDP, {'learning_rate': AdaptiveParameter(value=50)},
        GPOMDP, {'learning_rate': AdaptiveParameter(value=50)},
        GPOMDP, {'learning_rate': AdaptiveParameter(value=50)},
        PGPE, {'learning_rate': AdaptiveParameter(value=5e-4)})
         ]

    for alg_h, params_h, alg_m1, params_m1, alg_m2, params_m2,\
        alg_m3, params_m3, alg_m4, params_m4, alg_l, params_l in algs_and_params_hier:

        epsilon = Parameter(value=0.15)
        agent_h = build_high_level_agent(alg_h, params_h, mdp, epsilon)

        mu1 = 500; mu2 = 500; mu3 = 500; mu4 = 500
        sigma1 = 250; sigma2 = 250; sigma3 = 250; sigma4 = 250
        agent_m1 = build_mid_level_agent(alg_m1, params_m1, mdp, mu1, sigma1)
        agent_m2 = build_mid_level_agent(alg_m2, params_m2, mdp, mu2, sigma2)
        agent_m3 = build_mid_level_agent(alg_m3, params_m3, mdp, mu3, sigma3)
        agent_m4 = build_mid_level_agent(alg_m4, params_m4, mdp, mu4, sigma4)

        agent_l = build_low_level_agent(alg_l, params_l, mdp)

        ep_per_run_hier = ep_per_epoch_train // n_iterations

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        J = Parallel(n_jobs=n_jobs)(delayed(hierarchical_experiment)
                                    (mdp, agent_l, agent_m1,
                            agent_m2, agent_m3, agent_m4,
                            agent_h, n_epochs,
                            n_iterations, ep_per_epoch_train,
                            ep_per_epoch_eval, ep_per_fit_low, ep_per_fit_mid)
                                    for _ in range(how_many))
        np.save(subdir + '/H_' + alg_h.__name__ + '_' + alg_l.__name__, J)