import datetime
from joblib import Parallel, delayed


from mushroom.algorithms.policy_search import *
from mushroom.algorithms.value import *
from mushroom.utils.parameters import *

from mushroom.utils.folder import *

from mushroom_hierarchical.environments.prey_predator import PreyPredator
from mushroom_hierarchical.utils.parse_joblib import parse_joblib

import torch.optim as optim
from torch.nn import SmoothL1Loss

from hierarchical import *


if __name__ == '__main__':
    n_jobs = -1

    how_many = 1
    n_epochs = 50
    ep_per_epoch = 1000
    ep_per_eval = 500

    ep_per_fit_low = 10

    mdp = PreyPredator()

    # directory
    name = 'prey_predator'
    subdir = name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')\
             + '/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, name + '_latest')

    # HIERARCHICAL
    optimizer = {'class': optim.RMSprop,
                 'params': {'lr': 1e-3,
                            'centered': True}}

    p = dict(
        clip_reward=False,
        initial_replay_size=5000,
        max_replay_size=100000,
        target_update_frequency=200,
        batch_size=500,
        n_approximators=1,
    )

    algs_and_params_hier = [
        (DQN, p, GPOMDP, {'learning_rate': AdaptiveParameter(value=1e-3)})
    ]

    eps = ExponentialDecayParameter(1, -0.2)
    #eps = Parameter(1.0)
    std_low = 1e-1*np.ones(2)
    horizon = 10
    lr = 1e-3
    for alg_h, params_h, alg_l, params_l in algs_and_params_hier:
        agent_h = build_high_level_agent(alg_h, params_h, optimizer,
                                         SmoothL1Loss(), mdp, horizon, eps)
        agent_l = build_low_level_agent(alg_l, params_l, mdp, horizon, std_low)

        print('High: ', alg_h.__name__, ' Low: ', alg_l.__name__)
        res = Parallel(n_jobs=n_jobs)(delayed(experiment)
                                      (mdp, agent_h, agent_l,
                                       n_epochs,
                                       ep_per_epoch,
                                       ep_per_eval,
                                       ep_per_fit_low)
                                      for _ in range(how_many))

        J, L, Jlow = parse_joblib(res)
        np.save(subdir + '/J_H_' + alg_h.__name__ + '_' + alg_l.__name__, J)
        np.save(subdir + '/L_H_' + alg_h.__name__ + '_' + alg_l.__name__, L)
        np.save(subdir + '/Jlow_H_' + alg_h.__name__ + '_' + alg_l.__name__,
                Jlow)
