from joblib import Parallel, delayed

from mushroom.utils import spaces
from mushroom.environments import MDPInfo
from mushroom.algorithms.policy_search import *
from mushroom.distributions import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.dataset import compute_J
from mushroom.policy import DeterministicPolicy

from mushroom.utils.parameters import Parameter, AdaptiveParameter

from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.environments.segway_linear_motion import SegwayLinearMotion
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.utils.callbacks.collect_distribution_parameter import\
    CollectDistributionParameter
from mushroom_hierarchical.policy.deterministic_control_policy import DeterministicControlPolicy

from experiments.segway.lqr_cost_segway import lqr_cost_segway
from experiments.segway.angle_to_angle_diff_complete_state import *
from experiments.segway.pick_first_state import pick_first_state


def fall_reward(inputs):
    state = inputs[0]
    if abs(state[1]) > np.pi / 2:
        res = -10000.0
    else:
        res = 0.0
    return np.array([res])


def lqr_cost_segway(ins):
    x = np.concatenate(ins)
    Q = np.diag([3.0, 0.1, 0.1])
    J = x.dot(Q).dot(x)
    reward = -J

    return np.array([reward])


def segway_experiment(alg_high, alg_low, params_high, params_low):

    np.random.seed()

    # Model Block
    mdp = SegwayLinearMotion(goal_distance=1.0)

    #State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    #Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    #Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # Function Block 1
    function_block1 = fBlock(name='f1 (pick distance to goal state var)',
                             phi=pick_first_state)

    # Function Block 2
    function_block2 = fBlock(name='f2 (build state)',
                             phi=angle_to_angle_diff_complete_state)

    # Function Block 3
    function_block3 = fBlock(name='f3 (reward low level)',
                             phi=lqr_cost_segway)

    # Function Block 4
    function_block4 = addBlock(name='f4 (add block)')

    # Function Block 5
    function_block5 = fBlock(name='f5 (fall punish low level)', phi=fall_reward)


    # Features
    approximator1 = Regressor(LinearApproximator,
                             input_shape=(1,),
                             output_shape=(1,))

    # Policy H
    n_weights = approximator1.weights_size
    mu1 = np.zeros(n_weights)
    sigma1 = 2.0e-2*np.ones(n_weights)
    pi1 = DeterministicPolicy(approximator1)
    dist1 = GaussianDiagonalDistribution(mu1, sigma1)


    # Agent H
    lim = np.pi/2
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space,
                              action_space=spaces.Box(-lim, lim, (1,)),
                              gamma=mdp.info.gamma,
                              horizon=mdp.info.horizon)
    agent_high = alg_high(dist1, pi1, mdp_info_agent1, **params_high)

    # Policy L
    approximator2 = Regressor(LinearApproximator,
                              input_shape=(3,),
                              output_shape=(1,))
    n_weights2 = approximator2.weights_size
    mu2 = np.zeros(n_weights2)
    sigma2 = 2.0*np.ones(n_weights2)
    pi2 = DeterministicControlPolicy(approximator2)
    dist2 = GaussianDiagonalDistribution(mu2, sigma2)

    # Agent Low
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(
        low=mdp.info.observation_space.low[1:], #FIXME FALSE
        high=mdp.info.observation_space.high[1:], #FIXME FALSE
        shape=(3,)),
        action_space=mdp.info.action_space,
        gamma=mdp.info.gamma, horizon=mdp.info.horizon)

    agent_low = alg_low(dist2, pi2, mdp_info_agent2, **params_low)

    # Control Block 1
    parameter_callback1 = CollectDistributionParameter(dist1)
    control_block1 = ControlBlock(name='Control Block High', agent=agent_high,
                                  n_eps_per_fit=n_ep_per_fit*2,
                                  callbacks=[parameter_callback1])

    # Control Block 2
    parameter_callback2 = CollectDistributionParameter(dist2)
    control_block2 = ControlBlock(name='Control Block Low', agent=agent_low,
                                  n_eps_per_fit=n_ep_per_fit,
                                  callbacks=[parameter_callback2])
    control_block1.set_mask()

    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block1,
              control_block2, function_block1, function_block2,
              function_block3, function_block4, function_block5]

    state_ph.add_input(control_block2)
    reward_ph.add_input(control_block2)
    lastaction_ph.add_input(control_block2)
    control_block1.add_input(function_block1)
    control_block1.add_reward(reward_ph)
    control_block2.add_input(function_block2)
    control_block2.add_reward(function_block4)
    function_block1.add_input(state_ph)
    function_block2.add_input(control_block1)

    function_block2.add_input(state_ph)
    function_block3.add_input(function_block2)
    function_block5.add_input(state_ph)
    function_block4.add_input(function_block3)
    function_block4.add_input(function_block5)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    dataset_eval_run = core.evaluate(n_episodes=eval_run, render=False)
    J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    mask_done = False
    for n in range(n_epochs):
        print('ITERATION', n)

        if n == 2:
            control_block1.unset_mask()
        core.learn(n_episodes=n_iterations*n_ep_per_fit, skip=True)
        dataset_eval_run = core.evaluate(n_episodes=eval_run, render=False)
        J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
        print('J at iteration ' + str(n) + ': ' + str(np.mean(J)))
        print('dist H:', dist1.get_parameters())
        print('dist L mu:', dist2.get_parameters()[:3])
        print('dist L sigma:', dist2.get_parameters()[3:])


if __name__ == '__main__':
    learning_rate_high = Parameter(value=1e-5)
    learning_rate_low = AdaptiveParameter(value=1e-1)
    eps_high = 0.05
    eps_low = 0.05
    beta_high = 0.01
    beta_low = 2e-3

    algs_params = [
            #(REPS, REPS, {'eps': eps_high}, {'eps': eps_low}),
            (RWR, RWR, {'beta': beta_high}, {'beta': beta_low}),
            #(PGPE, PGPE, {'learning_rate': learning_rate_high},
             #{'learning_rate': learning_rate_low}),
        #(PGPE, RWR, {'learning_rate' : learning_rate_high}, {'beta' :
        # beta_low})
        ]

    n_jobs = 1
    how_many = 1
    n_epochs = 10
    n_iterations = 4
    n_ep_per_fit = 25
    eval_run = 20


    experiment_params = {'how_many': how_many,
                         'n_epochs': n_epochs,
                         'n_iterations': n_iterations,
                         'n_ep_per_fit': n_ep_per_fit,
                         'eval_run': eval_run}
    for alg_high, alg_low, params_high, params_low in algs_params:
        Js = Parallel(n_jobs=n_jobs)(delayed(segway_experiment)
                                 (alg_high, alg_low, params_high, params_low)
                                 for i in range(how_many))
