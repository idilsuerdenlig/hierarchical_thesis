import datetime
from joblib import Parallel, delayed

from mushroom.utils import spaces
from mushroom.environments import MDPInfo
from mushroom.algorithms.policy_search import *
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from mushroom.policy import DeterministicPolicy

from mushroom.utils.parameters import AdaptiveParameter

from library.core.hierarchical_core import HierarchicalCore
from library.environments.segway_linear_motion import SegwayLinearMotion
from library.blocks.computational_graph import ComputationalGraph
from library.blocks.control_block import ControlBlock
from library.blocks.basic_operation_block import *
from library.blocks.model_placeholder import PlaceHolder
from library.blocks.error_accumulator import ErrorAccumulatorBlock
from library.blocks.reward_accumulator import reward_accumulator_block
from library.blocks.functions.pick_first_state import pick_first_state
from library.blocks.functions.angle_to_angle_diff_complete_state \
    import angle_to_angle_diff_complete_state
from library.blocks.functions.lqr_cost_segway import lqr_cost_segway
from library.utils.callbacks.collect_distribution_parameter import\
    CollectDistributionParameter


def server_experiment_small(alg_high, alg_low, params, subdir, i):

    np.random.seed()

    # Model Block
    mdp = SegwayLinearMotion(goal_pos=1.0)

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

    # Integrator Block
    error_acc = ErrorAccumulatorBlock(name='error acc')


    # Features
    features1 = Features(basis_list=[PolynomialBasis()])
    approximator1 = Regressor(LinearApproximator,
                             input_shape=(features1.size,),
                             output_shape=(1,))

    # Policy 1
    n_weights = approximator1.weights_size
    mu1 = np.zeros(n_weights)
    sigma1 = 2e-0 * np.ones(n_weights)
    pi1 = DeterministicPolicy(approximator1)
    dist1 = GaussianDiagonalDistribution(mu1, sigma1)


    # Agent 1
    eps1 = params.get('eps')
    lim = 2 * np.pi
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space,
                              action_space=spaces.Box(0, lim, (1,)),
                              gamma=mdp.info.gamma,
                              horizon=20)

    agent1 = alg_low(distribution=dist1, policy=pi1, features=features1,
                     mdp_info=mdp_info_agent1, eps=eps1)

    # Policy 2
    basis = PolynomialBasis.generate(1, 3)
    features2 = Features(basis_list=basis)
    approximator2 = Regressor(LinearApproximator,
                              input_shape=(features2.size,),
                              output_shape=(1,))
    n_weights2 = approximator2.weights_size
    mu2 = np.zeros(n_weights2)
    sigma2 = 2e-0 * np.ones(n_weights2)
    pi2 = DeterministicPolicy(approximator2)
    dist2 = GaussianDiagonalDistribution(mu2, sigma2)

    # Agent 2
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(
        low=np.array([-np.pi, -np.pi, -np.pi]),
        high=np.array([np.pi, np.pi, np.pi]),
        shape=(3,)),
        action_space=mdp.info.action_space,
        gamma=mdp.info.gamma, horizon=30)

    agent2 = alg_low(distribution=dist2, policy=pi2, features=features2,
                     mdp_info=mdp_info_agent2, eps=eps1)

    # Control Block 1
    parameter_callback1 = CollectDistributionParameter(dist1)
    control_block1 = ControlBlock(name='Control Block 1', agent=agent1,
                                  n_eps_per_fit=ep_per_run,
                                  callbacks=[parameter_callback1])

    # Control Block 2
    parameter_callback2 = CollectDistributionParameter(dist2)
    control_block2 = ControlBlock(name='Control Block 2', agent=agent2,
                                  n_eps_per_fit=20,
                                  callbacks=[parameter_callback2])


    #Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp_info_agent1.gamma,
                                          name='reward_acc')


    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block1,
              control_block2, function_block1, function_block2,
              function_block3, function_block4, error_acc, reward_acc]

    state_ph.add_input(control_block2)
    reward_ph.add_input(control_block2)
    lastaction_ph.add_input(control_block2)
    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block2)
    control_block1.add_input(function_block1)
    control_block1.add_reward(reward_acc)
    control_block1.add_alarm_connection(control_block2)
    control_block2.add_input(function_block2)
    control_block2.add_reward(function_block4)
    function_block1.add_input(state_ph)
    function_block2.add_input(control_block1)
    function_block2.add_input(state_ph)
    function_block3.add_input(function_block2)
    error_acc.add_input(function_block3)
    error_acc.add_alarm_connection(control_block2)
    function_block4.add_input(function_block3)
    function_block4.add_input(error_acc)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    low_level_dataset_eval = list()
    dataset_eval = list()

    dataset_eval_run = core.evaluate(n_episodes=eval_run)
    J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    dataset_eval += dataset_eval_run

    for n in range(n_runs):
        print('ITERATION', n)
        core.learn(n_episodes=n_iterations*ep_per_run, skip=True)
        dataset_eval_run = core.evaluate(n_episodes=eval_run)
        dataset_eval += dataset_eval_run
        J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
        print('J at iteration ' + str(n) + ': ' + str(np.mean(J)))
        #low_level_dataset_eval += control_block2.dataset.get()

    # Save
    #parameter_dataset1 = parameter_callback1.get_values()
    #parameter_dataset2 = parameter_callback2.get_values()

    #mk_dir_recursive('./' + subdir + str(i))

    #np.save(subdir+str(i)+'/low_level_dataset_file', low_level_dataset_eval)
    #np.save(subdir+str(i)+'/parameter_dataset1_file', parameter_dataset1)
    #np.save(subdir+str(i)+'/parameter_dataset2_file', parameter_dataset2)
    #np.save(subdir+str(i)+'/dataset_eval_file', dataset_eval)


if __name__ == '__main__':

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + \
             '_big_hierarchical/'
    alg_high = REPS
    alg_low = REPS
    learning_rate_high = AdaptiveParameter(value=50)
    learning_rate_low = AdaptiveParameter(value=5e-4)
    eps = 0.01
    n_jobs = 1
    how_many = 1
    n_runs = 10
    n_iterations = 10
    ep_per_run = 20
    eval_run = 10
    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, 'latest')


    params = {'learning_rate_high': learning_rate_high,
              'learning_rate_low': learning_rate_low,
              'eps':eps}
    np.save(subdir + '/algorithm_params_dictionary', params)
    experiment_params = {'how_many': how_many,
                         'n_runs': n_runs,
                         'n_iterations': n_iterations,
                         'ep_per_run': ep_per_run,
                         'eval_run': eval_run}
    np.save(subdir + '/experiment_params_dictionary', experiment_params)
    Js = Parallel(n_jobs=n_jobs)(delayed(server_experiment_small)
                                 (alg_high, alg_low, params, subdir, i)
                                 for i in range(how_many))
