from library.core.hierarchical_core import HierarchicalCore
from library.blocks.computational_graph import ComputationalGraph
from library.blocks.control_block import ControlBlock
from library.blocks.error_accumulator import ErrorAccumulatorBlock
from mushroom.utils import spaces
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import *
from library.utils.callbacks.collect_policy_parameter import CollectPolicyParameter
from library.blocks.functions.feature_angle_diff_ship_steering import *
from library.blocks.basic_operation_block import *
from library.blocks.model_placeholder import PlaceHolder
from library.blocks.reward_accumulator import reward_accumulator_block
from library.environments.idilshipsteering import ShipSteering
from mushroom.environments import MDPInfo
import datetime
from joblib import Parallel, delayed
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from library.blocks.functions.cost_cosine import cost_cosine
from mushroom.distributions import GaussianDiagonalDistribution
from library.policy.deterministic_control_policy import DeterministicControlPolicy
from library.utils.callbacks.collect_distribution_parameter import CollectDistributionParameter



def server_experiment_small(alg_high, alg_low, params, subdir, i):

    np.random.seed()

    # Model Block
    mdp = ShipSteering(small=True, hard=True, n_steps_action=3)

    #State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    #Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    #Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # Function Block 1
    function_block1 = fBlock(name='f1 (angle difference)',phi=pos_ref_angle_difference)

    # Function Block 2
    function_block2 = fBlock(name='f2 (cost cosine)', phi=cost_cosine)

    #Error Acc
    error_acc = ErrorAccumulatorBlock(name='error acc')


    #Features
    features = Features(basis_list=[PolynomialBasis()])

    # Policy 1
    sigma1 = np.array([40, 40])
    approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
    approximator1.set_weights(np.array([75, 75]))

    pi1 = MultivariateDiagonalGaussianPolicy(mu=approximator1,std=sigma1)


    # Policy 2
    pi2 = DeterministicControlPolicy(weights=np.array([0, 0]))
    mu2 = np.zeros(pi2.weights_size)
    sigma2 = 1e-3 * np.ones(pi2.weights_size)
    distribution2 = GaussianDiagonalDistribution(mu2, sigma2)

    # Agent 1
    learning_rate1 = params.get('learning_rate_high')
    lim = 150
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space,
                              action_space=spaces.Box(0, lim, (2,)), gamma=mdp.info.gamma, horizon=100)
    agent1 = alg_high(policy=pi1, mdp_info=mdp_info_agent1, learning_rate=learning_rate1, features=features)

    # Agent 2
    learning_rate2 = params.get('learning_rate_low')
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi*2, np.pi*2, (2,)),
                              action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=100)
    agent2 = alg_low(distribution=distribution2, policy=pi2,
                     mdp_info=mdp_info_agent2, learning_rate=learning_rate2)

    # Control Block 1
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_block1 = ControlBlock(name='Control Block 1', agent=agent1, n_eps_per_fit=ep_per_run,
                                  callbacks=[parameter_callback1])

    # Control Block 2
    parameter_callback2 = CollectDistributionParameter(distribution2)
    control_block2 = ControlBlock(name='Control Block 2', agent=agent2, n_eps_per_fit=10,
                                  callbacks=[parameter_callback2])


    #Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp_info_agent1.gamma, name='reward_acc')


    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block1, control_block2,
              function_block1, function_block2, reward_acc, error_acc]

    state_ph.add_input(control_block2)
    reward_ph.add_input(control_block2)
    lastaction_ph.add_input(control_block2)
    control_block1.add_input(state_ph)
    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block2)
    control_block1.add_reward(reward_acc)
    control_block1.add_alarm_connection(control_block2)
    function_block1.add_input(control_block1)
    function_block1.add_input(state_ph)
    function_block2.add_input(function_block1)
    error_acc.add_input(function_block1)
    control_block2.add_input(function_block1)
    control_block2.add_input(error_acc)
    control_block2.add_reward(function_block2)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    low_level_dataset_eval = list()
    dataset_eval = list()

    dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    dataset_eval += dataset_eval_run

    for n in range(n_runs):
        print('ITERATION', n)
        core.learn(n_episodes=n_iterations*ep_per_run, skip=True)
        dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
        dataset_eval += dataset_eval_run
        J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
        print('J at iteration ' + str(n) + ': ' + str(np.mean(J)))
        low_level_dataset_eval += control_block2.dataset.get()

    # Save
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    mk_dir_recursive('./' + subdir + str(i))

    np.save(subdir+str(i)+'/low_level_dataset_file', low_level_dataset_eval)
    np.save(subdir+str(i)+'/parameter_dataset1_file', parameter_dataset1)
    np.save(subdir+str(i)+'/parameter_dataset2_file', parameter_dataset2)
    np.save(subdir+str(i)+'/dataset_eval_file', dataset_eval)


    return


if __name__ == '__main__':

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_small_hierarchical/'
    alg_high = GPOMDP
    alg_low = PGPE
    learning_rate_high = AdaptiveParameter(value=10)
    learning_rate_low = AdaptiveParameter(value=5e-4)
    how_many = 100
    n_runs = 25
    n_iterations = 10
    ep_per_run = 20
    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, 'latest')


    params = {'learning_rate_high': learning_rate_high, 'learning_rate_low': learning_rate_low}
    np.save(subdir + '/algorithm_params_dictionary', params)
    experiment_params = {'how_many': how_many, 'n_runs': n_runs,
                         'n_iterations': n_iterations, 'ep_per_run': ep_per_run}
    np.save(subdir + '/experiment_params_dictionary', experiment_params)
    Js = Parallel(n_jobs=-1)(delayed(server_experiment_small)(alg_high, alg_low, params,
                                                subdir, i) for i in range(how_many))
