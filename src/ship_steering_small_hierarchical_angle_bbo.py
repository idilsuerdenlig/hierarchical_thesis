from library.core.hierarchical_core import HierarchicalCore
from library.blocks.computational_graph import ComputationalGraph
from library.blocks.control_block import ControlBlock
from mushroom.features.tiles import Tiles
from mushroom.utils import spaces
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from library.agents.dummy_agent import SimpleAgent
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.policy import DeterministicPolicy
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import *
from library.utils.callbacks.collect_distribution_parameter import CollectDistributionParameter
from library.blocks.functions.feature_angle_diff_ship_steering import angle_ref_angle_difference
from library.blocks.basic_operation_block import *
from library.blocks.model_placeholder import PlaceHolder
from library.blocks.reward_accumulator import reward_accumulator_block
from library.environments.idilshipsteering import ShipSteering
from mushroom.environments import MDPInfo
from mushroom.algorithms.policy_search import RWR, PGPE, REPS
import datetime
from mushroom.features.basis import *
from joblib import Parallel, delayed
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from library.blocks.functions.lqr_cost import lqr_cost
from library.blocks.functions.cost_cosine import cost_cosine



def experiment(alg_high, alg_low, params_high, params_low, subdir, i):

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
    function_block1 = fBlock(name='f1 (angle difference)',
                             phi=angle_ref_angle_difference)

    # Function Block 2
    function_block2 = fBlock(name='f2 (lqr cost)', phi=cost_cosine)

    #Features
    features = Features(basis_list=[PolynomialBasis()])

    # Policy 1
    approximator1 = Regressor(LinearApproximator, input_shape=(features.size,),
                              output_shape=(1,))
    policy1 = DeterministicPolicy(mu=approximator1)
    sigma1 = 1e-3 * np.ones(policy1.weights_size)
    mu1 = np.zeros(policy1.weights_size)
    distribution1 = GaussianDiagonalDistribution(mu1, sigma1)

    # Agent 1
    high = [150, 150, np.pi, np.pi/12]
    low = [0, 0, -np.pi, -np.pi/12]
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space,
                              action_space=spaces.Box(low[2], high[2], (1,)),
                              gamma=mdp.info.gamma, horizon=100)
    agent1 = alg_high(distribution=distribution1,
                      policy=policy1,
                      mdp_info=mdp_info_agent1,
                      features=features, **params_high)


    # Policy 2
    approximator2 = Regressor(LinearApproximator, input_shape=(1,),
                              output_shape=mdp.info.action_space.shape)
    policy2 = DeterministicPolicy(mu=approximator2)
    mu2 = np.zeros(policy2.weights_size)
    sigma2 = 1e-3 * np.ones(policy2.weights_size)
    distribution2 = GaussianDiagonalDistribution(mu2, sigma2)

    # Agent 2
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(low[2], high[2], (1,)),
                              action_space=mdp.info.action_space,
                              gamma=mdp.info.gamma, horizon=100)
    agent2 = alg_low(distribution=distribution2,
                     policy=policy2,
                     mdp_info=mdp_info_agent2, **params_low)

    # Control Block 1
    parameter_callback1 = CollectDistributionParameter(distribution1)
    control_block1 = ControlBlock(name='Control Block 1', agent=agent1,
                                  n_eps_per_fit=ep_per_run,
                                  callbacks=[parameter_callback1])

    # Control Block 2
    parameter_callback2 = CollectDistributionParameter(distribution2)
    control_block2 = ControlBlock(name='Control Block 2', agent=agent2,
                                  n_eps_per_fit=10,
                                  callbacks=[parameter_callback2])


    #Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp_info_agent1.gamma,
                                          name='reward_acc')


    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block1, control_block2,
              function_block1, function_block2, reward_acc]

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
    function_block2.add_input(lastaction_ph)
    control_block2.add_input(function_block1)
    control_block2.add_reward(function_block2)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    low_level_dataset_eval = list()
    dataset_eval = list()

    dataset_eval_run = core.evaluate(n_episodes=ep_per_run, quiet=True)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    dataset_eval += dataset_eval_run

    for n in range(n_runs):
        print('ITERATION', n)
        core.learn(n_episodes=n_iterations*ep_per_run, skip=True, quiet=True)
        dataset_eval_run = core.evaluate(n_episodes=ep_per_run, quiet=True)
        dataset_eval += dataset_eval_run
        J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
        #print(parameter_callback1.get_values())
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

    how_many = 1
    n_runs = 25
    n_iterations = 10
    ep_per_run = 20
    alg_high = PGPE
    alg_low = PGPE

    learning_rate_high = AdaptiveParameter(value=1e-3)
    learning_rate_low = AdaptiveParameter(value=1e-5)

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_small_hierarchical_angle_bbo/'

    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, 'latest')

    params_high = {'learning_rate': learning_rate_high}
    params_low = {'learning_rate': learning_rate_low}
    np.save(subdir + '/algorithm_params_high_dictionary', params_high)
    np.save(subdir + '/algorithm_params_low_dictionary', params_low)
    experiment_params = {'how_many': how_many, 'n_runs': n_runs,
                         'n_iterations': n_iterations, 'ep_per_run': ep_per_run}
    np.save(subdir + '/experiment_params_dictionary', experiment_params)
    print('---------------------------------------------------------------')
    print(alg_low.__name__)
    Js = Parallel(n_jobs=1)(delayed(experiment)(alg_high, alg_low, params_high,
                                                params_low, subdir, i)
                            for i in range(how_many))

