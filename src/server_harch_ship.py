from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from control_block import ControlBlock
from mushroom.utils import spaces
from mushroom.environments import *
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import *
from collect_policy_parameter import CollectPolicyParameter
from feature_angle_diff_ship_steering import phi
from basic_operation_block import *
from model_placeholder import PlaceHolder
from pick_last_ep_dataset import pick_last_ep
from reward_accumulator import reward_accumulator_block
from error_accumulator import ErrorAccumulatorBlock
import datetime
import argparse
from mushroom.utils.folder import mk_dir_recursive
from lqr_cost import lqr_cost


def experiment():

    parser = argparse.ArgumentParser(description='server_harch_ship')
    parser.add_argument("--small", help="environment size small or big", action="store_true")
    args = parser.parse_args()
    small = args.small
    print 'SMALL IS', small

    np.random.seed()

    # Model Block
    mdp = ShipSteering(small)

    #State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    #Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    #Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # Function Block 1
    function_block1 = fBlock(name='f1 (angle difference)',phi=phi)

    # Function Block 2
    function_block2 = fBlock(name='f2 (lqr cost)', phi=lqr_cost)

    # Function Block 3
    function_block3 = addBlock(name='f3 (summation)')


    #Features
    features = Features(basis_list=[PolynomialBasis()])

    # Policy 1
    if small:
        sigma1 = np.array([0.01, 0.01])
        approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
        approximator1.set_weights(np.array([110, 110]))
    else:
        sigma1 = np.array([250, 250])
        approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
        approximator1.set_weights(np.array([500, 500]))

    pi1 = MultivariateDiagonalGaussianPolicy(mu=approximator1,sigma=sigma1)

    # Policy 2
    sigma2 = Parameter(value=.005)
    approximator2 = Regressor(LinearApproximator, input_shape=(1,), output_shape=mdp.info.action_space.shape)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma2)

    # Agent 1
    if small:
        learning_rate = Parameter(value=0)
    else:
        learning_rate = AdaptiveParameter(value=65)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}

    lim = 150 if small else 1000
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space,
                              action_space=spaces.Box(0,lim,(2,)), gamma=mdp.info.gamma, horizon=100)
    agent1 = GPOMDP(policy=pi1, mdp_info=mdp_info_agent1, params=agent_params, features=features)

    # Agent 2
    learning_rate = AdaptiveParameter(value=1e-4)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi, np.pi, (1,)),
                              action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=100)
    agent2 = GPOMDP(policy=pi2, mdp_info=mdp_info_agent2, params=agent_params, features=None)

    # Control Block 1
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_block1 = ControlBlock(name='Control Block 1', agent=agent1, n_eps_per_fit=10,
                                  callbacks=[parameter_callback1])

    # Control Block 2
    dataset_callback = CollectDataset()
    parameter_callback2 = CollectPolicyParameter(pi2)
    control_block2 = ControlBlock(name='Control Block 2', agent=agent2, n_eps_per_fit=20,
                                  callbacks=[dataset_callback, parameter_callback2])


    #Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp_info_agent1.gamma, name='reward_acc')

    #Error Accumulator
    #err_acc = ErrorAccumulatorBlock(name='err_acc')

    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block1, control_block2,
              function_block1, function_block2, function_block3, reward_acc]

    state_ph.add_input(control_block2)
    reward_ph.add_input(control_block2)
    lastaction_ph.add_input(control_block2)
    control_block1.add_input(state_ph)
    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block2)
    #err_acc.add_input(function_block1)
    #err_acc.add_alarm_connection(control_block2)
    control_block1.add_reward(reward_acc)
    control_block1.add_alarm_connection(control_block2)
    function_block1.add_input(control_block1)
    function_block1.add_input(state_ph)
    function_block2.add_input(function_block1)
    function_block2.add_input(lastaction_ph)
    function_block3.add_input(function_block1)
    function_block3.add_input(function_block2)
    function_block3.add_input(reward_ph)
    control_block2.add_input(function_block1)
    #control_block2.add_input(err_acc)
    control_block2.add_reward(function_block3)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    dataset_learn_visual = list()

    n_eps = 5 if small else 20
    for n in xrange(n_eps):
        print n
        dataset_learn = core.learn(n_episodes=1000)
        last_ep_dataset = pick_last_ep(dataset_learn)
        dataset_learn_visual += last_ep_dataset
        del dataset_learn

    # Evaluate
    dataset_eval = core.evaluate(n_episodes=10)

    # Save
    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

    low_level_dataset = dataset_callback.get()
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    mk_dir_recursive('./' + subdir)

    np.save(subdir+'/low_level_dataset_file', low_level_dataset)
    np.save(subdir+'/parameter_dataset1_file', parameter_dataset1)
    np.save(subdir+'/parameter_dataset2_file', parameter_dataset2)
    np.save(subdir+'/dataset_learn_visual_file', dataset_learn_visual)
    np.save(subdir+'/dataset_eval_file', dataset_eval)



    return

if __name__ == '__main__':
    experiment()
