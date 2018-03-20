from library.core.hierarchical_core import HierarchicalCore
from library.blocks.computational_graph import ComputationalGraph
from library.blocks.control_block import ControlBlock
from mushroom.utils import spaces
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import *
from library.utils.callbacks.collect_policy_parameter import CollectPolicyParameter
from library.blocks.functions.feature_angle_diff_ship_steering import phi
from library.blocks.basic_operation_block import *
from library.blocks.model_placeholder import PlaceHolder
from library.utils.pick_last_ep_dataset import pick_last_ep
from library.blocks.reward_accumulator import reward_accumulator_block
from library.blocks.error_accumulator import ErrorAccumulatorBlock
from library.environments.idilshipsteering import ShipSteering
from mushroom.environments import MDPInfo
import datetime
from joblib import Parallel, delayed

import argparse
from mushroom.utils.folder import *
from library.blocks.functions.lqr_cost import lqr_cost


def server_experiment_small(i, subdir):

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
    function_block1 = fBlock(name='f1 (angle difference)',phi=phi)

    # Function Block 2
    function_block2 = fBlock(name='f2 (lqr cost)', phi=lqr_cost)

    # Function Block 3
    function_block3 = addBlock(name='f3 (summation)')


    #Features
    features = Features(basis_list=[PolynomialBasis()])

    # Policy 1
    sigma1 = np.array([38, 38])
    approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
    approximator1.set_weights(np.array([75, 75]))

    pi1 = MultivariateDiagonalGaussianPolicy(mu=approximator1,sigma=sigma1)


    # Policy 2
    sigma2 = Parameter(value=.005)
    approximator2 = Regressor(LinearApproximator, input_shape=(1,), output_shape=mdp.info.action_space.shape)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma2)

    # Agent 1
    learning_rate1 = AdaptiveParameter(value=10)
    lim = 150
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space,
                              action_space=spaces.Box(0, lim, (2,)), gamma=mdp.info.gamma, horizon=100)
    agent1 = GPOMDP(policy=pi1, mdp_info=mdp_info_agent1, learning_rate=learning_rate1, features=features)

    # Agent 2
    learning_rate2 = AdaptiveParameter(value=5e-4)
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi, np.pi, (1,)),
                              action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=300)
    agent2 = GPOMDP(policy=pi2, mdp_info=mdp_info_agent2, learning_rate=learning_rate2)

    # Control Block 1
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_block1 = ControlBlock(name='Control Block 1', agent=agent1, n_eps_per_fit=10,
                                  callbacks=[parameter_callback1])

    # Control Block 2
    parameter_callback2 = CollectPolicyParameter(pi2)
    control_block2 = ControlBlock(name='Control Block 2', agent=agent2, n_eps_per_fit=100,
                                  callbacks=[parameter_callback2])


    #Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp_info_agent1.gamma, name='reward_acc')


    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block1, control_block2,
              function_block1, function_block2, function_block3, reward_acc]

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
    function_block3.add_input(function_block1)
    function_block3.add_input(function_block2)
    function_block3.add_input(reward_ph)
    control_block2.add_input(function_block1)
    control_block2.add_reward(function_block3)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    dataset_eval_visual = list()
    low_level_dataset_eval = list()
    dataset_eval = list()

    n_runs = 10
    for n in range(n_runs):
        print('ITERATION', n)
        core.learn(n_episodes=1000, skip=True)
        dataset_eval_run = core.evaluate(n_episodes=10)
        last_ep_dataset = pick_last_ep(dataset_eval_run)
        dataset_eval_visual += last_ep_dataset
        dataset_eval += dataset_eval_run
        low_level_dataset_eval += control_block2.dataset.get()

    # Save
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    mk_dir_recursive('./' + subdir + str(i))

    np.save(subdir+str(i)+'/low_level_dataset_file', low_level_dataset_eval)
    np.save(subdir+str(i)+'/parameter_dataset1_file', parameter_dataset1)
    np.save(subdir+str(i)+'/parameter_dataset2_file', parameter_dataset2)
    np.save(subdir+str(i)+'/dataset_eval_visual_file', dataset_eval_visual)
    np.save(subdir + str(i) + '/dataset_eval_file', dataset_eval)

    del low_level_dataset_eval
    del parameter_dataset1
    del parameter_dataset2
    del dataset_eval_visual
    del dataset_eval

    return

if __name__ == '__main__':
    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_small/'
    n_experiment = 50
    Js = Parallel(n_jobs=-1)(delayed(server_experiment_small)(i=i, subdir=subdir) for i in range(n_experiment))
