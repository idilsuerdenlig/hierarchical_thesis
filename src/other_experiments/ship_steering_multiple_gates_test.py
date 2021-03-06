from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from control_block import ControlBlock
from mushroom.utils import spaces
from ship_steering_multiple_gates import ShipSteeringMultiGate
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import *
from visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from visualize_control_block import visualize_control_block
from collect_policy_parameter import CollectPolicyParameter
from visualize_policy_params import visualize_policy_params
from feature_angle_diff_ship_steering import phi
from basic_operation_block import *
from model_placeholder import PlaceHolder
from pick_last_ep_dataset import pick_last_ep
from reward_accumulator import reward_accumulator_block
from topological_sort import topological_sort

def experiment():
    np.random.seed()


  # Model Block
    mdp = ShipSteeringMultiGate()

    #State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    #Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Function Block 1
    function_block1 = fBlock(name='f1 (angle difference)',phi=phi)

    # Function Block 2
    function_block2 = squarednormBlock(name='f2 (squared norm)')

    # Function Block 3
    function_block3 = addBlock(name='f3 (summation)')

    #Features
    features = Features(basis_list=[PolynomialBasis()])

    # Policy 1
    sigma1 = np.array([38, 38])
    approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
    approximator1.set_weights(np.array([75, 75]))

    pi1 = DiagonalGaussianPolicy(mu=approximator1,sigma=sigma1)

    # Policy 2
    sigma2 = Parameter(value=.01)
    approximator2 = Regressor(LinearApproximator, input_shape=(1,), output_shape=mdp.info.action_space.shape)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma2)

    # Agent 1
    learning_rate = AdaptiveParameter(value=10)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space,
                              action_space=spaces.Box(0, 150, (2,)), gamma=mdp.info.gamma, horizon=50)
    agent1 = GPOMDP(policy=pi1, mdp_info=mdp_info_agent1, params=agent_params, features=features)

    # Agent 2
    learning_rate = AdaptiveParameter(value=.001)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi,np.pi,(1,)),
                              action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=100)
    agent2 = GPOMDP(policy=pi2, mdp_info=mdp_info_agent2, params=agent_params, features=None)

    # Control Block 1
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_block1 = ControlBlock(name='Control Block 1', agent=agent1, n_eps_per_fit=5,
                                  callbacks=[parameter_callback1])

    # Control Block 2
    dataset_callback = CollectDataset()
    parameter_callback2 = CollectPolicyParameter(pi2)
    control_block2 = ControlBlock(name='Control Block 2', agent=agent2, n_eps_per_fit=10,
                                  callbacks=[dataset_callback, parameter_callback2])


    #Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp_info_agent1.gamma, name='reward_acc')

    # Algorithm
    blocks = [state_ph, reward_ph, control_block1, control_block2, function_block1, function_block2, function_block3, reward_acc]
    #order = [0, 1, 7, 2, 4, 5, 6, 3]
    state_ph.add_input(control_block2)
    reward_ph.add_input(control_block2)
    control_block1.add_input(state_ph)
    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block2)
    control_block1.add_reward(reward_acc)
    control_block1.add_alarm_connection(control_block2)
    function_block1.add_input(control_block1)
    function_block1.add_input(state_ph)
    function_block2.add_input(function_block1)
    function_block3.add_input(function_block2)
    function_block3.add_input(reward_ph)
    control_block2.add_input(function_block1)
    control_block2.add_reward(function_block3)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    #dataset_learn_visual = core.learn(n_episodes=2000)
    dataset_learn_visual = list()
    for n in range(4):
        dataset_learn = core.learn(n_episodes=500)
        last_ep_dataset = pick_last_ep(dataset_learn)
        dataset_learn_visual += last_ep_dataset
        del dataset_learn

    # Evaluate
    dataset_eval = core.evaluate(n_episodes=10)

    # Visualize
    low_level_dataset = dataset_callback.get()
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    visualize_policy_params(parameter_dataset1, parameter_dataset2)
    visualize_control_block(low_level_dataset, ep_count=20)
    visualize_ship_steering(dataset_learn_visual, name='learn', n_gates=4)

    visualize_ship_steering(dataset_eval, 'evaluate', n_gates=4)
    plt.show()

    return

if __name__ == '__main__':
    experiment()
