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
from visualize_ship_steering import visualizeShipSteering
import matplotlib.pyplot as plt
from visualize_control_block import VisualizeControlBlock
from collect_policy_parameter import CollectPolicyParameter
from visualize_policy_params import VisualizePolicyParams
from feature_angle_diff_ship_steering import phi
from basic_operation_block import *
from model_placeholder import PlaceHolder
from mux_block import MuxBlock
from mushroom.algorithms.value.td import QLearning
from mushroom.policy import EpsGreedy


def experiment():
    np.random.seed()


    # Model Block
    mdp = ShipSteering()

    #State Placeholder
    state_ph = PlaceHolder()

    #Reward Placeholder
    reward_ph = PlaceHolder()

    #Mux_Block
    mux_block = MuxBlock(wake_time = 1)

    #FeaturesH
    featuresH = Features(basis_list=[PolynomialBasis()])

    low = [0, 0, -np.pi, -np.pi/12]
    high = [150, 150, np.pi, np.pi/12]
    n_tiles = [5, 5, 36, 5]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 9

    tilings = list()
    offset = (high - low) / (np.array(n_tiles) * n_tilings - n_tilings + 1.)

    for i in xrange(n_tilings):
        x_min = low - (n_tilings - 1 - i) * offset
        x_max = high + i * offset
        x_range = [[x, y] for x, y in zip(x_min, x_max)]
        tilings.append(Tiles(x_range, n_tiles))

    #FeaturesL
    featuresL = Features(tilings=tilings)


    # PolicyH
    epsilon = Parameter(value=.1)
    pi = EpsGreedy(epsilon=epsilon)

    # AgentH
    learning_rate = Parameter(value=.1)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agentH = MDPInfo(observation_space=mdp.info.observation_space, action_space=spaces.Discrete(2), gamma=1, horizon=100)

    agentH = QLearning(pi, mdp_info_agentH, agent_params)

    # Control Block H
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_blockH = ControlBlock(wake_time=10, agent=agentH, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[parameter_callback1])

    # Control Block 2
    dataset_callback = CollectDataset()
    parameter_callback2 = CollectPolicyParameter(pi2)
    control_block2 = ControlBlock(wake_time=1, agent=agent2, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[dataset_callback, parameter_callback2])


    # Algorithm
    blocks = [state_ph, reward_ph, control_blockH, mux_block]
    order = [0, 1, 2, 3, 4, 6, 3]
    state_ph.add_input(control_block2)
    reward_ph.add_input(control_block2)
    control_block1.add_input(state_ph)
    control_block1.add_reward(reward_ph)
    function_block1.add_input(control_block1)
    function_block1.add_input(state_ph)
    function_block2.add_input(function_block1)
    function_block3.add_input(function_block2)
    #function_block3.add_input(reward_ph)
    control_block2.add_input(function_block1)
    control_block2.add_reward(function_block3)
    computational_graph = ComputationalGraph(blocks=blocks, order=order, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    dataset_learn = core.learn(n_episodes=3000)
    # Evaluate
    dataset_eval = core.evaluate(n_episodes=10)

    # Visualize
    low_level_dataset = dataset_callback.get()
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    VisualizePolicyParams(parameter_dataset1, parameter_dataset2)
    #VisualizeControlBlock(low_level_dataset)
    visualizeShipSteering(dataset_learn, range_eps=xrange(2980,2995))
    plt.suptitle('learn')
    visualizeShipSteering(dataset_eval)
    plt.suptitle('evaluate')
    plt.show()

    return

if __name__ == '__main__':
    experiment()
