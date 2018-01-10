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
from visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from visualize_control_block import visualize_control_block
from collect_policy_parameter import CollectPolicyParameter
from visualize_policy_params import visualize_policy_params
from feature_angle_diff_ship_steering import phi
from basic_operation_block import *
from model_placeholder import PlaceHolder
from mushroom.features.tiles import Tiles
from pick_last_ep_dataset import pick_last_ep



def experiment():
    np.random.seed()


    # Model Block
    mdp = ShipSteering()

    #State Placeholder
    state_ph = PlaceHolder()

    #Reward Placeholder
    reward_ph = PlaceHolder()

    # Function Block 1
    function_block1 = fBlock(wake_time=1, phi=phi)

    # Function Block 2
    function_block2 = squarednormBlock(wake_time=1)

    # Function Block 3
    function_block3 = plusBlock(wake_time=1)

    #Tiles

    low = [0, 0]
    high = [150, 150]
    n_tiles = [20, 20]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 1

    tilings = list()
    offset = (high - low) / (np.array(n_tiles) * n_tilings - n_tilings + 1.)

    for i in xrange(n_tilings):
        x_min = low - (n_tilings - 1 - i) * offset
        x_max = high + i * offset
        x_range = [[x, y] for x, y in zip(x_min, x_max)]
        tilings.append(Tiles(x_range, n_tiles))

    #Features
    features = Features(tilings=tilings)

    # Policy 1
    mean_tiles = np.zeros(shape=(2,n_tiles[0]*n_tiles[1]))
    for j in xrange(n_tiles[1]):
        for i in xrange(n_tiles[0]):
            index = i+j*n_tiles[0]
            mean_tiles[0][index] = ((high[0]-low[0])/(2*n_tiles[0])) + j*(high[0]-low[0])/n_tiles[0]
            mean_tiles[1][index] = ((high[1]-low[1])/(2*n_tiles[1])) + i*(high[1]-low[1])/n_tiles[1]

    sigma1 = np.eye(2, 2)*1.9
    approximator1 = Regressor(LinearApproximator, weights=mean_tiles, input_shape=(features.size,), output_shape=(2,))
    approximator1.set_weights(mean_tiles)
    pi1 = MultivariateGaussianPolicy(mu=approximator1,sigma=sigma1)

    # Policy 2
    sigma2 = Parameter(value=.1)
    approximator2 = Regressor(LinearApproximator, input_shape=(1,), output_shape=mdp.info.action_space.shape)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma2)
    #pi2.set_weights(np.array([-0.01]))

    # Agent 1
    learning_rate = AdaptiveParameter(value=0.6)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space, action_space=spaces.Box(0,150,(2,)), gamma=mdp.info.gamma, horizon=100)
    agent1 = GPOMDP(policy=pi1, mdp_info=mdp_info_agent1, params=agent_params, features=features)

    # Agent 2
    learning_rate = AdaptiveParameter(value=.001)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi,np.pi,(1,)), action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=100)
    agent2 = GPOMDP(policy=pi2, mdp_info=mdp_info_agent2, params=agent_params, features=None)

    # Control Block 1
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_block1 = ControlBlock(wake_time=10, agent=agent1, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[parameter_callback1])

    # Control Block 2
    dataset_callback = CollectDataset()
    parameter_callback2 = CollectPolicyParameter(pi2)
    control_block2 = ControlBlock(wake_time=1, agent=agent2, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[dataset_callback, parameter_callback2])


    # Algorithm
    blocks = [state_ph, reward_ph, control_block1, control_block2, function_block1, function_block2, function_block3]
    order = [0, 1, 2, 4, 5, 6, 3]
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
    dataset_learn_visual = list()
    #dataset_learn_visual = core.learn(n_episodes=4000)
    for n in xrange(3):
        dataset_learn = core.learn(n_episodes=1000)
        last_ep_dataset = pick_last_ep(dataset_learn)
        dataset_learn_visual += last_ep_dataset
        del dataset_learn
    # Evaluate
    dataset_eval = core.evaluate(n_episodes=10)

    # Visualize
    print np.reshape(pi1.get_weights(),(2,-1))
    low_level_dataset = dataset_callback.get()
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    visualize_policy_params(parameter_dataset1, parameter_dataset2)
    visualize_control_block(low_level_dataset, ep_count=20)
    visualize_ship_steering(dataset_learn_visual, 'learn')
    visualize_ship_steering(dataset_eval, 'evaluate')
    plt.show()

    return

if __name__ == '__main__':
    experiment()
