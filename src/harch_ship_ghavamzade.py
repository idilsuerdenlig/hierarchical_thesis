from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from control_block import ControlBlock
from mushroom.utils import spaces
from mushroom.environments import *
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from mushroom.utils.callbacks import CollectDataset, CollectQ
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
from mux_block import MuxBlock
from mushroom.algorithms.value.td import *
from mushroom.policy import EpsGreedy
from mushroom.features.tiles import Tiles
from pick_state import pick_state
from rototranslate import rototranslate
from hold_state import hold_state
from hi_lev_extr_rew_ghavamzade import G_high
from low_lev_extr_rew_ghavamzade import G_low
from collect_Q import Collect_Q

def experiment():
    np.random.seed()


    # Model Block
    mdp = ShipSteering()

    #State Placeholder
    state_ph = PlaceHolder()

    #Reward Placeholder
    reward_ph = PlaceHolder()

    #FeaturesH
    tilingsH= Tiles.generate(n_tilings=1, n_tiles=[20,20], low=[0,0], high=[150,150])
    featuresH = Features(tilings=tilingsH)

    # PolicyH
    epsilon = Parameter(value=0.01)
    piH = EpsGreedy(epsilon=epsilon)

    # AgentH
    learning_rate = Parameter(value=0.1)

    mdp_info_agentH = MDPInfo(observation_space=spaces.Box(low=0, high=150, shape=(2,)), action_space=spaces.Discrete(8), gamma=1, horizon=100)
    approximator_params = dict(input_shape=(featuresH.size,),
                               output_shape=mdp_info_agentH.action_space.size,
                               n_actions=mdp_info_agentH.action_space.n)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda': .9}
    fit_params = dict()
    agent_params = {'approximator_params': approximator_params,
                    'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agentH = TrueOnlineSARSALambda(policy=piH, mdp_info=mdp_info_agentH, params=agent_params, features=featuresH)

    # Control Block H
    control_blockH = ControlBlock(wake_time=100, agent=agentH, n_eps_per_fit=None, n_steps_per_fit=1)

    #FeaturesL
    low = [0, 0, -np.pi, -np.pi/12]
    high = [15, 15, np.pi, np.pi/12]
    n_tiles = [5, 5, 36, 5]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 1

    tilingsL= Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low, high=high)
    featuresL = Features(tilings=tilingsL)

    # Policy1
    sigma1 = np.eye(1, 1)*0.1
    approximator1 = Regressor(LinearApproximator, input_shape=(featuresL.size,), output_shape=mdp.info.action_space.shape)
    pi1 = MultivariateGaussianPolicy(mu=approximator1,sigma=sigma1)

    # Policy2
    sigma2 = np.eye(1, 1)*0.1
    approximator2 = Regressor(LinearApproximator, input_shape=(featuresL.size,), output_shape=mdp.info.action_space.shape)
    pi2 = MultivariateGaussianPolicy(mu=approximator2,sigma=sigma2)

    # Agent1
    learning_rate1 = AdaptiveParameter(value=.001)
    algorithm_params1 = dict(learning_rate=learning_rate1)
    fit_params1 = dict()
    agent_params1 = {'algorithm_params': algorithm_params1,
                    'fit_params': fit_params1}
    mdp_info_agent1 = MDPInfo(observation_space=spaces.Box(low=np.array([0,0,-np.pi,-np.pi/12]), high=np.array([15,15,np.pi,np.pi/12])), action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=100)
    agent1 = GPOMDP(policy=pi1, mdp_info=mdp_info_agent1, params=agent_params1, features=featuresL)

    # Agent2
    learning_rate2 = AdaptiveParameter(value=.001)
    algorithm_params2 = dict(learning_rate=learning_rate2)
    fit_params2 = dict()
    agent_params2 = {'algorithm_params': algorithm_params2,
                    'fit_params': fit_params2}
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(low=np.array([0,0,-np.pi,-np.pi/12]), high=np.array([15,15,np.pi,np.pi/12])), action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=100)
    agent2 = GPOMDP(policy=pi2, mdp_info=mdp_info_agent2, params=agent_params2, features=featuresL)

    # Control Block +
    dataset_callback1 = CollectDataset()
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_block1 = ControlBlock(wake_time=1, agent=agent1, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[dataset_callback1, parameter_callback1])

    # Control Block x
    dataset_callback2 = CollectDataset()
    parameter_callback2 = CollectPolicyParameter(pi2)
    control_block2 = ControlBlock(wake_time=1, agent=agent2, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[dataset_callback2, parameter_callback2])

    # Function Block 1: picks state for hi lev ctrl
    function_block1 = fBlock(wake_time=1, phi=pick_state)

    # Function Block 2: maps the env to low lev ctrl state
    function_block2 = fBlock(wake_time=1, phi=rototranslate)

    # Function Block 3: holds curr state as ref
    function_block3 = fBlock(wake_time=100, phi=hold_state)

    # Function Block 4: adds hi lev rew
    function_block4 = addBlock(wake_time=1)

    # Function Block 5: adds low lev rew
    function_block5 = addBlock(wake_time=1)

    # Function Block 6:ext rew of hi lev ctrl
    function_block6 = fBlock(wake_time=1, phi=G_high)

    # Function Block 7: ext rew of low lev ctrl
    function_block7 = fBlock(wake_time=1, phi=G_low)

    #Mux_Block
    mux_block = MuxBlock(wake_time = 1)
    mux_block.add_block_list([control_block1])
    mux_block.add_block_list([control_block2])

    # Algorithm
    blocks = [state_ph, reward_ph, control_blockH, mux_block,
              function_block1, function_block2, function_block3,
              function_block4, function_block5,
              function_block6, function_block7]

    order = [0, 1, 4, 7, 9, 6, 2, 5, 10, 8, 3]
    state_ph.add_input(mux_block)
    reward_ph.add_input(mux_block)
    control_blockH.add_input(function_block1)
    control_blockH.add_reward(function_block4)
    mux_block.add_input(control_blockH)
    mux_block.add_input(function_block2)
    control_block1.add_reward(function_block5)
    control_block2.add_reward(function_block5)
    function_block1.add_input(state_ph)
    function_block2.add_input(control_blockH)
    function_block2.add_input(state_ph)
    function_block2.add_input(function_block3)
    function_block3.add_input(state_ph)
    function_block4.add_input(function_block6)
    function_block4.add_input(reward_ph)
    function_block5.add_input(reward_ph)
    function_block5.add_input(function_block7)
    function_block6.add_input(reward_ph)
    function_block7.add_input(control_blockH)
    function_block7.add_input(function_block2)

    computational_graph = ComputationalGraph(blocks=blocks, order=order, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    dataset_learn = core.learn(n_episodes=3000)
    # Evaluate
    dataset_eval = core.evaluate(n_episodes=10)

    # Visualize
    low_level_dataset1 = dataset_callback1.get()
    low_level_dataset2 = dataset_callback2.get()
    hi_lev_params = agentH.Q.get_weights()
    hi_lev_params = np.reshape(hi_lev_params, (400, 8))

    print hi_lev_params

    blablabla = np.zeros(shape=(400,))
    for i in xrange(400):
        blablabla[i] = max(hi_lev_params[i])

    max_q_val = np.reshape(blablabla, (20,20))
    print max_q_val
    plt.imshow(max_q_val, cmap='hot', interpolation='nearest')

    #parameter_dataset1 = parameter_callback1.get_values()
    #parameter_dataset2 = parameter_callback2.get_values()
    #visualize_policy_params(parameter_dataset1, parameter_dataset2)
    #visualize_control_block(low_level_dataset1)
    #visualize_control_block(low_level_dataset2)
    visualize_ship_steering(dataset_learn, name='learn', range_eps=xrange(2980,2995))
    visualize_ship_steering(dataset_eval, name='evaluate')
    plt.show()

    return

if __name__ == '__main__':
    experiment()
