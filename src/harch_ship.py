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
from collect_J import CollectJ
from pick_last_ep_dataset import PickLastEp


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

    #Features
    features = Features(basis_list=[PolynomialBasis()])

    # Policy 1
    sigma1 = np.array([38, 38])
    approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
    approximator1.set_weights(np.array([75, 75]))
    pi1 = MultivariateDiagonalGaussianPolicy(mu=approximator1,sigma=sigma1)

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
    #dataset_gradient = CollectJ(agent1, kwargs1='dataset')
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
    #dataset_learn_visual = core.learn(n_episodes=3000)
    dataset_learn_visual = list()
    for n in xrange(6):
        dataset_learn = core.learn(n_episodes=1000)
        last_ep_dataset = PickLastEp(dataset_learn)
        dataset_learn_visual += last_ep_dataset
        del dataset_learn

    # Evaluate
    dataset_eval = core.evaluate(n_episodes=10)
    # Visualize
    low_level_dataset = dataset_callback.get()
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    VisualizePolicyParams(parameter_dataset1, parameter_dataset2)
    #VisualizeControlBlock(low_level_dataset)
    visualizeShipSteering(dataset_learn_visual, 'learn')
    visualizeShipSteering(dataset_eval, 'evaluate')
    plt.show()

    return

if __name__ == '__main__':
    experiment()
