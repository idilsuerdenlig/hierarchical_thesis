from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from M_block import MBlock
from control_block import ControlBlock
from J_block import JBlock
from f_block import fBlock
from mushroom.utils import spaces
from mushroom.environments import *
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy.gaussian_policy import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import REINFORCE
from visualize_ship_steering import visualizeShipSteering
import matplotlib.pyplot as plt
from visualize_control_block import VisualizeControlBlock
from collect_policy_parameter import CollectPolicyParameter
from visualize_policy_params import VisualizePolicyParams
from feature_angle_diff_ship_steering import phi


def experiment():
    np.random.seed()


    # Model Block
    mdp = ShipSteering()
    model_block = MBlock(env=mdp, render=False)

    # Feature Block 1
    function_block = fBlock(wake_time=1, phi=phi)

    # Feature Block 2
    basis = [PolynomialBasis()]
    features = Features(basis_list=basis)

    # Policy 1
    sigma1 = np.array([[.01, 0], [0, .01]])
    approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
    pi1 = MultivariateGaussianPolicy(mu=approximator1,sigma=sigma1)
    pi1.set_weights(np.array([90, 90]))

    # Policy 2
    sigma2 = Parameter(value=.1)
    approximator2 = Regressor(LinearApproximator, input_shape=(1,), output_shape=mdp.info.action_space.shape)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma2)
    #pi2.set_weights(np.array([-0.01]))

    # Agent 1
    learning_rate = AdaptiveParameter(value=.01)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agent1 = MDPInfo(observation_space=mdp.info.observation_space, action_space=spaces.Box(0,150,(2,)), gamma=mdp.info.gamma, horizon=100)
    agent1 = REINFORCE(policy=pi1, mdp_info=mdp_info_agent1, params=agent_params, features=features)

    # Agent 2
    learning_rate = AdaptiveParameter(value=.001)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi,np.pi,(1,)), action_space=mdp.info.action_space, gamma=mdp.info.gamma, horizon=100)
    agent2 = REINFORCE(policy=pi2, mdp_info=mdp_info_agent2, params=agent_params, features=None)

    # Control Block 1
    parameter_callback1 = CollectPolicyParameter(pi1)
    control_block1 = ControlBlock(wake_time=10, agent=agent1, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[parameter_callback1])

    # Control Block 2
    dataset_callback = CollectDataset()
    parameter_callback2 = CollectPolicyParameter(pi2)
    control_block2 = ControlBlock(wake_time=1, agent=agent2, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[dataset_callback, parameter_callback2])

    # Reward Block
    def dummy_reward(inputs, absorbing):
        if absorbing:
            reward = -10000
        else:
            reward = -(inputs.dot(inputs))
        return reward

    reward_block = JBlock(wake_time=1, reward_function=dummy_reward)

    # Algorithm
    blocks = [model_block, control_block1, control_block2, reward_block, function_block]
    order = [0, 1, 4, 3, 2]
    model_block.add_input(control_block2)
    reward_block.add_input(function_block)
    control_block1.add_input(model_block)
    control_block1.add_reward(model_block)
    function_block.add_input(control_block1)
    function_block.add_input(model_block)
    control_block2.add_input(function_block)
    control_block2.add_reward(reward_block)
    computational_graph = ComputationalGraph(blocks=blocks, order=order)
    core = HierarchicalCore(computational_graph)

    # Train
    dataset_learn = core.learn(n_episodes=1000, quiet=True)

    # Evaluate
    dataset_eval = core.evaluate(n_episodes=100)

    # Visualize
    low_level_dataset = dataset_callback.get()
    parameter_dataset1 = parameter_callback1.get_values()
    parameter_dataset2 = parameter_callback2.get_values()
    VisualizePolicyParams(parameter_dataset1, parameter_dataset2)
    VisualizeControlBlock(low_level_dataset)
    visualizeShipSteering(dataset_learn)
    visualizeShipSteering(dataset_eval)
    plt.show()

    return

if __name__ == '__main__':
    experiment()
