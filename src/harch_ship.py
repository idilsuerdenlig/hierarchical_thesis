import numpy as np

from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from M_block import MBlock
from control_block import ControlBlock
from J_block import JBlock
from f_block import fBlock
from simple_agent import SimpleAgent
from mushroom.utils import spaces
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter, AdaptiveParameter
from mushroom.utils.table import Table
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy import GaussianPolicy, MultivariateGaussianPolicy
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import REINFORCE
from visualize_ship_steering import visualizeShipSteering
import matplotlib.pyplot as plt
from visualize_control_block import VisualizeControlBlock


class CollectPolicyParameter:
    """
    This callback can be used to collect the values of a parameter
    (e.g. learning rate) during a run of the agent.

    """
    def __init__(self, policy):

        self._policy = policy
        self._p = list()

    def __call__(self, **kwargs):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        value = self._policy.get_weights()
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._p.append(value)

    def get_values(self):

        return self._p


def experiment():
    np.random.seed()
    # MDP
    mdp = ShipSteering()

    action_space = mdp.info.action_space
    observation_space = mdp.info.observation_space
    gamma = mdp.info.gamma

    # Model Block
    model_block = MBlock(env=mdp, render=False)


    # Feature function Block

    def phi(ins):
        x_ref = ins[0][0]
        y_ref = ins[0][1]
        x = ins[1][0]
        y = ins[1][1]
        theta = ins [1][2]
        del_x = x_ref-x
        del_y = y_ref-y
        theta_ref = np.arctan2(del_y, del_x)
        theta_ref = (theta_ref + np.pi) % (2 * np.pi) - np.pi
        theta = np.pi/2-theta
        del_theta = theta_ref-theta
        del_theta = (del_theta + np.pi) % (2 * np.pi) - np.pi
        return np.array([del_theta])

    function_block = fBlock(wake_time=1, phi=phi)

    # Agents
    mdp_info_agent1 = MDPInfo(observation_space=observation_space, action_space=spaces.Box(0,150,(2,)), gamma=gamma, horizon=100)
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi,np.pi,(1,)), action_space=action_space, gamma=gamma, horizon=100)

    learning_rate = AdaptiveParameter(value=.01)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}

    basis = [PolynomialBasis()]
    features = Features(basis_list=basis)

    # Policy
    sigma1 = np.array([[.001, 0], [0, .001]])
    approximator1 = Regressor(LinearApproximator, input_shape=(features.size,), output_shape=(2,))
    pi1 = MultivariateGaussianPolicy(mu=approximator1,sigma=sigma1)
    pi1.set_weights(np.array([100, 100]))

    sigma2 = Parameter(value=.1)
    approximator2 = Regressor(LinearApproximator, input_shape=(1,), output_shape=mdp.info.action_space.shape)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma2)
    pi2.set_weights(np.array([-0.1]))

    agent1 = REINFORCE(policy=pi1, mdp_info=mdp_info_agent1, params=agent_params, features=features)

    learning_rate = AdaptiveParameter(value=.01)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent2 = REINFORCE(policy=pi2, mdp_info=mdp_info_agent2, params=agent_params, features=None)

    # Control Blocks
    dataset_callback = CollectDataset()
    parameter_callback = CollectPolicyParameter(pi2)

    control_block1 = ControlBlock(wake_time=10, agent=agent1, n_eps_per_fit=10, n_steps_per_fit=None)
    control_block2 = ControlBlock(wake_time=1, agent=agent2, n_eps_per_fit=10, n_steps_per_fit=None, callbacks=[dataset_callback, parameter_callback])

    # Reward Block
    def dummy_reward(inputs):
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
    #print dataset_learn

    '''
    for dataset_learn_step in dataset_learn:

        print '---------------------------------------------------------'
        print dataset_learn_step[0]
        print dataset_learn_step[2]
        print dataset_learn_step[-1]
        print dataset_learn_step[-2]
    '''

    low_level_dataset = dataset_callback.get()
    parameter_dataset = parameter_callback.get_values()
    print parameter_dataset
    VisualizeControlBlock(low_level_dataset)
    #Evaluate
    dataset_eval = core.evaluate(n_episodes=100)
    #print dataset_eval

    #Visualize
    visualizeShipSteering(dataset_learn)
    visualizeShipSteering(dataset_eval)
    plt.show()

    return

if __name__ == '__main__':
    experiment()
