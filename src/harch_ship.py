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
from mushroom.features.basis import *
from mushroom.features.features import *
from mushroom.policy import GaussianPolicy
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.algorithms.policy_search import REINFORCE



def experiment():
    np.random.seed(3)
    # MDP
    mdp = ShipSteering()

    action_space = mdp.info.action_space
    observation_space = mdp.info.observation_space
    gamma = mdp.info.gamma

    # Model Block
    model_block = MBlock(env=mdp, render=False)

    # Policy
    sigma = Parameter(value=.05)
    approximator_params1 = dict(input_dim=4)
    approximator1 = Regressor(LinearApproximator, input_shape=(4,), output_shape=(2,), params=approximator_params1)
    pi1 =GaussianPolicy(mu=approximator1,sigma=sigma)

    approximator_params2 = dict(input_dim=1)
    approximator2 = Regressor(LinearApproximator, input_shape=(1,),
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params2)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma)

    # Feature function Block

    def phi(ins):
        x_ref = ins[0][0]
        y_ref = ins[0][1]
        x = ins[1][0]
        y = ins[1][1]
        theta = ins [1][2]
        del_x = x-x_ref
        del_y = y - y_ref
        del_theta = np.arctan((del_x/del_y))-theta
        print del_theta
        return np.array([del_theta])

    function_block = fBlock(wake_time=1, phi=phi)

    basis = PolynomialBasis()
    features = Features(basis)
    # Agents
    mdp_info_agent1 = MDPInfo(observation_space=observation_space, action_space=spaces.Box(0,150,(2,)), gamma=1, horizon=None)
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(0,150,(1,)), action_space=action_space, gamma=gamma, horizon=None)
    agent1 = SimpleAgent(name='HIGH', mdp_info=mdp_info_agent1, policy=pi1)

    learning_rate = AdaptiveParameter(value=.01)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent2 = REINFORCE(policy=pi2, mdp_info=mdp_info_agent2, params=agent_params, features=None)

    # Control Blocks
    control_block1 = ControlBlock(wake_time=10, agent=agent1, n_eps_per_fit=None, n_steps_per_fit=1, horizon=10)
    control_block2 = ControlBlock(wake_time=1, agent=agent2, n_eps_per_fit=1, n_steps_per_fit=None, horizon=20)

    # Reward Block
    def dummy_reward(inputs):
        if abs(inputs) <= 0.005:
            print 'GOOD'
            return 10
        elif abs(inputs) <= 0.05:
            print 'MEEH'
            return 0
        else:
            print 'NOPE'
            return -1

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
    core.learn(n_steps=40, quiet=True)
    return

if __name__ == '__main__':
    experiment()
