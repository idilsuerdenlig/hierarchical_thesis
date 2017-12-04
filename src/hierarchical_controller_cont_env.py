import numpy as np

from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from M_block import MBlock
from control_block import ControlBlock
from simple_agent import SimpleAgent
from mushroom.utils import spaces
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter
from mushroom.utils.table import Table
from mushroom.features.basis import GaussianRBF
from mushroom.features.features import Features
from mushroom.policy import GaussianPolicy
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor


def experiment():
    np.random.seed(3)
    # MDP
    mdp = ShipSteering()

    action_space = mdp.info.action_space
    observation_space = mdp.info.observation_space
    gamma = mdp.info.gamma

    # Model Block
    model_block = MBlock(env=mdp, render=False)
    input_shape = observation_space.shape

    approximator_params2 = dict(input_dim=observation_space.shape[0])
    approximator2 = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params2)
    sigma = Parameter(value=.05)
    pi2 = GaussianPolicy(mu=approximator2, sigma=sigma)

    approximator_params1 = dict(input_dim=4)
    approximator1 = Regressor(LinearApproximator, input_shape=(4,), output_shape=(4,), params=approximator_params1)
    pi1 =GaussianPolicy(mu=approximator1,sigma=sigma)


    #Agents
    mdp_info_agent1 = MDPInfo(observation_space=observation_space, action_space=spaces.Box(0,150,(4,)), gamma=1, horizon=3)
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(0,150,(4,)), action_space=action_space, gamma=gamma, horizon=5)
    agent1 = SimpleAgent(name='HIGH', mdp_info=mdp_info_agent1, policy = pi1)
    agent2 = SimpleAgent(name='LOW', mdp_info=mdp_info_agent2, policy = pi2)


    # Control Blocks
    control_block1 = ControlBlock(wake_time=10, agent=agent1, n_eps_per_fit=None, n_steps_per_fit=1, horizon=None)
    control_block2 = ControlBlock(wake_time=1, agent=agent2, n_eps_per_fit=None, n_steps_per_fit=1, horizon=None)

    # Algorithm
    blocks = [model_block, control_block1, control_block2]
    order = [0, 1, 2]
    model_block.add_input(control_block2)
    control_block1.add_input(model_block)
    control_block1.add_reward(model_block)
    control_block2.add_input(control_block1)
    control_block2.add_reward(model_block)
    computational_graph = ComputationalGraph(blocks=blocks, order=order)
    core = HierarchicalCore(computational_graph)


    # Train
    core.learn(n_steps=40, quiet=True)
    return


if __name__ == '__main__':
    experiment()


