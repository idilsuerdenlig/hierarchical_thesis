import numpy as np

from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from M_block import MBlock
from control_block import ControlBlock
from simple_agent import SimpleAgent
from mushroom.utils import spaces
from mushroom.environments import *




def experiment():
    np.random.seed(3)
    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=.9)

    action_space = mdp._mdp_info.action_space
    observation_space = mdp._mdp_info.observation_space
    gamma = mdp._mdp_info.gamma
    horizon = mdp._mdp_info.horizon

    # Model Block
    model_block = MBlock(env=mdp, render=False)

    #Agents
    mdp_info_agent1 = MDPInfo(observation_space=observation_space, action_space=spaces.Discrete(5), gamma=1, horizon=20)
    mdp_info_agent2 = MDPInfo(observation_space=spaces.Discrete(5), action_space=action_space, gamma=gamma, horizon=10)
    agent1 = SimpleAgent(name='HIGH', mdp_info=mdp_info_agent1)
    agent2 = SimpleAgent(name='LOW', mdp_info=mdp_info_agent2)


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


