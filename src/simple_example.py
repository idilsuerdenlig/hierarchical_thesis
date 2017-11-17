import numpy as np

from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from M_block import MBlock
from control_block import ControlBlock
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter
from mushroom.algorithms.value.td import QLearning


def experiment():
    np.random.seed()

    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=.9)
    # Model Block
    model_block = MBlock(env= mdp, render= False)



    # Agent

    epsilon = Parameter(value=.15)
    pi = EpsGreedy(epsilon=epsilon, observation_space=mdp.observation_space,
                   action_space=mdp.action_space)
    shape = mdp.observation_space.size + mdp.action_space.size
    learning_rate = Parameter(value=.2)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QLearning(shape, pi, mdp.gamma, agent_params)

    # Control Block
    control_block = ControlBlock(wake_time = 1, agent=agent, episode_length=8,fit_time = 5 )

    # Algorithm
    blocks = [model_block, control_block]
    order = [0, 1]
    model_block.add_input(control_block)
    control_block.add_input(model_block)
    control_block.add_reward(model_block)
    computational_graph = ComputationalGraph(blocks=blocks, order=order)
    core = HierarchicalCore(computational_graph)

    # Train
    core.learn(n_iterations=100)


if __name__ == '__main__':
    experiment()
