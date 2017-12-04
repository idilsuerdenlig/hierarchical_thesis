import numpy as np

from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from M_block import MBlock
from control_block import ControlBlock

from mushroom.algorithms.value.td import QLearning
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter
from mushroom.utils.callbacks import CollectDataset
from mushroom.core.core import Core
from mushroom.environments import *




def experiment():
    np.random.seed(3)
    print 'hierarchical     :'
    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=.9)

    # Model Block
    model_block = MBlock(env=mdp, render=False)

    # Policy
    epsilon = Parameter(value=.15)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.2)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QLearning(pi, mdp.info, agent_params)


    # Control Block
    control_block = ControlBlock(wake_time = 1, agent=agent, n_eps_per_fit=None, n_steps_per_fit=1, horizon=mdp.info.horizon)

    # Algorithm
    blocks = [model_block, control_block]
    order = [0, 1]
    model_block.add_input(control_block)
    control_block.add_input(model_block)
    control_block.add_reward(model_block)
    computational_graph = ComputationalGraph(blocks=blocks, order=order)
    core = HierarchicalCore(computational_graph)


    # Train
    core.learn(n_steps=100, quiet=True)
    return agent.Q.table

def experiment2():
    np.random.seed(3)
    print 'mushroom     :'

    # MDP
    mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
                                gamma=.9)

    # Policy
    epsilon = Parameter(value=.15)
    pi = EpsGreedy(epsilon=epsilon,)

    # Agent
    learning_rate = Parameter(value=.2)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QLearning(pi, mdp.info, agent_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)
    dataset=collect_dataset.get()
    return agent.Q.table


if __name__ == '__main__':
    Q1=experiment()
    Q2=experiment2()
    print Q1
    print Q2
    assert np.array_equal(Q1,Q2)

