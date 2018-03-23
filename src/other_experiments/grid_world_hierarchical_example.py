import numpy as np

from hierarchical_core import HierarchicalCore
from computational_graph import ComputationalGraph
from .library.blocks.model_placeholder import ModelPlaceholder
from blocks.control_block import ControlBlock

from mushroom.algorithms.value.td import QLearning, DoubleQLearning,\
    WeightedQLearning, SpeedyQLearning, SARSA
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter
from mushroom.utils.callbacks import CollectDataset
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.utils.parameters import ExponentialDecayParameter
from visualize_control_block import VisualizeControlBlock



def experiment():
    np.random.seed(3)
    print('hierarchical     :')

    mdp = GridWorldVanHasselt()

    # Model Block
    model_block = MBlock(env=mdp, render=False)

    # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                                        size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=1.,
                                              size=mdp.info.size)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = QLearning(pi, mdp.info, agent_params)


    # Control Block
    control_block = ControlBlock(name='controller', agent=agent, n_steps_per_fit=1)

    # Algorithm
    blocks = [model_block, control_block]
    order = [0, 1]
    model_block.add_input(control_block)
    control_block.add_input(model_block)
    control_block.add_reward(model_block)
    computational_graph = ComputationalGraph(blocks=blocks, order=order)
    core = HierarchicalCore(computational_graph)


    # Train
    core.learn(n_steps=2000, quiet=True)
    return agent.Q.table

def experiment2():
    np.random.seed(3)
    print('mushroom     :')

    mdp = GridWorldVanHasselt()

   # Policy
    epsilon = ExponentialDecayParameter(value=1, decay_exp=.5,
                                        size=mdp.info.observation_space.size)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = ExponentialDecayParameter(value=1., decay_exp=1.,
                                              size=mdp.info.size)
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
    core.learn(n_steps=2000, n_steps_per_fit=1, quiet=True)

    # Train
    dataset=collect_dataset.get()
    VisualizeControlBlock(dataset)
    return agent.Q.table


if __name__ == '__main__':
    Q1=experiment()
    Q2=experiment2()
    assert np.array_equal(Q1,Q2)


