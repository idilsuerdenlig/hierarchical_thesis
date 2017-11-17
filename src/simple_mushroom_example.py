import numpy as np

from mushroom.algorithms.value.td import QLearning
from mushroom.core.core import Core
from mushroom.environments import *
from mushroom.policy import EpsGreedy
from mushroom.utils.parameters import Parameter


def experiment():
    np.random.seed(3)

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
    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1)
    print agent.Q.table


if __name__ == '__main__':
    experiment()
