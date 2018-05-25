import numpy as np

from mushroom.environments import MDPInfo
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.distributions import GaussianDiagonalDistribution
from mushroom.policy.gaussian_policy import *
from mushroom.utils.dataset import compute_J
from mushroom.utils import spaces

from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.functions.feature_angle_diff_ship_steering\
    import *
from mushroom_hierarchical.blocks.functions.gate_to_pass import GateToPass
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.reward_accumulator import \
    reward_accumulator_block
from mushroom_hierarchical.blocks.functions.cost_cosine import cost_cosine
from mushroom_hierarchical.policy.deterministic_control_policy \
    import DeterministicControlPolicy


def build_high_level_agent(alg, params, mdp, mu, sigma):
    n = mdp.no_of_gates - 1
    w = np.array([mu] * n)

    mu_approximator = Regressor(LinearApproximator, input_shape=(n,),
                              output_shape=(2,))
    sigma_approximator = Regressor(LinearApproximator, input_shape=(n,),
                                   output_shape=(2,))
    w_mu = mu*np.ones(mu_approximator.weights_size)
    mu_approximator.set_weights(w_mu)

    w_sigma = sigma*np.ones(mu_approximator.weights_size)
    sigma_approximator.set_weights(w_sigma)

    pi = StateStdGaussianPolicy(mu=mu_approximator,
                                std=sigma_approximator)

    lim = mdp.info.observation_space.high[0]
    mdp_info_agent1 = MDPInfo(observation_space=spaces.Box(0, 1, (3,)),
                              action_space=spaces.Box(0, lim, (2,)),
                              gamma=mdp.info.gamma,
                              horizon=100)
    agent = alg(pi, mdp_info_agent1, **params)

    return agent


def build_low_level_agent(alg, params, mdp):
    pi = DeterministicControlPolicy(weights=np.array([0]))
    mu = np.zeros(pi.weights_size)
    sigma = 1e-3 * np.ones(pi.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    mdp_info_agent2 = MDPInfo(observation_space=spaces.Box(-np.pi, np.pi, (1,)),
                              action_space=mdp.info.action_space,
                              gamma=mdp.info.gamma, horizon=100)
    agent = alg(distribution,pi, mdp_info_agent2, **params)

    return agent


def build_computational_graph(mdp, agent_low, agent_high,
                              ep_per_fit_low, ep_per_fit_high):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # Function Block 1
    function_block1 = fBlock(name='f1 (angle difference)',
                             phi=pos_ref_angle_difference)

    # Function Block 2
    function_block2 = fBlock(name='f2 (cost cosine)', phi=cost_cosine)

    # Function Block 3
    gate_to_pass = GateToPass(n_gates=3)
    function_block3 = fBlock(name='f3 (gate index)',
                             phi=gate_to_pass)

    # Control Block H
    control_block_h = ControlBlock(name='Control Block H', agent=agent_high,
                                  n_eps_per_fit=ep_per_fit_high)

    # Control Block 2
    control_block_l = ControlBlock(name='Control Block L', agent=agent_low,
                                  n_eps_per_fit=ep_per_fit_low)

    # Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp.info.gamma,
                                          name='reward_acc')

    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block_h,
              control_block_l, function_block1, function_block2, function_block3,
              reward_acc]

    state_ph.add_input(control_block_l)
    reward_ph.add_input(control_block_l)
    lastaction_ph.add_input(control_block_l)

    function_block3.add_input(state_ph)

    control_block_h.add_input(function_block3)
    control_block_h.add_reward(reward_acc)
    control_block_h.add_alarm_connection(control_block_l)

    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block_l)

    function_block1.add_input(control_block_h)
    function_block1.add_input(state_ph)

    function_block2.add_input(function_block1)

    control_block_l.add_input(function_block1)
    control_block_l.add_reward(function_block2)

    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def hierarchical_experiment(mdp, agent_low, agent_high,
                            n_epochs, n_iterations,
                            ep_per_iteration, ep_per_eval,
                            ep_per_iteration_low):
    np.random.seed()

    computational_graph = build_computational_graph(mdp, agent_low, agent_high,
                                                    ep_per_iteration_low,
                                                    ep_per_iteration)

    core = HierarchicalCore(computational_graph)
    J_list = list()

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))

    for n in range(n_epochs):
        core.learn(n_episodes=n_iterations * ep_per_iteration, skip=True,
                   quiet=True)
        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))

    return J_list
