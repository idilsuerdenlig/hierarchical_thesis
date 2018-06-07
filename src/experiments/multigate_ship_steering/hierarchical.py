import numpy as np

from mushroom.algorithms.value import QLearning
from mushroom.policy import EpsGreedy
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
from mushroom_hierarchical.blocks.mux_block import MuxBlock
from mushroom_hierarchical.functions.feature_angle_diff_ship_steering\
    import *
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.reward_accumulator import \
    reward_accumulator_block
from mushroom_hierarchical.functions.cost_cosine import cost_cosine
from mushroom_hierarchical.policy.deterministic_control_policy \
    import DeterministicControlPolicy


#def count_gates(dataset):
#    gates = list()

#    for i in range(len(dataset)):
#        if dataset[i][-1]:
#            gates.append(dataset[i][0][4])

#    return np.mean(gates)

def hi_lev_state(ins):

    x = np.concatenate(ins)
    out = np.zeros(4)
    res = 0

    for i in [4, 5, 6, 7]:
        if x[i] > 0:
            out[i-4] = 1
            res += 2**(i-4)

    return np.array([res])

def build_high_level_agent(alg, params, mdp, epsilon):
    pi = EpsGreedy(epsilon=epsilon, )
    mdp_info_high = MDPInfo(observation_space=spaces.Discrete(16),
                              action_space=spaces.Discrete(4),
                              gamma=mdp.info.gamma,
                              horizon=100)

    agent = alg(pi, mdp_info_high, **params)

    return agent


def build_mid_level_agent(alg, params, mdp, mu, sigma):
    n = mdp.no_of_gates - 1

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


def build_computational_graph(mdp, agent_low, agent_m1,
                            agent_m2, agent_m3, agent_m4, agent_high,
                              ep_per_fit_low, ep_per_fit_mid):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # Function Block 0
    function_block0 = fBlock(name='f0 (state build for high level)', phi=hi_lev_state)

    # Function Block 1
    function_block1 = fBlock(name='f1 (angle difference)',
                             phi=pos_ref_angle_difference)

    # Function Block 2
    function_block2 = fBlock(name='f2 (cost cosine)', phi=cost_cosine)

    # Control Block H
    control_block_h = ControlBlock(name='Control Block H', agent=agent_high,
                                  n_steps_per_fit=1)
    # Cotrol Block M1
    control_block_m1 = ControlBlock(name='Control Block M1', agent=agent_m1,
                                    n_eps_per_fit=ep_per_fit_mid)
    # Cotrol Block M2
    control_block_m2 = ControlBlock(name='Control Block M2', agent=agent_m2,
                                  n_eps_per_fit=ep_per_fit_mid)
    # Cotrol Block M3
    control_block_m3 = ControlBlock(name='Control Block M3', agent=agent_m3,
                                    n_eps_per_fit=ep_per_fit_mid)
    # Cotrol Block M4
    control_block_m4 = ControlBlock(name='Control Block M4', agent=agent_m4,
                                    n_eps_per_fit=ep_per_fit_mid)
    # Control Block L
    control_block_l = ControlBlock(name='Control Block L', agent=agent_low,
                                  n_eps_per_fit=ep_per_fit_low)
    # Selector Block
    mux_block = MuxBlock(name='Mux Block')
    mux_block.add_block_list([control_block_m1])
    mux_block.add_block_list([control_block_m2])
    mux_block.add_block_list([control_block_m3])
    mux_block.add_block_list([control_block_m4])

    # Reward Accumulator
    reward_acc = reward_accumulator_block(gamma=mdp.info.gamma,
                                          name='reward_acc')

    # Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_block_h,
              control_block_l, function_block0,
              function_block1, function_block2,
              reward_acc, mux_block]

    state_ph.add_input(control_block_l)
    reward_ph.add_input(control_block_l)
    lastaction_ph.add_input(control_block_l)

    control_block_h.add_input(function_block0)
    control_block_h.add_reward(reward_ph)

    mux_block.add_input(control_block_h)
    mux_block.add_input(state_ph)

    control_block_m1.add_reward(reward_acc)
    control_block_m1.add_alarm_connection(control_block_l)

    control_block_m2.add_reward(reward_acc)
    control_block_m2.add_alarm_connection(control_block_l)

    control_block_m3.add_reward(reward_acc)
    control_block_m3.add_alarm_connection(control_block_l)

    control_block_m4.add_reward(reward_acc)
    control_block_m4.add_alarm_connection(control_block_l)

    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block_l)

    function_block0.add_input(state_ph)

    function_block1.add_input(mux_block)
    function_block1.add_input(state_ph)

    function_block2.add_input(function_block1)

    control_block_l.add_input(function_block1)
    control_block_l.add_reward(function_block2)

    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def hierarchical_experiment(mdp, agent_l, agent_m1,
                            agent_m2, agent_m3, agent_m4,
                            agent_h, n_epochs,
                            n_iterations, ep_per_epoch_train,
                            ep_per_epoch_eval, ep_per_fit_low, ep_per_fit_mid):
    np.random.seed()

    computational_graph = build_computational_graph(mdp, agent_l, agent_m1,
                                                    agent_m2, agent_m3, agent_m4,
                                                    agent_h,
                                                    ep_per_fit_low, ep_per_fit_mid)

    core = HierarchicalCore(computational_graph)
    J_list = list()

    dataset = core.evaluate(n_episodes=ep_per_epoch_eval, quiet=True)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    J_list.append(np.mean(J))
    print('J at start: ', np.mean(J))
    #print('Mean gates passed: ', count_gates(dataset))

    for n in range(n_epochs):
        core.learn(n_episodes=n_iterations * ep_per_epoch_train, skip=True,
                   quiet=True)
        dataset = core.evaluate(n_episodes=ep_per_epoch_eval, quiet=True)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        print('J at iteration ', n, ': ', np.mean(J))
        #print('Mean gates passed: ', count_gates(dataset))

    return J_list
