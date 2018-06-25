from mushroom.environments import MDPInfo
from mushroom.features.tiles import Tiles
from mushroom.features.features import *
from mushroom.features.basis import *
from mushroom.policy.gaussian_policy import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import compute_J
from mushroom.utils import spaces
from mushroom.utils.angles_utils import *


from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.reward_accumulator import *


def reward_low_level(ins):
    state = ins[0]

    value = np.cos(state[0]) - state[1]**2
    return np.array([value])


def pick_position(ins):
    state = ins[0]

    return state[:2]


def angle_and_distance(ins):
    state = ins[0]
    drift = ins[1]

    x = state[0]
    y = state[1]
    theta = state[2]
    x_prey = state[3]
    y_prey = state[4]

    del_x = x_prey - x
    del_y = y_prey - y
    theta_ref = normalize_angle(np.arctan2(del_y, del_x)+drift)

    del_theta = shortest_angular_distance(from_angle=theta,
                                          to_angle=theta_ref)

    pos_predator = ins[0][0:2]
    pos_prey = ins[0][3:5]

    distance = np.linalg.norm(pos_prey-pos_predator)

    return np.array([del_theta, distance])


def build_high_level_agent(alg, params, mdp, std):
    high = mdp.info.observation_space.high[:2]
    low = mdp.info.observation_space.low[:2]

    observation_space = spaces.Box(low=low, high=high)
    action_space = spaces.Box(low=np.array([-np.pi]), high=np.array([-np.pi]))

    mdp_info_agent = MDPInfo(observation_space=observation_space,
                             action_space=action_space,
                             gamma=mdp.info.gamma,
                             horizon=mdp.info.horizon)

    tiles = Tiles.generate(10, [5, 5], low, high)
    features = Features(tilings=tiles)
    approximator = Regressor(LinearApproximator, input_shape=(features.size,),
                             output_shape=(1,))

    pi1 = DiagonalGaussianPolicy(approximator, std)

    agent = alg(pi1, mdp_info_agent, features=features, **params)

    return agent


'''def build_high_level_bbo_agent(alg, params, mdp, std):
    high = mdp.info.observation_space.high[:3]
    low = mdp.info.observation_space.low[:3]

    observation_space = spaces.Box(low=low, high=high)
    action_space = spaces.Box(low=np.array([-np.pi]), high=np.array([-np.pi]))

    mdp_info_agent = MDPInfo(observation_space=observation_space,
                             action_space=action_space,
                             gamma=mdp.info.gamma,
                             horizon=mdp.info.horizon)

    tiles = Tiles.generate(3, [10, 10, 10], low, high)
    features = Features(tilings=tiles)
    approximator = Regressor(LinearApproximator, input_shape=(features.size,),
                             output_shape=(1,))

    mu = np.zeros(approximator.weights_size)
    std = std*np.ones(approximator.weights_size)
    dist = GaussianDiagonalDistribution(mu, std)
    pi = DeterministicPolicy(approximator)

    agent = alg(dist, pi, mdp_info_agent, features=features, **params)

    return agent'''


def build_low_level_agent(alg, params, mdp, horizon, std):
    basis = PolynomialBasis.generate(1, 2)
    features = Features(basis_list=basis)
    approximator = Regressor(LinearApproximator, input_shape=(features.size,),
                             output_shape=mdp.info.action_space.shape)

    pi = DiagonalGaussianPolicy(approximator, std)

    mdp_info_agent = MDPInfo(observation_space=spaces.Box(-np.pi, np.pi, (1,)),
                             action_space=mdp.info.action_space,
                             gamma=mdp.info.gamma, horizon=horizon)
    agent = alg(pi, mdp_info_agent, features=features, **params)

    return agent


def build_computational_graph(mdp, agent_low, agent_high,
                              ep_per_fit_low, ep_per_fit_high,
                              low_level_callbacks=[]):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    function_block1 = fBlock(name='pick position',
                             phi=pick_position)

    function_block2 = fBlock(name='angle and distance',
                             phi=angle_and_distance)
    function_block3 = fBlock(name='cost cosine', phi=reward_low_level)

    reward_acc = reward_accumulator_block(mdp.info.gamma,
                                          name='reward accumulator')

    control_block_h = ControlBlock(name='Control Block H', agent=agent_high,
                                   n_eps_per_fit=ep_per_fit_high)

    control_block_l = ControlBlock(name='Control Block L', agent=agent_low,
                                   n_eps_per_fit=ep_per_fit_low,
                                   callbacks=low_level_callbacks)

    blocks = [state_ph, reward_ph, lastaction_ph,
              control_block_h, control_block_l,
              function_block1, function_block2, function_block3,
              reward_acc]

    state_ph.add_input(control_block_l)
    reward_ph.add_input(control_block_l)
    lastaction_ph.add_input(control_block_l)

    function_block1.add_input(state_ph)

    reward_acc.add_input(reward_ph)
    reward_acc.add_alarm_connection(control_block_l)

    control_block_h.add_input(function_block1)
    control_block_h.add_reward(reward_acc)
    control_block_h.add_alarm_connection(control_block_l)

    function_block2.add_input(state_ph)
    function_block2.add_input(control_block_h)

    function_block3.add_input(function_block2)

    control_block_l.add_input(function_block2)
    control_block_l.add_reward(function_block3)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def experiment(mdp, agent_high, agent_low,
               n_epochs, n_episodes, ep_per_eval,
               ep_per_fit_low, ep_per_fit_high):
    np.random.seed()

    dataset_callback = CollectDataset()

    computational_graph = build_computational_graph(
        mdp, agent_low, agent_high, ep_per_fit_low, ep_per_fit_high,
        [dataset_callback])

    core = HierarchicalCore(computational_graph)
    J_list = list()

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=False)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    print('Reward at start :', np.mean(J))
    J_list.append(np.mean(J))

    for n in range(n_epochs):
        core.learn(n_episodes=n_episodes, skip=True, quiet=False)

        ll_dataset = dataset_callback.get()
        dataset_callback.clean()
        J_low = compute_J(ll_dataset, mdp.info.gamma)
        print('Low level reward at epoch', n, ':', np.mean(J_low))

        dataset = core.evaluate(n_episodes=ep_per_eval, quiet=False)
        J = compute_J(dataset, gamma=mdp.info.gamma)
        J_list.append(np.mean(J))
        print('Reward at epoch ', n, ':', np.mean(J))

    print('Press a key to run visualization')
    input()
    core.evaluate(n_episodes=1, render=True)
    return J_list
