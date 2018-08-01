from mushroom.environments import MDPInfo
from mushroom.features.tiles import Tiles
from mushroom.features.features import *
from mushroom.features.basis import *
from mushroom.policy.gaussian_policy import *
from mushroom.policy.td_policy import EpsGreedy
from mushroom.approximators.parametric import LinearApproximator, PyTorchApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import compute_J
from mushroom.utils import spaces
from mushroom.utils.angles import *


from mushroom_hierarchical.core.hierarchical_core import HierarchicalCore
from mushroom_hierarchical.blocks.computational_graph import ComputationalGraph
from mushroom_hierarchical.blocks.control_block import ControlBlock
from mushroom_hierarchical.blocks.basic_operation_block import *
from mushroom_hierarchical.blocks.model_placeholder import PlaceHolder
from mushroom_hierarchical.blocks.reward_accumulator import *

from network import Network

def reward_low_level(ins):
    state = ins[0]

    value = np.cos(state[0])
    return np.array([value])


def compute_angle(ins):
    n_actions = 4

    state = ins[0]
    action = int(np.asscalar(ins[1]))
    #print('compute_angle: ', action)

    if action == n_actions:
        x = state[0]
        y = state[1]
        x_prey = state[3]
        y_prey = state[4]

        del_x = x_prey - x
        del_y = y_prey - y
        theta_target = normalize_angle(np.arctan2(del_y, del_x))
    else:
        theta_target = 2*np.pi/n_actions*action-np.pi

    return np.array([theta_target])


def pick_position(ins):
    state = ins[0]

    '''
    x = state[0]
    y = state[1]
    #theta = state[2]
    x_prey = state[3]
    y_prey = state[4]

    del_x = x_prey - x
    del_y = y_prey - y
    theta_target = normalize_angle(np.arctan2(del_y, del_x))
    distance = np.sqrt(del_x**2 + del_y**2)

    return np.array([x, y, theta_target, distance])
    #return np.array([x, y, theta_target])
    '''

    return np.concatenate([state[0:2], state[3:5]], 0)


def angle_error(ins):
    theta = ins[0][2]
    theta_ref = np.asscalar(ins[1])

    #print('angle_error: ', theta_ref)

    error = shortest_angular_distance(from_angle=theta,
                                      to_angle=theta_ref)

    return np.array([error])


def build_high_level_agent(alg, params, optim, loss, mdp, eps):
    high = np.ones(4)
    low = np.zeros(4)

    high[:2] = mdp.info.observation_space.high[:2]
    low[:2] = mdp.info.observation_space.low[:2]

    high[2:] = mdp.info.observation_space.high[3:5]
    low[2:] = mdp.info.observation_space.low[3:5]

    n_actions = 5
    observation_space = spaces.Box(low=low, high=high)
    action_space = spaces.Discrete(n_actions)

    mdp_info = MDPInfo(observation_space=observation_space,
                             action_space=action_space,
                             gamma=mdp.info.gamma,
                             horizon=mdp.info.horizon)

    pi = EpsGreedy(eps)

    approximator_params = dict(network=Network,
                               optimizer={'class': optim,
                                          'params': {'lr': .001}},
                               loss=loss,
                               n_features=80,
                               input_shape=mdp_info.observation_space.shape,
                               output_shape=mdp_info.action_space.size,
                               n_actions=mdp_info.action_space.n)

    agent = alg(PyTorchApproximator, pi, mdp_info,
                approximator_params=approximator_params, **params)

    return agent


def build_low_level_agent(alg, params, mdp, horizon, std):
    basis = PolynomialBasis.generate(1, 2)
    features = Features(basis_list=basis)
    features = None
    approximator = Regressor(LinearApproximator, input_shape=(1,),
                             #input_shape=(features.size,),
                             output_shape=mdp.info.action_space.shape)

    pi = DiagonalGaussianPolicy(approximator, std)

    mdp_info_agent = MDPInfo(observation_space=spaces.Box(-np.pi, np.pi, (1,)),
                             action_space=mdp.info.action_space,
                             gamma=mdp.info.gamma, horizon=horizon)
    agent = alg(pi, mdp_info_agent, features=features, **params)

    return agent


def build_computational_graph(mdp, agent_low, agent_high,
                              ep_per_fit_low, low_level_callbacks=[]):

    # State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    # Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    # Last_In Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    function_block1 = fBlock(name='pick position',
                             phi=pick_position)

    function_block2 = fBlock(name='compute angle',
                             phi=compute_angle)

    function_block3 = fBlock(name='angle and distance',
                             phi=angle_error)
    function_block4 = fBlock(name='cost cosine', phi=reward_low_level)

    reward_acc = reward_accumulator_block(mdp.info.gamma,
                                          name='reward accumulator')

    control_block_h = ControlBlock(name='Control Block H', agent=agent_high,
                                   n_steps_per_fit=1)

    control_block_l = ControlBlock(name='Control Block L', agent=agent_low,
                                   n_eps_per_fit=ep_per_fit_low,
                                   callbacks=low_level_callbacks)

    blocks = [state_ph, reward_ph, lastaction_ph,
              control_block_h, control_block_l,
              function_block1, function_block2,
              function_block3, function_block4,
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

    function_block3.add_input(state_ph)
    function_block3.add_input(function_block2)

    function_block4.add_input(function_block3)

    control_block_l.add_input(function_block3)
    control_block_l.add_reward(function_block4)
    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)

    return computational_graph


def experiment(mdp, agent_high, agent_low,
               n_epochs, n_episodes, ep_per_eval,
               ep_per_fit_low):
    np.random.seed()

    dataset_callback = CollectDataset()

    computational_graph = build_computational_graph(
        mdp, agent_low, agent_high, ep_per_fit_low,
        [dataset_callback])

    core = HierarchicalCore(computational_graph)
    J_list = list()

    dataset = core.evaluate(n_episodes=ep_per_eval, quiet=False)
    J = compute_J(dataset, gamma=mdp.info.gamma)
    print('Reward at start :', np.mean(J))
    J_list.append(np.mean(J))

    #print('Press a key to run visualization')
    #input()
    #core.evaluate(n_episodes=1, render=True)

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

        #print('Press a key to run visualization')
        #input()
        core.evaluate(n_episodes=1, render=True)

    print('Press a key to run visualization')
    input()
    core.evaluate(n_episodes=1, render=True)
    return J_list
