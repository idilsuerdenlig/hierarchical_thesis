import datetime
from joblib import Parallel, delayed

from mushroom.environments.environment import MDPInfo
from mushroom.algorithms.policy_search import *
from mushroom.algorithms.value.td import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.features.features import *
from mushroom.features.tiles import Tiles
from mushroom.policy.gaussian_policy import *
from mushroom.policy import EpsGreedy
from mushroom.utils import spaces
from mushroom.utils.parameters import *
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from mushroom.environments import ShipSteering

from library.core.hierarchical_core import HierarchicalCore
from library.utils.callbacks.epsilon_update import EpsilonUpdate
from library.blocks.computational_graph import ComputationalGraph
from library.blocks.control_block import ControlBlock
from library.blocks.functions.pick_state import pick_state
from library.blocks.functions.rototranslate import rototranslate
from library.blocks.functions.hi_lev_extr_rew_ghavamzade import G_high
from library.blocks.functions.low_lev_extr_rew_ghavamzade import G_low
from library.blocks.reward_accumulator import reward_accumulator_block
from library.blocks.basic_operation_block import *
from library.blocks.model_placeholder import PlaceHolder
from library.blocks.mux_block import MuxBlock
from library.blocks.hold_state import hold_state
from library.blocks.discretization_block import DiscretizationBlock
from library.agents.Q_lambda_discrete import QLambdaDiscrete

class TerminationCondition(object):

    def __init__(self, active_dir):
        self.active_direction = active_dir

    def __call__(self, state):
        if self.active_direction == '+':
            goal_pos = np.array([140, 75])
        elif self.active_direction == 'x':
            goal_pos = np.array([140, 140])

        pos = np.array([state[0], state[1]])
        if np.linalg.norm(pos-goal_pos) <= 10 \
                or pos[0] > 150 or pos[0] < 0 \
                or pos[1] > 150 or pos[1] < 0:
            #if np.linalg.norm(pos-goal_pos) <= 10:
            #    print('reached ', self.active_direction)
            return True
        else:
            return False

def selector_function(inputs):
    action = np.asscalar(inputs[0])
    return 0 if action < 4 else 1


def experiment_ghavamzade(alg_high, alg_low, params, subdir, i):

    np.random.seed()

    # Model Block
    mdp = ShipSteering(small=False, n_steps_action=3)

    #State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    #Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    #Last action Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    # FeaturesH
    low_hi = 0
    lim_hi = 1000+1e-8
    n_tiles_high = [20, 20]
    n_tilings = 1
    # Discretization Block

    discretization_block = DiscretizationBlock(low=low_hi, high=lim_hi, n_tiles=n_tiles_high)

    # PolicyH
    epsilon = Parameter(value=0.1)
    piH = EpsGreedy(epsilon=epsilon)
    #beta = Parameter(value=1.0)
    #piH = Boltzmann(beta=beta)

    # AgentH
    learning_rate = params.get('learning_rate_high')


    mdp_info_agentH = MDPInfo(
        observation_space=spaces.Discrete(n_tiles_high[0]*n_tiles_high[1]),
        action_space=spaces.Discrete(8), gamma=1, horizon=10000)


    agentH = alg_high(policy=piH,
                      mdp_info=mdp_info_agentH,
                      learning_rate=learning_rate,
                      lambda_coeff=0.9)

    epsilon_update = EpsilonUpdate(piH)

    # Control Block H
    control_blockH = ControlBlock(name='control block H',
                                  agent=agentH,
                                  n_steps_per_fit=1)

    #FeaturesL
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 10]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 3

    tilingsL= Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles,
                             low=low, high=high)

    featuresL = Features(tilings=tilingsL)

    mdp_info_agentL = MDPInfo(
        observation_space=spaces.Box(low=np.array([0, 0]),
                                     high=np.array([150, 150]),
                                     shape=(2,)),
        action_space=mdp.info.action_space, gamma=0.99, horizon=10000)

    # Approximators
    input_shape = (featuresL.size,)

    approximator_params = dict(input_dim=input_shape[0])
    approximator1 = Regressor(LinearApproximator, input_shape=input_shape,
                              output_shape=mdp.info.action_space.shape,
                              **approximator_params)
    approximator2 = Regressor(LinearApproximator, input_shape=input_shape,
                              output_shape=mdp.info.action_space.shape,
                              **approximator_params)

    # Policy1
    std1 = np.array([3e-2])
    pi1 = DiagonalGaussianPolicy(mu=approximator1, std=std1)

    # Policy2
    std2 = np.array([3e-2])
    pi2 = DiagonalGaussianPolicy(mu=approximator2, std=std2)

    # Agent1
    learning_rate1 = params.get('learning_rate_low')
    agent1 = alg_low(pi1, mdp_info_agentL, learning_rate1, featuresL)

    # Agent2
    learning_rate2 = params.get('learning_rate_low')
    agent2 = alg_low(pi2, mdp_info_agentL, learning_rate2, featuresL)


    #Termination Conds
    termination_condition1 = TerminationCondition(active_dir='+')
    termination_condition2 = TerminationCondition(active_dir='x')

    low_ep_per_fit = params.get('low_ep_per_fit')

    # Control Block +
    control_block_plus = ControlBlock(name='control block 1', agent=agent1,
                                      n_eps_per_fit=low_ep_per_fit,
                                  termination_condition=termination_condition1)

    # Control Block x
    control_block_cross = ControlBlock(name='control block 2', agent=agent2,
                                       n_eps_per_fit=low_ep_per_fit,
                                  termination_condition=termination_condition2)

    # Function Block 1: picks state for hi lev ctrl
    function_block1 = fBlock(phi=pick_state, name='f1 pickstate')

    # Function Block 2: maps the env to low lev ctrl state
    function_block2 = fBlock(phi=rototranslate, name='f2 rotot')

    # Function Block 3: holds curr state as ref
    function_block3 = hold_state(name='f3 holdstate')

    # Function Block 4: adds hi lev rew
    function_block4 = addBlock(name='f4 add')

    # Function Block 5: adds low lev rew
    function_block5 = addBlock(name='f5 add')

    # Function Block 6:ext rew of hi lev ctrl
    function_block6 = fBlock(phi=G_high, name='f6 G_hi')

    # Function Block 7: ext rew of low lev ctrl
    function_block7 = fBlock(phi=G_low, name='f7 G_lo')



    #Reward Accumulator H:
    reward_acc_H = reward_accumulator_block(gamma=mdp_info_agentH.gamma,
                                            name='reward_acc_H')

    # Selector Block
    function_block8 = fBlock(phi=selector_function, name='f7 G_lo')

    #Mux_Block
    mux_block = MuxBlock(name='mux')
    mux_block.add_block_list([control_block_plus])
    mux_block.add_block_list([control_block_cross])

    #Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_blockH, mux_block,
              function_block1, function_block2, function_block3,
              function_block4, function_block5,
              function_block6, function_block7, function_block8,
              reward_acc_H, discretization_block]

    reward_acc_H.add_input(reward_ph)
    reward_acc_H.add_alarm_connection(control_block_plus)
    reward_acc_H.add_alarm_connection(control_block_cross)

    control_blockH.add_input(discretization_block)
    control_blockH.add_reward(function_block4)
    control_blockH.add_alarm_connection(control_block_plus)
    control_blockH.add_alarm_connection(control_block_cross)

    mux_block.add_input(function_block8)
    mux_block.add_input(function_block2)

    control_block_plus.add_reward(function_block5)
    control_block_cross.add_reward(function_block5)

    function_block1.add_input(state_ph)

    function_block2.add_input(control_blockH)
    function_block2.add_input(state_ph)
    function_block2.add_input(function_block3)

    function_block3.add_input(state_ph)
    function_block3.add_alarm_connection(control_block_plus)
    function_block3.add_alarm_connection(control_block_cross)

    function_block4.add_input(function_block6)
    function_block4.add_input(reward_acc_H)

    function_block5.add_input(reward_ph)
    #function_block5.add_input(function_block7)

    function_block6.add_input(reward_ph)

    function_block7.add_input(control_blockH)
    function_block7.add_input(function_block2)

    function_block8.add_input(control_blockH)

    discretization_block.add_input(function_block1)


    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    low_level_dataset_eval1 = list()
    low_level_dataset_eval2 = list()
    dataset_eval = list()

    dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
    dataset_eval += dataset_eval_run
    print('J at start : ' + str(np.mean(J)))
    for n in range(n_runs):
        print('ITERATION', n)

        core.learn(n_episodes=n_iterations*ep_per_run, skip=True)
        dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
        J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
        print('J at iteration ' + str(n) + ': ' + str(np.mean(J)))
        dataset_eval += dataset_eval_run

        dataset_plus = control_block_plus.dataset.get()
        J_plus = compute_J(dataset_plus, mdp.info.gamma)
        dataset_cross = control_block_cross.dataset.get()
        J_cross = compute_J(dataset_cross, mdp.info.gamma)

        low_level_dataset_eval1.append(dataset_plus)
        low_level_dataset_eval2.append(dataset_cross)

        print('J ll PLUS at iteration  ' + str(n) + ': ' + str(np.mean(J_plus)))
        print('J ll CROSS at iteration ' + str(n) + ': ' + str(np.mean(J_cross)))
        if n == 4:
            control_blockH.callbacks = [epsilon_update]




    # Tile data
    hi_lev_params = agentH.Q.table
    max_q_val = np.zeros(n_tiles_high[0]**2)
    act_max_q_val = np.zeros(n_tiles_high[0]**2)
    for n in range(n_tiles_high[0]**2):
        max_q_val[n] = np.amax(hi_lev_params[n])
        act_max_q_val[n] = np.argmax(hi_lev_params[n])
    #max_q_val_tiled = np.reshape(max_q_val, (n_tiles_high[0], n_tiles_high[1]))
    #act_max_q_val_tiled = np.reshape(act_max_q_val, (n_tiles_high[0],
    #                                                 n_tiles_high[1]))

    mk_dir_recursive('./' + subdir + str(i))

    np.save(subdir+str(i)+'/low_level_dataset1_file', low_level_dataset_eval1)
    np.save(subdir+str(i)+'/low_level_dataset2_file', low_level_dataset_eval2)
    np.save(subdir+str(i)+'/max_q_val_tiled_file', max_q_val)
    np.save(subdir+str(i)+'/act_max_q_val_tiled_file', act_max_q_val)
    np.save(subdir+str(i)+'/dataset_eval_file', dataset_eval)

    return


if __name__ == '__main__':

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') \
             + '_big_ghavamzade/'
    alg_high = QLambdaDiscrete
    alg_low = GPOMDP
    learning_rate_high = Parameter(value=8e-2)
    learning_rate_low = AdaptiveParameter(value=1e-2)
    n_jobs = -1
    how_many = 100
    n_runs = 25
    n_iterations = 20
    ep_per_run = 40
    low_ep_per_fit = 50
    mk_dir_recursive('./' + subdir)
    force_symlink('./' + subdir, './latest_big_ghavamzade')

    params = {'learning_rate_high': learning_rate_high,
              'learning_rate_low': learning_rate_low,
              'low_ep_per_fit': low_ep_per_fit}
    experiment_params = {'how_many': how_many,
                         'n_runs': n_runs,
                         'n_iterations': n_iterations,
                         'ep_per_run': ep_per_run}
    np.save(subdir + '/experiment_params_dictionary', experiment_params)
    Js = Parallel(n_jobs=n_jobs)(delayed(experiment_ghavamzade)
                                 (alg_high, alg_low, params,
                                  subdir, i) for i in range(how_many))