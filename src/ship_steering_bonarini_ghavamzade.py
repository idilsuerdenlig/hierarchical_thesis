from library.core.hierarchical_core import HierarchicalCore
from library.blocks.computational_graph import ComputationalGraph
from library.blocks.control_block import ControlBlock
from mushroom.utils import spaces
from mushroom.utils.parameters import *
from mushroom.utils.callbacks import CollectDataset
from mushroom.features.features import *
from mushroom.features.basis import PolynomialBasis
from mushroom.policy.gaussian_policy import *
from mushroom.algorithms.policy_search import *
from library.utils.callbacks.collect_policy_parameter import CollectPolicyParameter
from library.blocks.basic_operation_block import *
from library.blocks.model_placeholder import PlaceHolder
from library.blocks.mux_block import MuxBlock
from mushroom.algorithms.value.td import *
from mushroom.policy import EpsGreedy
from mushroom.features.tiles import Tiles
from library.blocks.functions.pick_state import pick_state
from library.blocks.functions.rototranslate import rototranslate
from library.blocks.hold_state import hold_state
from library.blocks.functions.hi_lev_extr_rew_ghavamzade import G_high
from library.blocks.functions.low_lev_extr_rew_ghavamzade import G_low
from library.blocks.reward_accumulator import reward_accumulator_block
from mushroom.utils.dataset import compute_J
import datetime
import argparse
from mushroom.utils.folder import mk_dir_recursive
from library.approximator.CMAC import CMACApproximator
from library.environments.bonarinishipsteering import BonariniShipSteering
from mushroom.environments.environment import MDPInfo
from library.agents.ghavamzade_agent import GhavamzadeAgent
from library.utils.pick_last_ep_dataset import pick_last_ep
from library.blocks.hold_state import hold_state
import datetime
import argparse
from mushroom.utils.folder import *
from joblib import Parallel, delayed

class TerminationCondition(object):

    def __init__(self, active_dir, small=True):
        self.active_direction = active_dir
        self.small = small

    def __call__(self, state):
        if self.active_direction <= 4:
            goal_pos = np.array([140, 75])
        else:
            goal_pos = np.array([140, 140])

        pos = np.array([state[0], state[1]])
        if np.linalg.norm(pos-goal_pos) <= 10 or pos[0] > 150 or pos[0] < 0 or pos[1] > 150 or pos[1] < 0:
            return True
        else:
            return False


def experiment_bonarini_ghavamzade(alg_high, alg_low, params, experiment_params ,subdir, i):

    np.random.seed()

    # Model Block
    small=False
    mdp = BonariniShipSteering(small=False, hard=False, n_steps_action=3)

    #State Placeholder
    state_ph = PlaceHolder(name='state_ph')

    #Reward Placeholder
    reward_ph = PlaceHolder(name='reward_ph')

    #Last action Placeholder
    lastaction_ph = PlaceHolder(name='lastaction_ph')

    #FeaturesH
    lim = 1000
    n_tiles_high = [20, 20]
    n_tilings = 1

    tilingsH= Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles_high, low=[0,0], high=[lim, lim])
    featuresH = Features(tilings=tilingsH)

    # PolicyH
    epsilon = LinearDecayParameter(value=0.1, min_value=0.0, n=10000)
    piH = EpsGreedy(epsilon=epsilon)

    # AgentH
    learning_rate = params.get('learning_rate_high')


    mdp_info_agentH = MDPInfo(observation_space=spaces.Box(low=np.array([0, 0]),
                                                           high=np.array([lim, lim]), shape=(2,)),
                              action_space=spaces.Discrete(8), gamma=1, horizon=10000)
    approximator_paramsH = dict(input_shape=(featuresH.size,),
                               output_shape=mdp_info_agentH.action_space.size,
                               n_actions=mdp_info_agentH.action_space.n)

    agentH = alg_high(policy=piH, mdp_info=mdp_info_agentH, learning_rate=learning_rate,
                                   lambda_coeff=0.9, approximator_params=approximator_paramsH, features=featuresH)

    # Control Block H
    control_blockH = ControlBlock(name='control block H', agent=agentH, n_steps_per_fit=1)

    #FeaturesL
    high = [150, 150, np.pi, np.pi/12]
    low = [0, 0, -np.pi, -np.pi/12]
    n_tiles = [5, 5, 36, 5]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 9

    tilingsL= Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low, high=high)

    featuresL = Features(tilings=tilingsL)

    # Policy1
    input_shape = (featuresL.size,)

    approximator_params = dict(input_dim=input_shape[0])
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape,
                             **approximator_params)
    sigma = np.array([[1.3e-2]])
    pi1 = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

    # Policy2
    pi2 = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

    # Agent1
    learning_rate1 = params.get('learning_rate_low')
    agent1 = alg_low(pi1, mdp.info, learning_rate1, featuresL)

    # Agent2
    learning_rate2 = params.get('learning_rate_low')
    agent2 = alg_low(pi2, mdp.info, learning_rate2, featuresL)


    #Termination Conds
    termination_condition1 = TerminationCondition(active_dir=1, small=small)
    termination_condition2 = TerminationCondition(active_dir=5, small=small)

    # Control Block +
    control_block1 = ControlBlock(name='control block 1', agent=agent1, n_steps_per_fit=1,
                                  termination_condition=termination_condition1)

    # Control Block x
    control_block2 = ControlBlock(name='control block 2', agent=agent2, n_steps_per_fit=1,
                                  termination_condition=termination_condition2)

    # Function Block 1: picks state for hi lev ctrl
    function_block1 = fBlock(phi=pick_state, name='f1 pickstate')

    # Function Block 2: maps the env to low lev ctrl state
    function_block2 = fBlock(phi=rototranslate(small=small), name='f2 rotot')

    # Function Block 3: holds curr state as ref
    function_block3 = hold_state(name='f3 holdstate')

    # Function Block 4: adds hi lev rew
    function_block4 = addBlock(name='f4 add')

    # Function Block 5: adds low lev rew
    function_block5 = addBlock(name='f5 add')

    # Function Block 6:ext rew of hi lev ctrl
    function_block6 = fBlock(phi=G_high, name='f6 G_hi')

    # Function Block 7: ext rew of low lev ctrl
    function_block7 = fBlock(phi=G_low(small=small), name='f7 G_lo')

    #Reward Accumulator H:
    reward_acc_H = reward_accumulator_block(gamma=mdp_info_agentH.gamma, name='reward_acc_H')

    #Mux_Block
    mux_block = MuxBlock(name='mux')
    mux_block.add_block_list([control_block1])
    mux_block.add_block_list([control_block2])

    #Algorithm
    blocks = [state_ph, reward_ph, lastaction_ph, control_blockH, mux_block,
              function_block1, function_block2, function_block3,
              function_block4, function_block5,
              function_block6, function_block7, reward_acc_H]

    #state_ph.add_input(mux_block)
    #reward_ph.add_input(mux_block)
    #lastaction_ph.add_input(mux_block)
    reward_acc_H.add_input(reward_ph)
    reward_acc_H.add_alarm_connection(control_block1)
    reward_acc_H.add_alarm_connection(control_block2)
    control_blockH.add_input(function_block1)
    control_blockH.add_reward(function_block4)
    control_blockH.add_alarm_connection(control_block1)
    control_blockH.add_alarm_connection(control_block2)
    mux_block.add_input(control_blockH)
    mux_block.add_input(function_block2)
    control_block1.add_reward(function_block5)
    control_block2.add_reward(function_block5)
    function_block1.add_input(state_ph)
    function_block2.add_input(control_blockH)
    function_block2.add_input(state_ph)
    function_block2.add_input(function_block3)
    function_block3.add_input(state_ph)
    function_block3.add_alarm_connection(control_block1)
    function_block3.add_alarm_connection(control_block2)
    function_block4.add_input(function_block6)
    function_block4.add_input(reward_acc_H)
    function_block5.add_input(reward_ph)
    function_block5.add_input(function_block7)
    function_block6.add_input(reward_ph)
    function_block7.add_input(control_blockH)
    function_block7.add_input(function_block2)


    computational_graph = ComputationalGraph(blocks=blocks, model=mdp)
    core = HierarchicalCore(computational_graph)

    # Train
    dataset_eval_visual = list()
    low_level_dataset_eval1 = list()
    low_level_dataset_eval2 = list()
    dataset_eval = list()

    dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
    last_ep_dataset = pick_last_ep(dataset_eval_run)
    dataset_eval_visual += last_ep_dataset
    dataset_eval += dataset_eval_run
    print('J at start : ' + str(np.mean(J)))


    for n in range(n_runs):
        print('ITERATION', n)
        core.learn(n_episodes=n_iterations*ep_per_run, skip=True)
        dataset_eval_run = core.evaluate(n_episodes=ep_per_run)
        J = compute_J(dataset_eval_run, gamma=mdp.info.gamma)
        print('J at iteration ' + str(n) + ': ' + str(np.mean(J)))
        last_ep_dataset = pick_last_ep(dataset_eval)
        dataset_eval_visual += last_ep_dataset
        low_level_dataset_eval1 += control_block1.dataset.get()
        low_level_dataset_eval2 += control_block2.dataset.get()



    # Tile data
    hi_lev_params = agentH.Q.get_weights()
    hi_lev_params = np.reshape(hi_lev_params, (8, n_tiles_high[0]**2))
    max_q_val = np.zeros(shape=(n_tiles_high[0]**2,))
    act_max_q_val = np.zeros(shape=(n_tiles_high[0]**2,))
    for n in range(n_tiles_high[0]**2):
        max_q_val[n] = np.amax(hi_lev_params[:,n])
        act_max_q_val[n] = np.argmax(hi_lev_params[:,n])
    max_q_val_tiled = np.reshape(max_q_val, (n_tiles_high[0], n_tiles_high[1]))
    act_max_q_val_tiled = np.reshape(act_max_q_val, (n_tiles_high[0], n_tiles_high[1]))

    mk_dir_recursive('./' + subdir + str(i))


    np.save(subdir+str(i)+'/low_level_dataset1_file', low_level_dataset_eval1)
    np.save(subdir+str(i)+'/low_level_dataset2_file', low_level_dataset_eval2)
    np.save(subdir+str(i)+'/max_q_val_tiled_file', max_q_val_tiled)
    np.save(subdir+str(i)+'/act_max_q_val_tiled_file', act_max_q_val_tiled)
    np.save(subdir+str(i)+'/dataset_eval_file', dataset_eval_visual)

    return


if __name__ == '__main__':

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_bonarini_ghavamzade/'
    alg_high = TrueOnlineSARSALambda
    alg_low = GhavamzadeAgent
    learning_rate_high = Parameter(value=0.2)
    learning_rate_low = ExponentialDecayParameter(value=1e-8, decay_exp=0.5)
    how_many = 1
    n_runs = 2
    n_iterations = 2
    ep_per_run = 5
    mk_dir_recursive('./' + subdir)

    params = {'learning_rate_high': learning_rate_high, 'learning_rate_low': learning_rate_low}
    experiment_params = {'how_many': how_many, 'n_runs': n_runs,
                         'n_iterations': n_iterations, 'ep_per_run': ep_per_run}
    np.save(subdir + '/experiment_params_dictionary', experiment_params)
    Js = Parallel(n_jobs=1)(delayed(experiment_bonarini_ghavamzade)(alg_high, alg_low, params,
                                                                    experiment_params,
                                                                    subdir, i) for i in range(how_many))