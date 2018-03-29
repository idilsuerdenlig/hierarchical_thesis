import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mushroom.utils.dataset import compute_J
from library.utils.pick_last_ep_dataset import pick_last_ep
from library.utils.pick_eps import pick_eps



def visualize_control_block_ghavamzade(datalist_control, ep_count = None, gamma=1, n_runs=1, ep_per_run=1):

    plt.figure()

    ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2)
    ax3 = plt.subplot2grid((4,2), (2,0))
    ax4 = plt.subplot2grid((4,2), (0,1))
    ax5 = plt.subplot2grid((4,2), (1,1))
    ax6 = plt.subplot2grid((4,2), (2,1))
    ax7 = plt.subplot2grid((4,2), (3,0))


    dataset_list = list()
    state_list = list()
    x_list = list()
    y_list = list()
    theta_list =list()
    theta_dot_list = list()
    action_list = list()
    reward_list = list()
    next_state_list = list()

    dataset_ep = list()
    state_ep = list()
    x_ep = list()
    y_ep = list()
    theta_ep = list()
    theta_dot_ep = list()
    action_ep = list()
    reward_ep = list()
    size_eps = list()
    next_state_ep = list()
    datalist_control_vis = list()
    J_avg = np.zeros((n_runs,))
    size_eps_avg = np.zeros((n_runs,))
    J_exp = []
    size_eps = list()
    size_exp_list = list()
    ep_step_no = 0
    i = 0
    n_eps = 0


    for run in range(n_runs):
        dataset_control_run = datalist_control[run]
        last_ep_of_run = pick_last_ep(dataset_control_run)
        for step in last_ep_of_run:
            datalist_control_vis.append(step)

    for i in range(n_runs):
        J_runs_eps= compute_J(datalist_control[run], gamma)
        J_avg[i] = np.mean(J_runs_eps[ep_per_run * i:ep_per_run * i + ep_per_run], axis=0)
    fig = plt.figure()
    ax1n = fig.add_subplot(121)
    ax2n = fig.add_subplot(122)
    ax1n.plot(J_avg)
    ax1n.set_title('J_eps_averaged')

    for dataset_step in datalist_control_vis:
        if not dataset_step[-1]:
            state_step = dataset_step[0]
            x_step = state_step[0]
            y_step = state_step[1]
            theta_step = state_step[2]
            theta_dot_step = state_step[3]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            next_state_step = dataset_step[3]

            dataset_ep.append(dataset_step)
            state_ep.append(state_step)
            x_ep.append(x_step)
            y_ep.append(y_step)
            theta_ep.append(theta_step)
            theta_dot_ep.append(theta_dot_step)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            next_state_ep.append(next_state_step)
            i += 1

        else:
            dataset_ep.append(dataset_step)
            state_step = dataset_step[0]
            x_step = state_step[0]
            y_step = state_step[1]
            theta_step = state_step[2]
            theta_dot_step = state_step[3]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            next_state_step = dataset_step[3]

            state_ep.append(state_step)
            x_ep.append(x_step)
            y_ep.append(y_step)
            theta_ep.append(theta_step)
            theta_dot_ep.append(theta_dot_step)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            next_state_ep.append(next_state_step)
            i += 1
            size_eps.append(i)

            i = 0
            dataset_list.append(dataset_ep)
            state_list.append(state_ep)
            x_list.append(x_ep)
            y_list.append(y_ep)
            theta_list.append(theta_ep)
            theta_dot_list.append(theta_dot_ep)
            reward_list.append(reward_ep)
            action_list.append(action_ep)
            next_state_list.append(next_state_ep)
            state_ep = []
            x_ep = []
            y_ep = []
            theta_ep = []
            theta_dot_ep = []
            reward_ep = []
            action_ep = []
            next_state_ep = []
            n_eps += 1

    if ep_count is None:
        ep_count = n_eps

    range_eps = range(n_eps-ep_count, n_eps)

    for episode in range_eps:
        x_ep = np.array(x_list[episode])
        y_ep = np.array(y_list[episode])
        ax1.plot(x_ep, y_ep)
        ax1.set_title('trajectories')

    for episode in range_eps:
        theta_ep = np.array(theta_list[episode])
        ax3.plot(theta_ep)
        ax3.set_ylabel('theta')

    for episode in range_eps:
        state_ep = np.array(state_list[episode])
        theta_dot_ep = state_ep[:, 3]
        ax7.plot(theta_dot_ep)
        ax7.set_ylabel('theta_dot')

    for episode in range_eps:
        action_ep = action_list[episode]
        ax4.plot(action_ep)
        ax4.set_ylabel('action')

    for episode in range_eps:
        reward_ep = reward_list[episode]
        ax5.plot(reward_ep)
        ax5.set_ylabel('reward')


    ax2n.plot(size_eps_avg)
    ax2n.set_title('size_eps')




    plt.tight_layout()
