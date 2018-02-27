import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_control_block_ghavamzade(datalist_control, J = None, ep_count = None):

    plt.figure()

    ax1 = plt.subplot2grid((4,2), (0,0))
    ax2 = plt.subplot2grid((4,2), (1,0))
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
    action_list = list()
    reward_list = list()
    next_state_list = list()

    dataset_ep = list()
    state_ep = list()
    x_ep = list()
    y_ep = list()
    theta_ep = list()
    action_ep = list()
    reward_ep = list()
    size_eps = list()
    next_state_ep = list()

    i = 0
    n_eps = 0

    for dataset_step in datalist_control:
        if not dataset_step[-1]:
            state_step = dataset_step[0]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            next_state_step = dataset_step[3]

            dataset_ep.append(dataset_step)
            state_ep.append(state_step)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            next_state_ep.append(next_state_step)
            i += 1

        else:
            dataset_ep.append(dataset_step)
            state_step = dataset_step[0]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            next_state_step = dataset_step[3]

            state_ep.append(state_step)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            next_state_ep.append(next_state_step)
            i += 1
            size_eps.append(i)

            i=0
            dataset_list.append(dataset_ep)
            state_list.append(state_ep)
            reward_list.append(reward_ep)
            action_list.append(action_ep)
            next_state_list.append(next_state_ep)
            state_ep = []
            reward_ep = []
            action_ep = []
            next_state_ep = []
            n_eps += 1

    if ep_count is None:
        ep_count = n_eps

    range_eps = xrange(n_eps-ep_count,n_eps)

    '''for dataset_ep in dataset_list[n_eps-ep_count:n_eps]:
        print '-------------------------------------------------------------------------'
        for dataset_step in dataset_ep:
            print dataset_step
'''
    for episode in range_eps:
        state_ep = np.array(state_list[episode])
        x_ep = state_ep[:, 0]
        ax1.plot(x_ep)
        ax1.set_ylabel('x')

    for episode in range_eps:
        state_ep = np.array(state_list[episode])
        y_ep = state_ep[:, 1]
        ax2.plot(y_ep)
        ax2.set_ylabel('y')

    for episode in range_eps:
        state_ep = np.array(state_list[episode])
        theta_ep = state_ep[:, 2]
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
        #theoretical_reward =[-x**2 for x in next_state_list[episode]]
        #ax3.plot(time, theoretical_reward)
        ax5.plot(reward_ep)
        ax5.set_ylabel('reward')


    ax6.plot(size_eps)
    ax6.set_title('size_eps')


    plt.tight_layout()
