import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_control_block(datalist_control, J = None, ep_count = None):

    plt.figure()

    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))


    dataset_list = list()
    state_list = list()
    action_list = list()
    reward_list = list()
    next_state_list = list()
    dataset_ep = list()
    state_ep = list()
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
            state_ep.append((state_step/np.pi)*180)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            next_state_ep.append((next_state_step/np.pi)*180)
            i += 1

        else:
            state_step = dataset_step[0]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            next_state_step = dataset_step[3]

            dataset_ep.append(dataset_step)
            state_ep.append((state_step/np.pi)*180)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            next_state_ep.append((next_state_step/np.pi)*180)
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




    for episode in range_eps:
        state_ep = state_list[episode]
        time =np.arange(len(state_ep))
        ax1.plot(time, state_ep)
        #ax1.plot(time, state_ep[:,1])
        ax1.set_ylabel('state')

    for episode in range_eps:
        action_ep = action_list[episode]
        time =np.arange(len(action_ep))
        ax2.plot(time,action_ep)
        ax2.set_ylabel('action')

    for episode in range_eps:
        reward_ep = reward_list[episode]
        time =np.arange(len(reward_ep))
        #theoretical_reward =[-x**2 for x in next_state_list[episode]]
        #ax3.plot(time, theoretical_reward)
        ax3.plot(time,reward_ep)
        ax3.set_ylabel('reward')

    time = np.arange(len(size_eps))
    ax4.plot(time, size_eps)
    ax4.set_title('size_eps')


    plt.tight_layout()
