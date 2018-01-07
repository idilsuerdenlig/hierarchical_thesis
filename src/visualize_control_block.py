import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def VisualizeControlBlock(datalist_control, J=None):

    plt.figure()

    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))



    state_list = list()
    action_list = list()
    reward_list = list()
    next_state_list = list()
    state_ep = list()
    action_ep = list()
    reward_ep = list()
    size_eps = list()
    next_state_ep = list()

    i = 0
    for dataset_step in datalist_control:
        if not dataset_step[-1]:
            state_step = dataset_step[0]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            next_state_step = dataset_step[3]

            state_ep.append(state_step)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            next_state_ep.append(next_state_step)
            i += 1

        else:
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
            state_list.append(state_ep)
            reward_list.append(reward_ep)
            action_list.append(action_ep)
            next_state_list.append(next_state_ep)
            state_ep = []
            reward_ep = []
            action_ep = []
            next_state_ep = []

    for episode in xrange(len(state_list)):
        state_ep = state_list[episode]
        time =np.arange(len(state_ep))
        ax1.plot(time,state_ep)
        ax1.set_ylabel('state')

    for episode in xrange(len(action_list)):
        action_ep = action_list[episode]
        time =np.arange(len(action_ep))
        ax2.plot(time,action_ep)
        ax2.set_ylabel('action')

    for episode in xrange(len(reward_list)):
        reward_ep = reward_list[episode]
        time =np.arange(len(reward_ep))
        #theoretical_reward =[-x**2 for x in next_state_list[episode]]
        #ax3.plot(time, theoretical_reward)
        ax3.plot(time,reward_ep)
        ax3.set_ylabel('reward')


    ax4.plot(size_eps)
    ax4.set_title('size_eps')


    plt.tight_layout()
