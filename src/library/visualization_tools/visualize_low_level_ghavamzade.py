import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_low_level_ghavamzade(datalist_eval, success_per_thousand_eps, small=False, how_many=1):




    plt.figure()

    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))


    x_ep = list()
    y_ep = list()
    theta_ep = list()
    thetadot_ep = list()
    action_ep = list()
    reward_ep = list()
    size_eps = list()
    x_list = list()
    y_list = list()
    theta_list = list()
    thetadot_list = list()
    reward_list = list()
    ep_size = 0
    n_eps = 0

    for dataset_step in datalist_eval:
        if not dataset_step[-1]:
            states_step = dataset_step[0]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            x_step = states_step[0]
            y_step = states_step[1]
            theta_step = states_step[2]
            thetadot_step = states_step[3]

            x_ep.append(x_step)
            y_ep.append(y_step)
            theta_ep.append(theta_step)
            thetadot_ep.append(thetadot_step)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            ep_size += 1
        else:
            size_eps.append(ep_size)
            ep_size = 0
            x_ep.append(dataset_step[3][0])
            y_ep.append(dataset_step[3][1])
            theta_ep.append(dataset_step[3][2])
            thetadot_ep.append(dataset_step[3][3])
            x_list.append(x_ep)
            y_list.append(y_ep)
            theta_list.append(theta_ep)
            thetadot_list.append(thetadot_ep)
            reward_list.append(reward_ep)
            x_ep = []
            y_ep = []
            theta_ep = []
            thetadot_ep = []
            action_ep = []
            reward_ep = []
            n_eps += 1


    for episode in xrange(len(x_list)):
        x = x_list[episode]
        y = y_list[episode]
        ax1.plot(x,y)
        ax3.plot(reward_list[episode])
    ax2.plot(success_per_thousand_eps)

    plt.show()