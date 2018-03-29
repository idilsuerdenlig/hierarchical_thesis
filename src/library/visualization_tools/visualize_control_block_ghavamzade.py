import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mushroom.utils.dataset import compute_J
from library.utils.pick_last_ep_dataset import pick_last_ep
from library.utils.pick_eps import pick_eps



def visualize_control_block_ghavamzade(datalist_control, ep_count = None, gamma=1, n_runs=1, ep_per_run=1, name=None):

    plt.figure()

    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax4 = plt.subplot2grid((3, 2), (0, 1))
    ax5 = plt.subplot2grid((3, 2), (1, 1))
    ax7 = plt.subplot2grid((3, 2), (2, 1))

    datalist_control_vis = list()
    J_avg = np.zeros((n_runs,))
    size_eps = list()

    for run in range(n_runs):
        dataset_control_run = datalist_control[run]
        last_ep_of_run = pick_last_ep(dataset_control_run)
        datalist_control_vis.append(last_ep_of_run)

    for i in range(n_runs):
        J_runs_eps= compute_J(datalist_control[run], gamma)
        J_avg[i] = np.mean(J_runs_eps, axis=0)

    fig = plt.figure()
    ax1n = fig.add_subplot(121)
    ax2n = fig.add_subplot(122)
    ax1n.plot(J_avg)
    ax1n.set_title('J_eps_averaged')

    x_ep = list()
    y_ep = list()
    theta_ep = list()
    theta_dot_ep = list()

    for episode in datalist_control_vis:

        state_ep = episode[:,0]
        for step_ep in state_ep:
            x_ep.append(step_ep[0])
            y_ep.append(step_ep[1])
            theta_ep.append(step_ep[2])
            theta_dot_ep.append(step_ep[3])

        action_ep = episode[:, 1]
        reward_ep = episode[:, 2]
        size_eps.append(len(episode))
        ax1.plot(x_ep, y_ep)
        ax1.scatter(x_ep[0], y_ep[0], marker='o', color='g')
        ax1.scatter(x_ep[-1], y_ep[-1],  marker='o', color='r')
        ax1.set_xlim(0, 150)
        ax1.set_ylim(0, 150)
        ax1.set_title('trajectories '+name)
        ax3.plot(theta_ep)
        ax3.set_ylabel('theta')
        ax7.plot(theta_dot_ep)
        ax7.set_ylabel('theta_dot')
        ax4.plot(action_ep)
        ax4.set_ylabel('action')
        ax5.plot(reward_ep)
        ax5.set_ylabel('reward')
        size_eps.append(len(state_ep))


    ax2n.plot(size_eps)
    ax2n.set_title('size_eps')




    plt.tight_layout()
