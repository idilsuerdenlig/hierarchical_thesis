import numpy as np
import matplotlib.pyplot as plt
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from matplotlib.patches import Circle


def check_no_of_eps(dataset):
    no_of_eps = 0

    for dataset_step in dataset:
        if dataset_step[-1]:
            no_of_eps += 1
    return no_of_eps

def pick_eps(dataset, start, end):

    dataset_ep = list()
    dataset_ep_list = list()
    for dataset_step in dataset:
        if not dataset_step[-1]:
            dataset_ep.append(dataset_step)
        else:
            dataset_ep.append(dataset_step)
            dataset_ep_list.append(dataset_ep)
            dataset_ep = list()
    return dataset_ep_list[start:end]


def pick_last_ep(dataset):

    dataset_ep = list()
    dataset_ep_list = list()
    for dataset_step in dataset:
        if not dataset_step[-1]:
            dataset_ep.append(dataset_step)
        else:
            dataset_ep.append(dataset_step)
            dataset_ep_list.append(dataset_ep)
            dataset_ep = list()
    return dataset_ep_list[-1]


def pick_first_ep(dataset):
    dataset_ep = list()
    for dataset_step in dataset:
        if not dataset_step[-1]:
            dataset_ep.append(dataset_step)
        else:
            dataset_ep.append(dataset_step)
            break
    return dataset_ep

def plot_arrows(act_max_q_val_tiled):
    plt.figure()
    ax = plt.axes()
    n_tiles = [act_max_q_val_tiled.shape[0], act_max_q_val_tiled.shape[1]]
    ax.set_xlim([0, n_tiles[0]*50])
    ax.set_ylim([0, n_tiles[1]*50])

    for i in range(n_tiles[0]):
        for j in range(n_tiles[1]):
            xs = 25 + i*50
            ys = 25 + j*50
            act_no = act_max_q_val_tiled[i, j]
            if act_no == 0:
                ax.arrow(xs, ys, 12, 0, head_width=5, head_length=2, fc='k', ec='k')

            elif act_no == 1:
                ax.arrow(xs, ys, 0, -12, head_width=5, head_length=2, fc='k', ec='k')

            elif act_no == 2:
                ax.arrow(xs, ys, -12, 0, head_width=5, head_length=2, fc='k', ec='k')

            elif act_no == 3:
                ax.arrow(xs, ys, 0, 12, head_width=5, head_length=2, fc='k', ec='k')

            elif act_no == 4:
                ax.arrow(xs, ys, 15, 15, head_width=5, head_length=2, fc='k', ec='k')

            elif act_no == 5:
                ax.arrow(xs, ys, 15, -15, head_width=5, head_length=2, fc='k', ec='k')

            elif act_no == 6:
                ax.arrow(xs, ys, -15, -15, head_width=5, head_length=2, fc='k', ec='k')

            else:
                ax.arrow(xs, ys, -15, 15, head_width=5, head_length=2, fc='k', ec='k')


def visualize_control_block_ghavamzade(controller_dataset_vis, J_avg, ep_step_avg, name, epochs=None):

    plt.figure()

    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((3, 3), (2, 0))
    ax3 = plt.subplot2grid((3, 3), (2, 1))
    ax4 = plt.subplot2grid((3, 3), (0, 2))
    ax5 = plt.subplot2grid((3, 3), (1, 2))

    for episode in controller_dataset_vis:
        x_ep = list()
        y_ep = list()
        theta_ep = list()
        theta_dot_ep = list()
        action_ep = list()
        reward_ep = list()

        for step in episode:
            x_ep.append(step[0][0])
            y_ep.append(step[0][1])
            theta_ep.append(step[0][2])
            theta_dot_ep.append(step[0][3])
            action_ep.append(step[1])
            reward_ep.append(step[2])
        ax1.plot(x_ep, y_ep)
        ax1.scatter(x_ep[0], y_ep[0], marker='o', color='g')
        ax2.plot(theta_ep)
        ax2.plot(theta_ep)
        ax3.plot(theta_dot_ep)
        ax4.plot(action_ep)
        ax5.plot(reward_ep)
        ax1.scatter(x_ep[-1], y_ep[-1], marker='o', color='r')
        
        ax1.set_xlim(0, 150)
        ax1.set_ylim(0, 150)
        ax1.set_title('trajectories '+name)
        ax2.set_ylabel('theta')
        ax3.set_ylabel('theta_dot')
        ax4.set_ylabel('action')
        ax5.set_ylabel('reward')

    x_llo = 140

    if name == 'ctrl+':
        y_llo = 75
    else:
        y_llo = 140

    goal_area = Circle(xy=(x_llo, y_llo), radius=10,
                               edgecolor='k', fc='None', lw=2)
    ax1.add_patch(goal_area)

    if epochs is not None:
        ax1.legend(epochs)


    plt.figure()
    plt.plot(J_avg)
    plt.title('J_avg' + name)

    plt.figure()
    plt.plot(ep_step_avg)
    plt.title('ep_step_avg'+name)


    plt.tight_layout()


def visualize_traj(dataset_eval_vis, name):

    plt.figure()

    for dataset_ep in dataset_eval_vis:
        x_ep = list()
        y_ep = list()

        for step in dataset_ep:

            x_step = step[0][0]
            y_step = step[0][1]

            x_ep.append(x_step)
            y_ep.append(y_step)

        plt.plot(x_ep, y_ep)


    xs = 350
    xe = 450
    ys = 400
    ye = 400

    xg = [xs, xe]
    yg = [ys, ye]
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)

    plt.plot(xg, yg)
    plt.title(name)



def ghavamzade_plot(epochs):

    dir = 'latest/'
    experiment_params = np.load(dir + 'experiment_params_dictionary.npy')

    how_many = experiment_params.item().get('how_many')
    n_epochs = experiment_params.item().get('n_runs')
    ep_per_run = experiment_params.item().get('ep_per_run')

    act_max_q_val_tiled = np.load('latest/' + str(how_many-1) + '/act_max_q_val_tiled_file.npy')
    max_q_val_tiled = np.load('latest/' + str(how_many-1) + '/max_q_val_tiled_file.npy')
    plot_arrows(act_max_q_val_tiled)
    fig, axis = plt.subplots()
    heatmap = axis.pcolor(max_q_val_tiled, cmap=plt.cm.Blues)
    plt.colorbar(heatmap)

    x = np.arange(20)
    labels = np.arange(0, 1000, 50)
    plt.xticks(x, labels)
    plt.yticks(x, labels)

    dataset_eval = np.load('latest/'+str(how_many-1)+'/dataset_eval_file.npy')
    for run in epochs:
        dataset_eval_vis = list()
        dataset_eval_epoch = pick_eps(dataset_eval, start=run * ep_per_run, end=run * ep_per_run + ep_per_run)
        dataset_eval_vis += dataset_eval_epoch[30:40]
        visualize_traj(dataset_eval_vis, 'last 10 trajectories of epoch_'+ str(run))

    low_level_dataset1 = np.load('latest/'+str(how_many-1)+'/low_level_dataset1_file.npy')
    low_level_dataset2 = np.load('latest/'+str(how_many-1)+'/low_level_dataset2_file.npy')
    low_level_dataset1_vis = list()
    low_level_dataset2_vis = list()
    eps_step_avg1 = list()
    eps_step_avg2 = list()
    J_avg1 = np.zeros(n_epochs)
    J_avg2 = np.zeros(n_epochs)

    for run in epochs:
        low_level_dataset1_run = low_level_dataset1[run]
        low_level_dataset2_run = low_level_dataset2[run]
        last_ep_of_run1 = pick_first_ep(low_level_dataset1_run)
        last_ep_of_run2 = pick_first_ep(low_level_dataset2_run)
        low_level_dataset1_vis.append(last_ep_of_run1)
        low_level_dataset2_vis.append(last_ep_of_run2)

    for run in range(n_epochs):
        low_level_dataset1_run = low_level_dataset1[run]

        low_level_dataset2_run = low_level_dataset2[run]
        J_runs_eps1= compute_J(low_level_dataset1_run, 0.99)
        J_runs_eps2 = compute_J(low_level_dataset2_run, 0.99)
        total_steps1 = len(low_level_dataset1_run)
        total_steps2 = len(low_level_dataset2_run)

        n_eps1 = len(J_runs_eps1)
        n_eps2 = len(J_runs_eps2)
        eps_step_avg1.append(total_steps1 // n_eps1)
        eps_step_avg2.append(total_steps2 // n_eps2)
        J_avg1[run] = np.mean(J_runs_eps1)
        J_avg2[run] = np.mean(J_runs_eps2)


    visualize_control_block_ghavamzade(low_level_dataset1_vis, J_avg1, eps_step_avg1, 'ctrl+', epochs)
    visualize_control_block_ghavamzade(low_level_dataset2_vis, J_avg2, eps_step_avg2, 'ctrlx', epochs)


if __name__ == '__main__':

    output_dir = './small'
    mk_dir_recursive(output_dir)

    ghavamzade_plot(epochs=[0, 2, 4])

    plt.show()





