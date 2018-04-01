import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from tqdm import tqdm, trange
from matplotlib2tikz import save as tikz_save


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
    eps_step_avg = list()


    for run in range(n_runs):
        dataset_control_run = datalist_control[run]
        total_steps = len(dataset_control_run)
        last_ep_of_run = pick_last_ep(dataset_control_run)
        datalist_control_vis.append(last_ep_of_run)
        J_runs_eps= compute_J(dataset_control_run, gamma)
        n_eps = len(J_runs_eps)
        eps_step_avg.append(total_steps/n_eps)
        J_avg[run] = np.mean(J_runs_eps, axis=0)

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


    ax2n.plot(eps_step_avg)
    ax2n.set_title('size_eps')

    plt.tight_layout()


def visualize_traj(dataset_eval_vis, name, output_dir):

    fig = plt.figure()
    x_ep = list()
    y_ep = list()
    size_eps = list()
    x_list = list()
    y_list = list()
    ep_size = 0
    n_eps = 0

    xs = 350
    xe = 450
    ys = 400
    ye = 400

    for dataset_step in dataset_eval_vis:

        if not dataset_step[-1]:
            states_step = dataset_step[0]
            x_step = states_step[0]
            y_step = states_step[1]

            x_ep.append(x_step)
            y_ep.append(y_step)
            ep_size += 1
        else:
            size_eps.append(ep_size)
            ep_size = 0
            x_ep.append(dataset_step[3][0])
            y_ep.append(dataset_step[3][1])
            x_list.append(x_ep)
            y_list.append(y_ep)
            x_ep = []
            y_ep = []
            n_eps += 1


    for episode in range(len(x_list)):
        x = x_list[episode]
        y = y_list[episode]
        plt.plot(x, y)

    xg = [xs, xe]
    yg = [ys, ye]
    plt.xlim(-100, 1100)
    plt.ylim(-100, 1100)

    plt.plot(xg, yg)
    plt.title(name)

    tikz_save(output_dir + '/' + name + '.tex',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')


def ghavamzade_plot(epochs, output_dir):

    dir = '/home/dave/Documenti/results_idil/Ghavamzade/'
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

    dataset_eval = np.load('latest/'+str(how_many-1)+'/dataset_eval_file.npy')
    for run in epochs:
        dataset_eval_vis = list()
        dataset_eval_epoch = pick_eps(dataset_eval, start=run * ep_per_run, end=run * ep_per_run + ep_per_run)
        for traj in range(3):
            dataset_eval_vis += dataset_eval_epoch[-traj - 1]
        visualize_traj(dataset_eval_vis+'_'+ str(run), output_dir)


    low_level_dataset1 = np.load('latest/'+str(how_many-1)+'/low_level_dataset1_file.npy')
    low_level_dataset2 = np.load('latest/'+str(how_many-1)+'/low_level_dataset2_file.npy')

    visualize_control_block_ghavamzade(low_level_dataset1, ep_count=2)
    plt.suptitle('ctrl+')
    visualize_control_block_ghavamzade(low_level_dataset2, ep_count=2)
    plt.suptitle('ctrlx')




if __name__ == '__main__':

    output_dir = './small'
    mk_dir_recursive(output_dir)
    ghavamzade_plot(epochs=[0, 6, 10, 20], output_dir=output_dir)

    plt.show()





