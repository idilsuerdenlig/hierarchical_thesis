import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from tqdm import tqdm, trange
from matplotlib2tikz import save as tikz_save
from library.utils.pick_eps import pick_eps


def trajectory_plot_small(n_trajectories, output_dir):
    base_dir = '/home/dave/Documenti/results_idil/Small/'
    algorithms = ['H-GPOMDP', 'H-PGPE', 'H-PI']

    for alg in tqdm(algorithms):
        dir = base_dir + alg + '/'

        experiment_params = np.load(dir + 'experiment_params_dictionary.npy')

        how_many = experiment_params.item().get('how_many')
        n_epochs = experiment_params.item().get('n_runs')
        ep_per_run = experiment_params.item().get('ep_per_run')

        dataset_eval = np.load(dir+ str(how_many - 1) + '/dataset_eval_file.npy')
        dataset_eval_vis = list()
        for i, run in enumerate(n_epochs):
            dataset_eval_epoch = pick_eps(dataset_eval, start=run * ep_per_run, end=run * ep_per_run + ep_per_run)
            for traj in range(n_trajectories):
                dataset_eval_vis.append(dataset_eval_epoch[-traj+1])

        visualize_traj(dataset_eval_vis, alg, output_dir)

def visualize_traj(dataset_eval_vis, name, output_dir):
    fig = plt.figure()
    x_ep = list()
    y_ep = list()
    size_eps = list()
    x_list = list()
    y_list = list()
    ep_size = 0
    n_eps = 0

    xs = 100
    xe = 120
    ys = 120
    ye = 100

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

    for episode in len(x_list):
        x = x_list[episode]
        y = y_list[episode]
        plt.plot(x, y)

    xg = [xs, xe]
    yg = [ys, ye]
    plt.xlim(0, 160)
    plt.ylim(0, 160)

    plt.plot(xg, yg)
    plt.set_title(name)

    tikz_save(output_dir + '/' + name + '.tex',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')


if __name__ == '__main__':

    output_dir = './small'
    mk_dir_recursive(output_dir)
    trajectory_plot_small(n_trajectories=2, output_dir=output_dir)

    plt.show()





