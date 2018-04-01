import numpy as np
import matplotlib.pyplot as plt
from mushroom.utils.folder import *
from tqdm import tqdm
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


def trajectory_plot_small(run, epochs, n_traj, alg, base_dir, output_dir):
    dir = base_dir + alg + '/'

    experiment_params = np.load(dir + 'experiment_params_dictionary.npy')
    ep_per_run = experiment_params.item().get('ep_per_run')

    dataset_eval = np.load(dir + str(run) + '/dataset_eval_file.npy')

    for run in epochs:
        dataset_eval_vis = list()
        dataset_eval_epoch = pick_eps(dataset_eval, start=run * ep_per_run, end=run * ep_per_run + ep_per_run)
        for traj in range(n_traj):
            dataset_eval_vis += dataset_eval_epoch[-traj-1]
        visualize_traj(dataset_eval_vis, alg+str(run), run, output_dir)


def visualize_traj(dataset_eval_vis, name, epoch, output_dir):
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


    subsampling = 4
    plt.figure()

    for episode in range(len(x_list)):
        x = x_list[episode]
        y = y_list[episode]

        x_plotted = x[0::subsampling]
        y_plotted = y[0::subsampling]

        if len(x_plotted) % subsampling is not 0:
            x_plotted.append(x[-1])
            y_plotted.append(y[-1])

        plt.plot(x_plotted, y_plotted)


    xg = [xs, xe]
    yg = [ys, ye]

    plt.xlim(0, 150)
    plt.ylim(0, 150)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel(str(epoch))

    plt.plot(xg, yg)


    tikz_save(output_dir + '/' + name + '.tex',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth',
              extra_axis_parameters=['ticks=none'])

    plt.title(name)


if __name__ == '__main__':

    output_dir = './small/traj'
    mk_dir_recursive(output_dir)
    base_dir = '/home/dave/Documenti/results_idil/Small/'
    algorithms = ['H-GPOMDP','H-PGPE','H-PI', 'RWR', 'REPS', 'PGPE', 'GPOMDP']
    for alg in tqdm(algorithms):
        trajectory_plot_small(run=25,
                              epochs=[0, 3, 12, 24],
                              n_traj=5,
                              alg=alg,
                              base_dir=base_dir,
                              output_dir=output_dir)

    plt.show()





