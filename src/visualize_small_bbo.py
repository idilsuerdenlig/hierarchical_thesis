from library.visualization_tools.visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from library.utils.pick_eps import pick_eps
from mushroom.utils.dataset import compute_J
import numpy as np


def visualize_small_bbo(folder, alg, gamma=1, range_vis=None):

    experiment_params = np.load(folder + 'experiment_params_dictionary.npy')
    how_many = experiment_params.item().get('how_many')
    n_runs = experiment_params.item().get('n_runs')
    ep_per_run = experiment_params.item().get('ep_per_run')

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    size_eps = list()
    size_exp_list = list()
    J_avg = np.zeros((n_runs,))
    size_eps_avg = np.zeros((n_runs,))
    J_exp = []
    ep_step_no = 0

    for exp_no in range(how_many):

        dataset_eval = np.load(folder + str(exp_no) + '/dataset_eval_file.npy')

        J_runs_eps = compute_J(dataset_eval, gamma)

        for i in range(n_runs):
            J_avg[i] = np.mean(J_runs_eps[ep_per_run * i:ep_per_run * i + ep_per_run], axis=0)

        for dataset_step in dataset_eval:
            if not dataset_step[-1]:
                ep_step_no += 1
            else:
                size_eps.append(ep_step_no)
                ep_step_no = 0
        J_exp.append(J_avg)
        J_avg = np.zeros((n_runs,))

        for i in range(n_runs):
            size_eps_avg[i] = np.mean(size_eps[ep_per_run * i:ep_per_run * i + ep_per_run], axis=0)

        size_exp_list.append(size_eps_avg)
        size_eps = list()
        size_eps_avg = np.zeros((n_runs,))

    J_all_avg = np.mean(J_exp, axis=0)
    J_all_avg_err = np.std(J_exp, axis=0) / np.sqrt(how_many) * 1.96 if how_many > 1 else 0
    size_all_avg = np.mean(size_exp_list, axis=0)
    size_all_avg_err = np.std(size_all_avg, axis=0) / np.sqrt(how_many) * 1.96 if how_many > 1 else 0
    ax1.errorbar(x=np.arange(n_runs) + 1, y=J_all_avg, yerr=J_all_avg_err)
    ax1.set_title('J_eps_averaged')

    ax2.errorbar(x=np.arange(n_runs) + 1, y=size_all_avg, yerr=size_all_avg_err)
    ax2.set_title('size_eps_averaged')
    fig.suptitle(alg)

    dataset_eval = np.load(folder + str(how_many - 1) + '/dataset_eval_file.npy')
    dataset_eval_vis = list()
    for run in range(n_runs):
        dataset_eval_run = pick_eps(dataset_eval,
                                    start=run * ep_per_run,
                                    end=run * ep_per_run + ep_per_run)
        last_ep_of_run = dataset_eval_run[-1]
        for step in last_ep_of_run:
            dataset_eval_vis.append(step)
    visualize_ship_steering(dataset_eval_vis, alg + ' evaluate', small=True,
                            range_eps=range_vis)

    plt.show()


if __name__ == '__main__':
    # range_vis must be a range()
    base_folder = 'latest/'

    algs = [
        'REPS',
        'RWR',
        'PGPE'
    ]

    for alg in algs:
        folder = base_folder + alg + '/'
        visualize_small_bbo(folder, alg, gamma=0.99, range_vis=None)