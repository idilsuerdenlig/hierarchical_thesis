import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from tqdm import tqdm, trange
from matplotlib2tikz import save as tikz_save


def compute_episode_lenght(dataset_eval, n_runs, eval_run):
    size_eps_avg = np.zeros(n_runs+1)
    size_eps = list()
    ep_step_no = 0
    for dataset_step in dataset_eval:
        if not dataset_step[-1]:
            ep_step_no += 1
        else:
            size_eps.append(ep_step_no)
            ep_step_no = 0

    for i in range(n_runs+1):
        size_eps_avg[i] = np.mean(size_eps[eval_run * i:eval_run * i + eval_run], axis=0)

    return size_eps_avg


def compute_mean_J(dataset_eval, n_runs, eval_run, gamma):
    J_runs_eps = compute_J(dataset_eval, gamma)
    J_avg = np.zeros(n_runs+1)
    for i in range(n_runs+1):
        J_avg[i] = np.mean(J_runs_eps[eval_run * i:eval_run * i + eval_run], axis=0)

    return J_avg


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval


def compute_data():
    base_dir = '/home/dave/Documenti/results_idil/Big/'
    algorithms = ['H-PGPE', 'H-PI']

    len_results = dict()
    j_results = dict()

    for alg in tqdm(algorithms):
        dir = base_dir + alg + '/'


        experiment_params = np.load(dir + 'experiment_params_dictionary.npy')

        how_many = experiment_params.item().get('how_many')
        n_runs = experiment_params.item().get('n_runs')
        eval_run = experiment_params.item().get('eval_run')

        Jep = list()
        Lep = list()

        for exp_no in trange(how_many):
            dataset_eval = np.load(dir + str(exp_no) + '/dataset_eval_file.npy')
            J_avg = compute_mean_J(dataset_eval, n_runs, eval_run, 0.99)
            L_avg = compute_episode_lenght(dataset_eval, n_runs, eval_run)

            Jep.append(J_avg)
            Lep.append(L_avg)

        Jmean, Jerr = get_mean_and_confidence(Jep)
        Lmean, Lerr = get_mean_and_confidence(Lep)

        j_results[alg] = (Jmean, Jerr)
        len_results[alg] = (Lmean, Lerr)

    return j_results, len_results


def create_plot(algs, colors, dictionary, plot_name, y_label, legend=False, x_label='epoch'):
    plt.figure()
    for alg, c in zip(algs, colors):
        (mean, err) = dictionary[alg]
        plt.errorbar(x=np.arange(len(mean)), y=mean, yerr=err, color=c)

    if legend:
        plt.legend(algs)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    tikz_save(output_dir + '/' + plot_name + '.tex',
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth')


if __name__ == '__main__':
    load = True

    output_dir = './big'
    mk_dir_recursive(output_dir)

    if load:
        j_results = np.load(output_dir + '/J.npy').item()
        len_results = np.load(output_dir + '/L.npy').item()
    else:
        j_results, len_results = compute_data()

        np.save(output_dir + '/J.npy', j_results)
        np.save(output_dir + '/L.npy', len_results)


    # hierarchical comparison
    algs = ['H-PGPE', 'H-PI']
    colors = ['b', 'tab:orange']
    create_plot(algs, colors, j_results, 'hier_J', 'J', True)
    create_plot(algs, colors, len_results, 'hier_L', 'episode length')

    plt.show()


