import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mushroom.utils.dataset import compute_J
from mushroom.utils.folder import *
from tqdm import tqdm, trange
from matplotlib2tikz import save as tikz_save



def compute_data():
    base_dir = '/home/dave/Documenti/results_idil/Small/'
    algorithms = ['H-GPOMDP', 'H-PGPE', 'H-PI']

    len_results = dict()
    j_results = dict()

    for alg in tqdm(algorithms):
        dir = base_dir + alg + '/'


        experiment_params = np.load(dir + 'experiment_params_dictionary.npy')

        how_many = experiment_params.item().get('how_many')
        n_runs = experiment_params.item().get('n_runs')
        ep_per_run = experiment_params.item().get('ep_per_run')

        Jep = list()
        Lep = list()

        for exp_no in trange(how_many):
            dataset_params = np.load(dir + str(exp_no) + '/parameter_dataset1_file.npy')


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
    load = False

    output_dir = './small'
    mk_dir_recursive(output_dir)

    if load:
        j_results = np.load(output_dir + '/J.npy').item()
        len_results = np.load(output_dir + '/L.npy').item()
    else:
        j_results, len_results = compute_data()

        np.save(output_dir + '/J.npy', j_results)
        np.save(output_dir + '/L.npy', len_results)

    # State of the art comparison
    create_plot(['H-PGPE', 'REPS', 'RWR', 'PGPE', 'GPOMDP'],
                ['b', 'r', 'g', 'c', 'm'],
                j_results, 'art_J', 'J', True,
                )
    create_plot(['REPS', 'RWR', 'PGPE', 'GPOMDP'],
                ['r', 'g', 'c', 'm'],
                len_results, 'art_L', 'episode length')
    create_plot(['H-PGPE', 'REPS'],
                ['b', 'r'],
                len_results, 'comp_art_L', 'episode length')

    # hierarchical comparison
    algs = ['H-PGPE', 'H-PI', 'H-GPOMDP']
    colors = ['b', 'tab:orange', 'tab:purple']
    create_plot(algs, colors, j_results, 'hier_J', 'J', True)
    create_plot(algs, colors, len_results, 'hier_L', 'episode length')

    plt.show()


