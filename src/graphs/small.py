import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mushroom.utils.dataset import compute_J
from tqdm import tqdm, trange

def compute_episode_lenght(dataset_eval, n_runs, ep_per_run):
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
        size_eps_avg[i] = np.mean(size_eps[ep_per_run * i:ep_per_run * i + ep_per_run], axis=0)

    return size_eps_avg

def compute_mean_J(dataset_eval, n_runs, ep_per_run, gamma):
    J_runs_eps = compute_J(dataset_eval, gamma)
    J_avg = np.zeros(n_runs+1)
    for i in range(n_runs+1):
        J_avg[i] = np.mean(J_runs_eps[ep_per_run * i:ep_per_run * i + ep_per_run], axis=0)

    return J_avg


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)

    interval, _ = st.t.interval(0.95, n-1, scale=se)

    return mean, interval

def compute_data():
    base_dir = '/home/dave/Documenti/results_idil/Small/'
    algorithms = ['GPOMDP', 'hierarchical_GPOMDP', 'hierarchical_PGPE', 'hierarchical_PI', 'PGPE', 'REPS', 'RWR']



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
            dataset_eval = np.load(dir + str(exp_no) + '/dataset_eval_file.npy')
            J_avg = compute_mean_J(dataset_eval, n_runs, ep_per_run, 0.99)
            L_avg = compute_episode_lenght(dataset_eval, n_runs, ep_per_run)

            Jep.append(J_avg)
            Lep.append(L_avg)

        Jmean, Jerr = get_mean_and_confidence(Jep)
        Lmean, Lerr = get_mean_and_confidence(Lep)

        j_results[alg] = (Jmean, Jerr)
        len_results[alg] = (Lmean, Lerr)

    return j_results, len_results


if __name__ == '__main__':
    load = False

    if load:
        j_results = np.load('small_J.npy').item()
        len_results = np.load('small_L.npy').item()
    else:
        j_results, len_results = compute_data()

        np.save('small_J.npy', j_results)
        np.save('small_L.npy', len_results)

    # J plot
    plt.figure()
    for alg, (Jmean, Jerr) in j_results.items():
        plt.errorbar(x=np.arange(len(Jmean)), y=Jmean, yerr=Jerr)

    plt.legend(list(j_results.keys()))

    # Episode len plot
    plt.figure()
    for alg, (Lmean, Lerr) in len_results.items():
        plt.errorbar(x=np.arange(len(Lmean)), y=Lmean, yerr=Lerr)


    plt.show()

