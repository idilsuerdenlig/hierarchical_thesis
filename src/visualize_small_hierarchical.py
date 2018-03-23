from library.visualization_tools.visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from library.visualization_tools.visualize_control_block import visualize_control_block
from library.visualization_tools.visualize_policy_parameters import visualize_policy_params
from library.visualization_tools.arrows import plot_arrows
from mushroom.utils.dataset import compute_J
import numpy as np
from tqdm import tqdm


def visualize_small_hierarchical(gamma=1, range_vis=None):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    experiment_params = np.load('latest/experiment_params_dictionary.npy')
    how_many = experiment_params.item(0).get('how_many')
    n_runs = experiment_params.item(1).get('n_runs')
    ep_per_run = experiment_params.item(3).get('ep_per_run')
    parameter_dataset1 = list()
    parameter_dataset2 = list()
    size_eps = list()
    size_exp_list = list()
    J_avg = np.zeros((n_runs,))
    size_eps_avg = np.zeros((n_runs,))
    J_exp = []
    ep_step_no = 0

    for exp_no in range(how_many):

        parameter_dataset1 += [np.load('latest/' + str(exp_no) + '/parameter_dataset1_file.npy')]
        parameter_dataset2 += [np.load('latest/' + str(exp_no) + '/parameter_dataset2_file.npy')]
        dataset_eval = np.load('latest/' + str(exp_no) + '/dataset_eval_file.npy')

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

    dataset_eval = np.load('latest/' + str(how_many - 1) + '/dataset_eval_visual_file.npy')
    low_level_dataset = np.load('latest/' + str(exp_no) + '/low_level_dataset_file.npy')

    small=True

    visualize_policy_params(parameter_dataset1, parameter_dataset2, small=small, how_many=how_many)
    visualize_ship_steering(dataset_eval, 'evaluate', small=small, range_eps=range_vis, n_gates=1, how_many=how_many)
    visualize_control_block(datalist_control=low_level_dataset, ep_count=5, how_many=how_many)

    plt.show()

if __name__ == '__main__':
    # range_vis must be a range()
    visualize_small_hierarchical(gamma=0.99, range_vis=None)