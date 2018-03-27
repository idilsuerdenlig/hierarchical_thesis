from library.visualization_tools.visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from library.visualization_tools.visualize_control_block_ghavamzade import visualize_control_block_ghavamzade
from library.visualization_tools.arrows import plot_arrows
from library.utils.pick_last_ep_dataset import pick_last_ep
from library.utils.pick_eps import pick_eps
import numpy as np

def visualize_bonarini_hierarchical(ep_count=10):

    experiment_params = np.load('latest/experiment_params_dictionary.npy')
    how_many = experiment_params.item().get('how_many')
    n_runs = experiment_params.item().get('n_runs')
    ep_per_run = experiment_params.item().get('ep_per_run')

    for i in range(how_many):

        act_max_q_val_tiled = np.load('latest/'+str(i)+'/act_max_q_val_tiled_file.npy')
        max_q_val_tiled = np.load('latest/'+str(i)+'/max_q_val_tiled_file.npy')
        low_level_dataset1 = np.load('latest/'+str(i)+'/low_level_dataset1_file.npy')
        low_level_dataset2 = np.load('latest/'+str(i)+'/low_level_dataset2_file.npy')
        plot_arrows(act_max_q_val_tiled)
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(max_q_val_tiled, cmap=plt.cm.Blues)
        plt.colorbar(heatmap)
        visualize_control_block_ghavamzade(low_level_dataset1, ep_count=ep_count)
        plt.suptitle('ctrl+')
        visualize_control_block_ghavamzade(low_level_dataset2, ep_count=ep_count)
        plt.suptitle('ctrlx')

    dataset_eval = np.load('latest/' + str(how_many-1) + '/dataset_eval_file.npy')
    dataset_eval_vis = list()
    for run in range(n_runs):
        dataset_eval_run = pick_eps(dataset_eval, start=run * ep_per_run, end=run * ep_per_run + ep_per_run)
        last_ep_of_run = pick_last_ep(dataset_eval_run)
        for step in last_ep_of_run:
            dataset_eval_vis.append(step)
    visualize_ship_steering(dataset_eval_vis, name='evaluate', small=False, n_runs=n_runs, ep_per_run=ep_per_run)

    plt.show()

if __name__ == '__main__':
    # range_vis must be a range()
    visualize_bonarini_hierarchical(ep_count=10)
