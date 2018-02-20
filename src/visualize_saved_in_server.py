from visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from visualize_control_block import visualize_control_block
from visualize_control_block_ghavamzade import visualize_control_block_ghavamzade
from visualize_policy_params import visualize_policy_params
from arrows import plot_arrows
import numpy as np


def visualize_saved_in_server(our_approach=True, small=True, n_gates=1):


    if our_approach:
        if small:
            low_level_dataset = np.load('low_level_dataset_file.npy')
            visualize_control_block(low_level_dataset, ep_count=20)

        parameter_dataset1 = np.load('parameter_dataset1_file.npy')
        parameter_dataset2 = np.load('parameter_dataset2_file.npy')

        dataset_learn_visual = np.load('dataset_learn_visual_file.npy')
        dataset_eval = np.load('dataset_eval_file.npy')

        visualize_policy_params(parameter_dataset1, parameter_dataset2, small=small)
        visualize_ship_steering(dataset_learn_visual, name='learn', small=small, n_gates=n_gates)
        visualize_ship_steering(dataset_eval, 'evaluate', small=small, n_gates=n_gates)

    else:

        act_max_q_val_tiled = np.load('act_max_q_val_tiled_file.npy')
        max_q_val_tiled = np.load('max_q_val_tiled_file.npy')
        low_level_dataset1 = np.load('low_level_dataset1_file.npy')
        low_level_dataset2 = np.load('low_level_dataset2_file.npy')
        dataset_learn = np.load('dataset_learn_file.npy')
        dataset_eval = np.load('dataset_eval_file.npy')

        plot_arrows(act_max_q_val_tiled)
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(max_q_val_tiled, cmap=plt.cm.Blues)
        plt.colorbar(heatmap)
        visualize_control_block_ghavamzade(low_level_dataset1, ep_count=5)
        plt.suptitle('ctrl+')
        visualize_control_block_ghavamzade(low_level_dataset2, ep_count=5)
        plt.suptitle('ctrlx')
        visualize_ship_steering(dataset_learn, name='learn', range_eps=xrange(1980, 1995), small=small)
        visualize_ship_steering(dataset_eval, name='evaluate', small=small)

    plt.show()
