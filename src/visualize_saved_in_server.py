from visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from visualize_control_block import visualize_control_block
from visualize_control_block_ghavamzade import visualize_control_block_ghavamzade
from visualize_policy_params import visualize_policy_params
from arrows import plot_arrows
import numpy as np


def visualize_saved_in_server(our_approach=True, small=True, n_gates=1, how_many=1):

    if our_approach:

        parameter_dataset1 = list()
        parameter_dataset2 = list()
        dataset_learn_visual = list()
        dataset_eval = list()
        if small:
            low_level_dataset = list()

        for i in xrange(how_many):
            if small:
                low_level_dataset += [np.load('latest/'+str(i)+'/low_level_dataset_file.npy')]

            parameter_dataset1 += [np.load('latest/'+str(i)+'/parameter_dataset1_file.npy')]
            parameter_dataset2 += [np.load('latest/'+str(i)+'/parameter_dataset2_file.npy')]

            dataset_learn_visual += [np.load('latest/'+str(i)+'/dataset_learn_visual_file.npy')]
            dataset_eval += [np.load('latest/'+str(i)+'/dataset_eval_file.npy')]


        visualize_policy_params(parameter_dataset1, parameter_dataset2, small=small, how_many=how_many)
        visualize_ship_steering(dataset_learn_visual, name='learn', small=small, n_gates=n_gates, how_many= how_many)
        visualize_ship_steering(dataset_eval, 'evaluate', small=small, n_gates=n_gates, how_many=how_many)
        if small:
            visualize_control_block(low_level_dataset, ep_count=20, how_many=how_many)

    else:

        for i in xrange(how_many):

            act_max_q_val_tiled = np.load('latest/'+str(i)+'/act_max_q_val_tiled_file.npy')
            max_q_val_tiled = np.load('latest/'+str(i)+'/max_q_val_tiled_file.npy')
            low_level_dataset1 = np.load('latest/'+str(i)+'/low_level_dataset1_file.npy')
            low_level_dataset2 = np.load('latest/'+str(i)+'/low_level_dataset2_file.npy')
            dataset_learn = np.load('latest/'+str(i)+'/dataset_learn_file.npy')
            dataset_eval = np.load('latest/'+str(i)+'/dataset_eval_file.npy')

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
