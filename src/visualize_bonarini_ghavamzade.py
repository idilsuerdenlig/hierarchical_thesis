from library.visualization_tools.visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from library.visualization_tools.visualize_control_block import visualize_control_block
from library.visualization_tools.visualize_control_block_ghavamzade import visualize_control_block_ghavamzade
from library.visualization_tools.visualize_policy_parameters import visualize_policy_params
from library.visualization_tools.arrows import plot_arrows
from mushroom.utils.dataset import compute_J
import numpy as np
from tqdm import tqdm

def visualize_bonarini_hierarchical(gamma=1, ep_count=2, how_many=1):

    #experiment_params = np.load('latest/experiment_params_dictionary.npy')
    #how_many = experiment_params.item(0).get('how_many')
    #n_runs = experiment_params.item(1).get('n_runs')
    #ep_per_run = experiment_params.item(3).get('ep_per_run')

    for i in range(how_many):

        act_max_q_val_tiled = np.load('latest/'+str(i)+'/act_max_q_val_tiled_file.npy')
        max_q_val_tiled = np.load('latest/'+str(i)+'/max_q_val_tiled_file.npy')
        low_level_dataset1 = np.load('latest/'+str(i)+'/low_level_dataset1_file.npy')
        low_level_dataset2 = np.load('latest/'+str(i)+'/low_level_dataset2_file.npy')
        dataset_eval = np.load('latest/'+str(i)+'/dataset_eval_file.npy')
        plot_arrows(act_max_q_val_tiled)
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(max_q_val_tiled, cmap=plt.cm.Blues)
        plt.colorbar(heatmap)
        visualize_control_block_ghavamzade(low_level_dataset1, ep_count=ep_count)
        plt.suptitle('ctrl+')
        visualize_control_block_ghavamzade(low_level_dataset2, ep_count=ep_count)
        plt.suptitle('ctrlx')
        visualize_ship_steering(dataset_eval, name='evaluate', small=False)

    plt.show()

if __name__ == '__main__':
    # range_vis must be a range()
    visualize_bonarini_hierarchical(gamma=0.99, ep_count=2, how_many=1)
