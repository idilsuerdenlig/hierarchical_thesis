from .visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from .visualize_control_block import visualize_control_block
from .visualize_control_block_ghavamzade import visualize_control_block_ghavamzade
from .visualize_policy_parameters import visualize_policy_params
from .arrows import plot_arrows
from mushroom.utils.dataset import compute_J
import numpy as np
from tqdm import tqdm


def visualize_saved_in_server(our_approach=True, small=True, n_gates=1, how_many=1, gamma=1):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    if our_approach:

        parameter_dataset1 = list()
        parameter_dataset2 = list()
        i = 0
        size_eps = list()
        size_exp_list = list()
        max_lenJ = 0
        max_leni = 0
        J_avg = np.zeros((10,))
        size_eps_avg = np.zeros((10,))
        J_exp = []
        ep_step_no = 0

        for exp_no in range(how_many):

            parameter_dataset1 += [np.load('latest/'+str(exp_no)+'/parameter_dataset1_file.npy')]
            parameter_dataset2 += [np.load('latest/'+str(exp_no)+'/parameter_dataset2_file.npy')]
            dataset_eval = np.load('latest/' + str(exp_no) + '/dataset_eval_file.npy')

            J_runs_eps = compute_J(dataset_eval, gamma)
            for i in range(10):
                J_avg[i] = np.mean(J_runs_eps[10*i:10*i+10], axis=0)

            for dataset_step in dataset_eval:
                if not dataset_step[-1]:
                    ep_step_no += 1
                else:
                    size_eps.append(ep_step_no)
                    ep_step_no = 0
            J_exp.append(J_avg)
            J_avg = np.zeros((10,))

            for i in range(10):
                size_eps_avg[i] = np.mean(size_eps[10*i:10*i+10], axis=0)

            size_exp_list.append(size_eps_avg)
            size_eps = list()
            size_eps_avg = np.zeros((10,))



        ''' a = list()
        for _ in range(how_many):
            a.append(list())
        for exp_no in tqdm(range(how_many), dynamic_ncols=True,
                                   disable=False, leave=False):
            diff_len = max_leni-len(size_exp_list[exp_no])
            for each in tqdm(size_exp_list[exp_no], dynamic_ncols=True,
                                   disable=False, leave=False):
                a[exp_no].append(each)
            if diff_len is not 0:
                for _ in range(diff_len):
                    a[exp_no].append(np.NaN)'''

        J_all_avg = np.mean(J_exp, axis=0)
        J_all_avg_err = np.std(J_exp, axis=0)/np.sqrt(10)*1.96
        size_all_avg = np.mean(size_exp_list, axis=0)
        size_all_avg_err = np.std(size_all_avg, axis=0)/np.sqrt(10)*1.96
        ax1.errorbar(x=np.arange(10)+1, y=J_all_avg, yerr=J_all_avg_err)
        ax1.set_title('J_eps_averaged')

        ax2.errorbar(x=np.arange(10)+1, y=size_all_avg, yerr=size_all_avg_err)
        ax2.set_title('size_eps_averaged')

        dataset_eval = np.load('latest/'+str(how_many-1)+'/dataset_eval_visual_file.npy')
        visualize_policy_params(parameter_dataset1, parameter_dataset2, small=small, how_many=how_many)
        visualize_ship_steering(dataset_eval, 'evaluate', small=small, n_gates=n_gates, how_many=how_many)


    else:

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
            visualize_control_block_ghavamzade(low_level_dataset1, ep_count=2)
            plt.suptitle('ctrl+')
            visualize_control_block_ghavamzade(low_level_dataset2, ep_count=2)
            plt.suptitle('ctrlx')
            visualize_ship_steering(dataset_eval, name='evaluate', small=small)

    plt.show()
