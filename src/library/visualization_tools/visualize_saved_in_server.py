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
        dataset_learn_visual = list()
        dataset_eval = list()
        i = 0
        size_eps = list()
        size_exp_list = list()
        J_eps = 0
        J_eps_exp = list()
        J_exp_list = list()
        max_lenJ = 0
        max_leni = 0


        for exp_no in range(how_many):

            parameter_dataset1 += [np.load('latest/'+str(exp_no)+'/parameter_dataset1_file.npy')]
            parameter_dataset2 += [np.load('latest/'+str(exp_no)+'/parameter_dataset2_file.npy')]
            low_level_dataset = np.load('latest/'+str(exp_no)+'/low_level_dataset_file.npy')
            for dataset_step in low_level_dataset:
                if not dataset_step[-1]:
                    df = gamma**i  #df *= gamma
                    reward_step = dataset_step[2]
                    J_eps += df*reward_step
                    i += 1
                else:
                    df = gamma**i
                    reward_step = dataset_step[2]
                    J_eps += df*reward_step
                    J_eps_exp.append(J_eps)
                    size_eps.append(i)
                    i = 0
                    J_eps = 0

            J_exp_list.append(J_eps_exp)
            size_exp_list.append(size_eps)
            max_lenJ = max(max_lenJ, len(J_eps_exp))
            max_leni = max(max_leni, len(size_eps))
            del J_eps_exp
            del size_eps
            J_eps_exp = list()
            size_eps = list()

        a = list()
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
                    a[exp_no].append(np.NaN)

        size_eps_avg = np.nanmean(a, axis=0)
        time = np.arange(max_leni)
        ax1.plot(time, size_eps_avg)
        ax1.set_title('size_eps_averaged')

        a = list()
        for _ in range(how_many):
            a.append(list())

        for exp_no in tqdm(range(how_many), dynamic_ncols=True,
                                   disable=False, leave=False):
            diff_len = max_lenJ-len(J_exp_list[exp_no])
            for each in tqdm(J_exp_list[exp_no], dynamic_ncols=True,
                                   disable=False, leave=False):
                a[exp_no].append(each)
            if diff_len is not 0:
                for _ in range(diff_len):
                    a[exp_no].append(np.NaN)

        J_eps_avg = np.nanmean(a, axis=0)
        time = np.arange(max_lenJ)
        ax2.plot(time, J_eps_avg)
        ax2.set_title('J_eps_averaged')

        dataset_eval = np.load('latest/'+str(i)+'/dataset_eval_file.npy')
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
