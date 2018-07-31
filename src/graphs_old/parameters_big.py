import numpy as np
import matplotlib.pyplot as plt
from mushroom.utils.folder import *
from tqdm import tqdm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib2tikz import save as tikz_save


def high_level_parameter_plot_small(output_dir):
    base_dir = '/home/dave/Documenti/results_idil/Big/'
    algorithms = ['H-PGPE','H-PI']

    for alg in tqdm(algorithms):
        dir = base_dir + alg + '/'

        experiment_params = np.load(dir + 'experiment_params_dictionary.npy')

        how_many = experiment_params.item().get('how_many')

        parameter_dataset = list()
        for exp_no in range(how_many):
            parameter_dataset += [np.load(dir + str(exp_no) + '/parameter_dataset1_file.npy')]
        paramx_all = list()
        paramy_all = list()
        paramsigmax_all = list()
        paramsigmay_all = list()

        max_len = 0
        for i in range(how_many):
            paramx_data = list()
            paramy_data = list()
            paramsigmax_data = list()
            paramsigmay_data = list()
            for param_set in parameter_dataset[i]:
                paramx_data.append(param_set[0])
                paramy_data.append(param_set[1])
                paramsigmax_data.append(param_set[-2])
                paramsigmay_data.append(param_set[-1])
            max_len = max(max_len, len(paramx_data))
            paramx_all.append(paramx_data)
            paramy_all.append(paramy_data)
            paramsigmax_all.append(paramsigmax_data)
            paramsigmay_all.append(paramsigmay_data)

        for i in range(how_many):
            diff_len = max_len - len(paramx_all[i])
            if diff_len is not 0:
                for m in range(diff_len):
                    paramx_all[i].append(np.NaN)
                    paramy_all[i].append(np.NaN)
                    paramsigmax_all[i].append(np.NaN)
                    paramsigmay_all[i].append(np.NaN)

        paramx_avg = np.nanmean(paramx_all, axis=0)
        paramy_avg = np.nanmean(paramy_all, axis=0)
        paramsigmax_avg = np.nanmean(paramsigmax_all, axis=0)
        paramsigmay_avg = np.nanmean(paramsigmay_all, axis=0)

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        time = np.arange(len(paramx_avg))
        ax1.plot(paramx_avg, paramy_avg, time)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('t')

        ax1.view_init(30, 75)

        plt.savefig(output_dir + '/' + alg + '-mu.png', bbox_inches='tight')
        #ax1.set_title(alg+'_pi1 parameters averaged')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        x0 = paramx_avg[0]
        y0 = paramy_avg[0]
        sigmax0 = paramsigmax_avg[0]
        sigmay0 = paramsigmay_avg[0]
        x1 = paramx_avg[len(paramx_avg) // 2]
        y1 = paramy_avg[len(paramy_avg) // 2]
        sigmax1 = paramsigmax_avg[len(paramsigmax_avg) // 2]
        sigmay1 = paramsigmay_avg[len(paramsigmay_avg) // 2]
        x2 = paramx_avg[-1]
        y2 = paramy_avg[-1]
        sigmax2 = paramsigmax_avg[-1]
        sigmay2 = paramsigmay_avg[-1]
        ellipse0 = Ellipse(xy=(x0, y0), width=4 * sigmax0, height=4 * sigmay0,
                           edgecolor='g', fc='None', lw=2)
        ellipse1 = Ellipse(xy=(x1, y1), width=4 * sigmax1, height=4 * sigmay1,
                           edgecolor='b', fc='None', lw=2)
        ellipse2 = Ellipse(xy=(x2, y2), width=4 * sigmax2, height=4 * sigmay2,
                           edgecolor='r', fc='None', lw=2)
        ax2.add_patch(ellipse0)
        ax2.add_patch(ellipse1)
        ax2.add_patch(ellipse2)
        ax2.set_xlim(-100, 1100)
        ax2.set_ylim(-100, 1100)

        xs = 350
        xe = 450
        ys = 400
        ye = 400

        xg = [xs, xe]
        yg = [ys, ye]

        plt.plot(xg, yg, color='k')

        plt.ylabel('y')
        plt.xlabel('x')

        tikz_save(output_dir + '/' + alg + '-sigma.tex',
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth')

        ax2.set_title(alg + '_95% interval pi1 parameters averaged')

if __name__ == '__main__':
    load = False

    output_dir = './big'
    mk_dir_recursive(output_dir)
    high_level_parameter_plot_small(output_dir)

    plt.show()


