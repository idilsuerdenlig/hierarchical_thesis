import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.patches import Ellipse
import math


def visualize_policy_params(parameter_dataset1, parameter_dataset2, parameter_dataset2_2=None, small=True, how_many=1):

    if parameter_dataset2 is None:
        print(parameter_dataset1)

    else:
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
            for param_set in parameter_dataset1[i]:
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
            diff_len = max_len-len(paramx_all[i])
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
        ax1 = fig.add_subplot(221, projection='3d')
        time =np.arange(len(paramx_avg))
        ax1.plot(paramx_avg, paramy_avg, time)
        ax1.set_title('pi1 parameters averaged')

        ax2 = fig.add_subplot(223)
        x0 = paramx_avg[0]
        y0 = paramy_avg[0]
        sigmax0 = paramsigmax_avg[0]
        sigmay0 = paramsigmay_avg[0]
        x1 = paramx_avg[len(paramx_avg)//2]
        y1 = paramy_avg[len(paramy_avg)//2]
        sigmax1 = paramsigmax_avg[len(paramsigmax_avg)//2]
        sigmay1 = paramsigmay_avg[len(paramsigmay_avg)//2]
        x2 = paramx_avg[-1]
        y2 = paramy_avg[-1]
        sigmax2 = paramsigmax_avg[-1]
        sigmay2 = paramsigmay_avg[-1]
        ellipse0 = Ellipse(xy=(x0, y0), width=4*sigmax0, height=4*sigmay0,
                            edgecolor='g', fc='None', lw=2)
        ellipse1 = Ellipse(xy=(x1, y1), width=4*sigmax1, height=4*sigmay1,
                            edgecolor='b', fc='None', lw=2)
        ellipse2 = Ellipse(xy=(x2, y2), width=4*sigmax2, height=4*sigmay2,
                            edgecolor='r', fc='None', lw=2)
        ax2.add_patch(ellipse0)
        ax2.add_patch(ellipse1)
        ax2.add_patch(ellipse2)
        if small:
            ax2.set_xlim(-50, 150)
            ax2.set_ylim(-50, 150)
        else:
            ax2.set_xlim(-500, 1500)
            ax2.set_ylim(-500, 1500)
        ax2.set_title('95% interval pi1 parameters averaged')



        if parameter_dataset2_2 is not None:
            ax4 = fig.add_subplot(224)
            ax4.plot(parameter_dataset2_2)



