import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.patches import Ellipse
import math


def visualize_policy_params(parameter_dataset1, parameter_dataset2, parameter_dataset2_2=None, small=True, how_many=1):

    paramx_avg = list()
    paramy_avg = list()
    paramsigmax_avg = list()
    paramsigmay_avg = list()

    paramx_all = list()
    paramy_all = list()
    paramsigmax_all = list()
    paramsigmay_all = list()

    max_len = 0
    for i in xrange(how_many):
        paramx_data = list()
        paramy_data = list()
        paramsigmax_data = list()
        paramsigmay_data = list()
        print 'parameter_dataset[i]:    ', parameter_dataset1[i]
        for param_set in parameter_dataset1[i]:
            print 'param set in parameter_dataset[i]:   ',param_set
            paramx_data.append(param_set[0])
            paramy_data.append(param_set[1])
            paramsigmax_data.append(param_set[-2])
            paramsigmay_data.append(param_set[-1])
        max_len = max(max_len, len(paramx_data))
        paramx_all.append(paramx_data)
        paramy_all.append(paramy_data)
        paramsigmax_all.append(paramsigmax_data)
        paramsigmay_all.append(paramsigmay_data)

    print paramx_all
    print paramy_all
    print paramsigmax_all
    print paramsigmay_all

    for n in xrange(max_len):
        counter = 0
        x_total = 0
        y_total = 0
        sigmax_total = 0
        sigmay_total = 0

        for i in xrange(how_many):
            one_set = [item[0] for item in paramx_all]
            second_set = [item[1] for item in paramx_all]
            #thirdset = [item[2] for item in paramx_all]
            print one_set
            print second_set
            #print thirdset
            exit()
            if paramx_all[i][n] is not None:
                counter += 1
                x_total += paramx_all[i][n]
                y_total += paramy_all[i][n]
                sigmax_total += paramsigmax_all[i][n]
                sigmay_total += paramsigmay_all[i][n]
        paramx_avg.append(x_total/counter)
        paramy_avg.append(y_total/counter)
        paramsigmax_avg.append(sigmax_total/counter)
        paramsigmay_avg.append(sigmay_total/counter)


    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    time =np.arange(len(paramx_avg))
    ax1.plot(paramx_avg, paramy_avg, time)
    ax1.scatter(paramx_avg[0], paramy_avg[0], time[0], marker='o',color='g')
    ax1.scatter(paramx_avg[-1], paramy_avg[-1], time[-1], marker='o', color='r')
    ax1.set_title('pi1 parameters averaged')

    ax2 = fig.add_subplot(223)
    x0 = paramx_avg[0]
    y0 = paramy_avg[0]
    sigmax0 = paramsigmax_avg[0]
    sigmay0 = paramsigmay_avg[0]
    x1 = paramx_avg[int(len(paramx_avg)/2)]
    y1 = paramy_avg[int(len(paramy_avg)/2)]
    sigmax1 = paramsigmax_avg[int(len(paramsigmax_avg)/2)]
    sigmay1 = paramsigmay_avg[int(len(paramsigmay_avg)/2)]
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
        ax2.set_xlim(-50, 1100)
        ax2.set_ylim(-50, 1100)
    ax2.set_title('95% interval pi1 parameters averaged')

    param2_avg = list()

    max_len2 = 0
    for i in xrange(how_many):
        params2_one_experiment = parameter_dataset2[i]
        max_len2 = (len(params2_one_experiment), max_len2)

    for n in xrange(max_len2):
        counter = 0
        total2 = 0
        for i in xrange(how_many):
            if parameter_dataset2[i][n] is not None:
                counter += 1
                total2 += parameter_dataset2[i][n]
        param2_avg.append(total2/counter)

    ax3 = fig.add_subplot(222)
    ax3.plot(param2_avg)
    ax3.set_title('pi2 parameters averaged')

    if parameter_dataset2_2 is not None:
        ax4 = fig.add_subplot(224)
        ax4.plot(parameter_dataset2_2)



