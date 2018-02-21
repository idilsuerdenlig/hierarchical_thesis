import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.patches import Ellipse
import math


def visualize_policy_params(parameter_dataset1, parameter_dataset2, parameter_dataset2_2=None, small=True):

    #np.concatenate(self._approximator.get_weights(), self._sigma)
    #if sigmas is not None:
    #    print sigmas
    paramx_data = list()
    paramy_data = list()
    paramsigmax_data = list()
    paramsigmay_data = list()
    for param_set in parameter_dataset1:
        paramx_data.append(param_set[0])
        paramy_data.append(param_set[1])
        paramsigmax_data.append(param_set[-2])
        paramsigmay_data.append(param_set[-1])

    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    #ax4 = fig.add_subplot(224)
    time =np.arange(len(paramx_data))
    ax1.plot(paramx_data, paramy_data, time)
    ax1.scatter(paramx_data[0], paramy_data[0], time[0], marker='o',color='g')
    ax1.scatter(paramx_data[-1], paramy_data[-1], time[-1], marker='o', color='r')
    ax1.set_title('pi1 parameters')
    ax2.plot(parameter_dataset2)
    #ax4.plot(parameter_dataset2_2)
    ax2.set_title('pi2 parameters')

    ax3 = fig.add_subplot(223)
    x0 = paramx_data[0]
    y0 = paramy_data[0]
    sigmax0 = paramsigmax_data[0]
    sigmay0 = paramsigmay_data[0]

    x1 = paramx_data[int(len(paramx_data)/2)]
    y1 = paramy_data[int(len(paramy_data)/2)]
    sigmax1 = paramsigmax_data[int(len(paramsigmax_data)/2)]
    sigmay1 = paramsigmay_data[int(len(paramsigmay_data)/2)]

    x2 = paramx_data[-1]
    y2 = paramy_data[-1]
    sigmax2 = paramsigmax_data[-1]
    sigmay2 = paramsigmay_data[-1]

    ellipse0 = Ellipse(xy=(x0, y0), width=4*sigmax0, height=4*sigmay0,
                        edgecolor='g', fc='None', lw=2)
    ellipse1 = Ellipse(xy=(x1, y1), width=4*sigmax1, height=4*sigmay1,
                        edgecolor='b', fc='None', lw=2)
    ellipse2 = Ellipse(xy=(x2, y2), width=4*sigmax2, height=4*sigmay2,
                        edgecolor='r', fc='None', lw=2)
    ax3.add_patch(ellipse0)
    ax3.add_patch(ellipse1)
    ax3.add_patch(ellipse2)
    if small:
        ax3.set_xlim(-50, 150)
        ax3.set_ylim(-50, 150)
    else:
        ax3.set_xlim(-50, 1100)
        ax3.set_ylim(-50, 1100)
    ax3.set_title('95% interval pi1 parameters')
