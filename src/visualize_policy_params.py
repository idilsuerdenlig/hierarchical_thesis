import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def VisualizePolicyParams(parameter_dataset1, parameter_dataset2):

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
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    time =np.arange(len(paramx_data))
    ax1.plot(paramx_data, paramy_data, time)
    ax1.scatter(paramx_data[0], paramy_data[0], time[0], marker='o',color='g')
    ax1.scatter(paramx_data[-1], paramy_data[-1], time[-1], marker='o', color='r')
    ax1.set_title('pi1 parameters')
    ax2.plot(parameter_dataset2)
    ax2.set_title('pi2 parameters')

