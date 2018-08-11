import matplotlib.pyplot as plt
import numpy as np

from common import *


base_dir = '../results/ship_steering_big_2018-08-06_16-23-52'
algs = ['H_GPOMDP_PGPE',
        'ghavamzadeh']

colors = ['b', 'r', 'g', 'c', 'm']

J_results = dict()
L_results = dict()

for alg in algs:
    J = np.load(base_dir + '/J_' + alg + '.npy')
    J_results[alg] = get_mean_and_confidence(J)

    L = np.load(base_dir + '/L_' + alg + '.npy')
    L_results[alg] = get_mean_and_confidence(L)

create_plot(algs, colors, J_results, 'J', legend=True)
create_plot(algs, colors, L_results, 'L', legend=True)

plt.show()




