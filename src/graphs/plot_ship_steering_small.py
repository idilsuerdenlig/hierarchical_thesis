import matplotlib.pyplot as plt
import numpy as np

from common import *


base_dir = '/media/dave/af351f33-87a3-4142-a426-ad42d5930c4e/hierarchical/ship_steering_small_2018-07-25_14-07-57'

algs = ['GPOMDP',
        'PGPE',
        'REPS',
        'RWR',
        'H_GPOMDP_PGPE']

colors = ['b', 'r', 'g', 'c', 'm']

J_results = dict()

for alg in algs:
    J = np.load(base_dir + '/' + alg + '.npy')
    J_results[alg] = get_mean_and_confidence(J)

create_plot(algs, colors, J_results, 'J', legend=True)

plt.show()




