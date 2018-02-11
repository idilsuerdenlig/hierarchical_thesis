import numpy as np
import matplotlib.pyplot as plt


def plot_arrows(act_max_q_val_tiled):

    plt.figure()
    ax = plt.axes()
    ax.set_xlim([0, 150])
    ax.set_ylim([0, 150])
    for i in xrange(20):
        for j in xrange(20):
            xs = 3.25 + i*7.5
            ys = 3.25 + j*7.5
            act_no = act_max_q_val_tiled[i, j]
            if act_no == 0:
                ax.arrow(xs, ys, 2, 0, head_width=0.75, head_length=1, fc='k', ec='k')

            elif act_no == 1:
                ax.arrow(xs, ys, 0, -2, head_width=0.75, head_length=1, fc='k', ec='k')

            elif act_no == 2:
                ax.arrow(xs, ys, -2, 0, head_width=0.75, head_length=1, fc='k', ec='k')

            elif act_no == 3:
                ax.arrow(xs, ys, 0, 2, head_width=0.75, head_length=1, fc='k', ec='k')

            elif act_no == 4:
                ax.arrow(xs, ys, 1.5, 1.5, head_width=0.75, head_length=1, fc='k', ec='k')

            elif act_no == 5:
                ax.arrow(xs, ys, 1.5, -1.5, head_width=0.75, head_length=1, fc='k', ec='k')

            elif act_no == 6:
                ax.arrow(xs, ys, -1.5, -1.5, head_width=0.75, head_length=1, fc='k', ec='k')

            else:
                ax.arrow(xs, ys, -1.5, 1.5, head_width=0.75, head_length=1, fc='k', ec='k')



