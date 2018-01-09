import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_ship_steering(datalist_eval, name, J=None, range_eps=None):

    plt.figure()

    ax1 = plt.subplot2grid((4,3), (0,0), rowspan=4, projection='3d')
    ax2 = plt.subplot2grid((4,3), (0,1))
    ax3 = plt.subplot2grid((4,3), (1,1))
    ax4 = plt.subplot2grid((4,3), (2,1))
    ax5 = plt.subplot2grid((4,3), (3,1))
    ax6 = plt.subplot2grid((4,3), (0,2), rowspan=4)

    xs = 100
    xe = 120
    ys = 120
    ye = 100

    x_ep = list()
    y_ep = list()
    theta_ep = list()
    thetadot_ep = list()
    action_ep = list()
    reward_ep = list()
    size_eps = list()
    x_list = list()
    y_list = list()
    theta_list = list()
    thetadot_list = list()
    ep_size = 0
    n_eps = 0

    for dataset_step in datalist_eval:
        if not dataset_step[-1]:
            states_step = dataset_step[0]
            action_step = dataset_step[1]
            reward_step = dataset_step[2]
            x_step = states_step[0]
            y_step = states_step[1]
            theta_step = states_step[2]
            thetadot_step = states_step[3]

            x_ep.append(x_step)
            y_ep.append(y_step)
            theta_ep.append(theta_step)
            thetadot_ep.append(thetadot_step)
            reward_ep.append(reward_step)
            action_ep.append(action_step)
            ep_size += 1
        else:
            size_eps.append(ep_size)
            ep_size = 0
            x_ep.append(dataset_step[3][0])
            y_ep.append(dataset_step[3][1])
            theta_ep.append(dataset_step[3][2])
            thetadot_ep.append(dataset_step[3][3])
            x_list.append(x_ep)
            y_list.append(y_ep)
            theta_list.append(theta_ep)
            thetadot_list.append(thetadot_ep)
            x_ep = []
            y_ep = []
            theta_ep = []
            thetadot_ep = []
            action_ep = []
            reward_ep = []
            n_eps += 1


    maxt = 0
    if range_eps is None:
        range_eps = xrange(len(x_list))

    for episode in range_eps:
        time =np.arange(len(x_list[episode]))
        x = x_list[episode]
        y = y_list[episode]
        ax1.plot(x,y,time)
        maxt = max(maxt,len(x))

    xg=[xs, xe, xe, xs]
    yg=[ys, ye, ye, ys]
    zg=[0, 0, maxt, maxt]
    verts = [list(zip(xg,yg,zg))]
    ax1.set_xlim([0,160])
    ax1.set_ylim([0,160])

    ax1.add_collection3d(Poly3DCollection(verts),zs=zg)

    for episode in range_eps:
        x_ep = x_list[episode]
        y_ep = y_list[episode]
        theta_ep = theta_list[episode]
        thetadot_ep = thetadot_list[episode]
        time =np.arange(len(x_ep))
        ax2.plot(time,x_ep)
        ax2.set_ylabel('x')
        ax3.plot(time,y_ep)
        ax3.set_ylabel('y')
        ax4.plot(time,theta_ep)
        ax4.set_ylabel('theta')
        ax5.plot(time,thetadot_ep)
        ax5.set_ylabel('thetadot')
        ax5.set_xlabel('time')

    ax6.plot(size_eps)

    if J is not None:
        fig4=plt.figure()
        plt.plot(J)

        axes=fig4.gca()
        axes.set_ylim([-700, -50])
        plt.title('J')

    plt.tight_layout()

    plt.suptitle(name)

