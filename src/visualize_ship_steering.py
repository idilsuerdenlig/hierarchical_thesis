import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualizeShipSteering(datalist_eval, J):

    xs=100
    xe=120
    ys=120
    ye=100

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
    i = 0

    print 'list to eps...'
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
            i+=1

        else:
            size_eps.append(i)
            i=0
            x_list.append(x_ep)
            y_list.append(y_ep)
            theta_list.append(theta_ep)
            thetadot_list.append(thetadot_ep)
            x_ep = []
            y_ep = []
            theta_ep = []
            thetadot_ep = []


    print 'plotting 3d...'
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    maxt=0
    for episode in xrange(len(x_list)):
        time =np.arange(len(x_list[episode]))
        x = np.array(x_list[episode])
        y = np.array(y_list[episode])
        plt.plot(x,y,time)
        maxt = max(maxt,len(x))

    xg=[xs, xe, xe, xs]
    yg=[ys, ye, ye, ys]
    zg=[0, 0, maxt, maxt]
    verts = [list(zip(xg,yg,zg))]
    ax.set_xlim([0,160])
    ax.set_ylim([0,160])

    ax.add_collection3d(Poly3DCollection(verts),zs=zg)

    print 'plotting 2d state plots'
    fig3 = plt.figure()
    axarr = fig3.subplots(4, sharex=True)

    for episode in xrange(len(x_list)):
        x_ep = x_list[episode]
        y_ep = y_list[episode]
        theta_ep = theta_list[episode]
        thetadot_ep = thetadot_list[episode]
        time =np.arange(len(x_ep))
        axarr[0].plot(time,x_ep)
        axarr[0].set_ylabel('x')
        axarr[1].plot(time,y_ep)
        axarr[1].set_ylabel('y')
        axarr[2].plot(time,theta_ep)
        axarr[2].set_ylabel('theta')
        axarr[3].plot(time,thetadot_ep)
        axarr[3].set_ylabel('thetadot')
        axarr[3].set_xlabel('time')

    print 'plotting length of episodes...'
    fig2 = plt.figure()
    plt.plot(size_eps)
    plt.title('size_eps')

    print 'plotting J for each episode...'

    fig4=plt.figure()
    plt.plot(J)

    axes=fig4.gca()
    axes.set_ylim([-700, -50])
    plt.title('J')

    plt.show()



