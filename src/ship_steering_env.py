import numpy as np
from matplotlib import pyplot as plt
from random import randint

from parsing import parse
from mushroom.environments import ship_steering



def experiment(trajAction):

    mdp = ship_steering.ShipSteering()
    initStates= np.array(mdp.reset())

    trajx=np.empty(len(trajAction)+1)
    trajy=np.empty(len(trajAction)+1)
    trajtheta=np.empty(len(trajAction)+1)
    trajthetadot=np.empty(len(trajAction)+1)
    trajreward=np.empty(len(trajAction))
    trajx[0] = initStates[0]
    trajy[0] = initStates[1]
    trajtheta[0] = initStates[2]
    trajthetadot[0] = initStates[3]

    for index in xrange(len(trajAction)):
        stepAction = np.array([trajAction[index]])
        stepStatevars, stepreward, _, _ = mdp.step(action=stepAction)
        trajx[index+1] = stepStatevars[0]
        trajy[index+1] = stepStatevars[1]
        trajtheta[index+1] = stepStatevars[2]
        trajthetadot[index+1] = stepStatevars[3]
        trajreward[index] = stepreward

    return trajx,trajy,trajtheta,trajthetadot,trajreward

dataSet = parse('/home/idil/hierarchical_thesis/src/ship_r_medium.log')
actionSet = dataSet[4]
xSet = dataSet[0]
ySet = dataSet[1]
thetaSet = dataSet[2]
thetadotSet = dataSet[3]
rewardSet =dataSet[5]

nTrials = 0
while nTrials < 10:

    i = randint(0,len(actionSet)-1)
    #i=-1 a trajectory in the dataset medium that ends outside limits.
    actionep=actionSet[i]
    listep=experiment(actionep)
    xep = listep[0]
    yep = listep[1]
    thetaep = listep[2]
    thetadotep = listep[3]
    rewardep = listep[4]
    xerr = xSet[i]-xep
    yerr = ySet[i]-yep
    thetaerr = thetaSet[i]-thetaep
    thetadoterr = thetadotSet[i]-thetadotep
    rewarderr = rewardSet[i] - rewardep
    nTrials=nTrials+1


time =np.arange(len(xerr))

fig = plt.figure()
axarr = fig.subplots(5, sharex=True)

axarr[0].plot(time,xerr)
axarr[0].set_ylabel('xerr')
axarr[1].plot(time,yerr)
axarr[1].set_ylabel('yerr')
axarr[2].plot(time,thetaerr)
axarr[2].set_ylabel('thetaerr')
axarr[3].plot(time,thetadoterr)
axarr[3].set_ylabel('thetadoterr')
axarr[4].plot(time[:-1], rewarderr)
axarr[4].set_ylabel('rewarderr')
axarr[4].set_xlabel('time')

plt.show()
