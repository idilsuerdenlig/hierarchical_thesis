import numpy as np
from matplotlib import pyplot as plt
from parsing import parse

dataSet = parse('/home/idil/hierarchical_thesis/src/ship_r_medium.log')
actionSet = dataSet[4]
xSet = dataSet[0]
ySet = dataSet[1]
thetaSet = dataSet[2]
thetadotSet = dataSet[3]
rewardSet =dataSet[5]

gamma = 0.99
Jlist=[]
rlast=[]
lentraj=[]
for i,rewardepisode in enumerate(rewardSet):
    rewardlist=[]
    for j,rewardstep in enumerate(rewardepisode):
        rewardstep=rewardstep*(gamma**j)
        rewardlist.append(rewardstep)
    rlast.append(rewardepisode[-1])
    J=np.sum(rewardlist)
    lentraj.append(len(rewardepisode))
    Jlist.append(J)
print Jlist[10]
print rlast[10]
print lentraj[10]

print Jlist[22]
print rlast[22]
print lentraj[22]
plt.plot(Jlist)
plt.title('Jlist')
fig=plt.figure()
plt.plot(rlast)
plt.title('rlast')
plt.show()

