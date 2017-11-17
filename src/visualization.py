import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from parsing import parse

print "parsing..."



dataList = parse('/home/idil/hierarchical_thesis/src/ship_r_medium.log')

xs=100
xe=120
ys=120
ye=100

print "plotting..."
fig = plt.figure()
axarr = fig.subplots(4, sharex=True)

for i in range(len(dataList[0])):
    x=np.array(dataList[0][i])
    y=np.array(dataList[1][i])
    theta=np.array(dataList[2][i])
    thetadot=np.array(dataList[3][i])
    time =np.arange(len(x))
    axarr[0].plot(time,x)
    axarr[0].set_ylabel('x')
    axarr[1].plot(time,y)
    axarr[1].set_ylabel('y')
    axarr[2].plot(time,theta)
    axarr[2].set_ylabel('theta')
    axarr[3].plot(time,thetadot)
    axarr[3].set_ylabel('thetadot')
    axarr[3].set_xlabel('time')


fig2 = plt.figure()
ax = fig2.gca(projection='3d')

maxt=0
for i in range(len(dataList[0])):
    x=np.array(dataList[0][i])
    y=np.array(dataList[1][i])
    time =np.arange(len(x))
    plt.plot(x,y,time)
    maxt = max(maxt,len(x))


print maxt
xg=[xs,xe,xe,xs]
yg=[ys,ye,ye,ys]
zg=[0,0,maxt,maxt]
verts = [list(zip(xg,yg,zg))]

print verts


ax.add_collection3d(Poly3DCollection(verts))

plt.show()

