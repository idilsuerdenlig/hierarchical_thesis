import numpy as np
from mushroom.utils.spaces import Box
from mushroom.features.basis import PolynomialBasis
import matplotlib.pyplot as plt


fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot([1,2,3,4,5], [10,5,10,5,10], 'r-')

ax2 = fig.add_subplot(2,2,2)
ax2.plot([1,2,3,4], [1,4,9,16], 'k-')

ax3 = fig.add_subplot(2,2,3)
ax3.plot([1,2,3,4], [1,10,100,1000], 'b-')

ax4 = fig.add_subplot(2,2,4)
ax4.plot([1,2,3,4], [0,0,1,1], 'g-')
