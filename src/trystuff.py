import numpy as np
from mushroom.utils.spaces import Box
from mushroom.features.basis import PolynomialBasis
import matplotlib.pyplot as plt
from library.visualization_tools.visualize_policy_parameters import visualize_policy_params
from library.visualization_tools.visualize_saved_in_server import visualize_saved_in_server
from mushroom.policy.gaussian_policy import *
from library.approximator.CMAC import *
from library.environments.idilshipsteering import ShipSteering
from mushroom.approximators import *
from mushroom.features import *
from mushroom.features.tiles import *
from mushroom.approximators.parametric.linear import *



visualize_saved_in_server(our_approach=True, how_many=2, small=False)
'''
mdp = ShipSteering(small=False, hard=True, n_steps_action=3)


# FeaturesL
high = [150, 150, np.pi, np.pi / 12]
low = [0, 0, -np.pi, -np.pi / 12]
n_tiles = [5, 5, 36, 5]
low = np.array(low, dtype=np.float)
high = np.array(high, dtype=np.float)
n_tilings = 9

tilingsL = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low, high=high)

featuresL = Features(tilings=tilingsL)
# Policy1
input_shape = (featuresL.size,)

approximator_params = dict(input_dim=input_shape[0])
approximator = Regressor(LinearApproximator, input_shape=input_shape,
                         output_shape=mdp.info.action_space.shape,
                         **approximator_params)
sigma = np.array([[1.3e-2]])
pi1 = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

for i in xrange(100000000):

    state = np.random.rand(40500)
    action = np.array([i-1000])
    pi1.diff_log(state, action)


'''

print np.load('success_per_thousand_eps.npy')