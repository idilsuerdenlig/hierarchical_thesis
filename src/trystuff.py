import numpy as np
from mushroom.utils.spaces import Box
from mushroom.features.basis import PolynomialBasis
import matplotlib.pyplot as plt
from visualize_policy_parameters import visualize_policy_params
from visualize_saved_in_server import visualize_saved_in_server

'''param1data = [[1, 1, 4, 5], [2, 2, 4, 5]], [[10, 10, 10, 10], [20, 20, 10, 5], [-5, -5, 5, 10]]
param2data = [1, 2, 3, 4, 5, 6, 7], [0, 0, 16, 15, 14]
visualize_policy_params(parameter_dataset1=param1data, parameter_dataset2=param2data, how_many=2)'''
visualize_saved_in_server(our_approach=False, how_many=1)
