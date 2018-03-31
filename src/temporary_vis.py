from library.visualization_tools.visualize_ship_steering import visualize_ship_steering
import matplotlib.pyplot as plt
from library.visualization_tools.visualize_control_block import visualize_control_block
from library.visualization_tools.visualize_policy_parameters import visualize_policy_params
from library.utils.pick_last_ep_dataset import pick_last_ep
from library.utils.pick_eps import pick_eps
from library.utils.check_no_of_eps import check_no_of_eps

from mushroom.utils.dataset import compute_J
import numpy as np
from tqdm import tqdm


for i, run in enumerate(4):
    print(i, run)