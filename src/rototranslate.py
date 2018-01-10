import numpy as np
from mushroom.utils.angles_utils import normalize_angle


def rototranslate(inputs):
    ##input0 from ControlH
    active_direction = np.where(inputs[0])[0]
    if active_direction == 0 or active_direction == 4:
        return inputs[1]
    elif active_direction == 1 or active_direction == 5:
        new_theta = normalize_angle(inputs[1][2] + np.pi/2)
        inputs[1][2] = new_theta
        return inputs[1]
    elif active_direction == 2 or active_direction == 6:
        new_theta = normalize_angle(inputs[1][2] + np.pi)
        inputs[1][2] = new_theta
        return inputs[1]
    else:
        new_theta = normalize_angle(inputs[1][2] + np.pi*1.5)
        inputs[1][2] = new_theta
        return inputs[1]
