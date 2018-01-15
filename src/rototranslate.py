import numpy as np
from mushroom.utils.angles_utils import normalize_angle


def rototranslate(inputs):
    new_states = np.zeros(shape=(4,))
    active_direction = inputs[0]
    x = inputs[1][0]
    y = inputs[1][1]
    theta = inputs[1][2]

    x0 = inputs[2][0]
    y0 = inputs[2][1]

    if active_direction == 0:   #R
        new_states[0] = x-x0+40
        new_states[1] = y-y0+75
    elif active_direction == 1: #D
        new_states[0] = y-y0+75
        new_states[1] = x-x0-40
        new_states[2] = normalize_angle(theta + np.pi/2)
    elif active_direction == 2: #L
        new_states[0] = x-x0-40
        new_states[1] = y-y0-75
        new_states[2] = normalize_angle(theta +np.pi)
    elif active_direction == 3: #U
        new_states[0] = y-y0+40
        new_states[1] = x-x0-75
        new_states[2] = normalize_angle(theta+1.5*np.pi)
    elif active_direction == 4: #UR
        new_states[0] = x-x0+40
        new_states[1] = y-y0+40
    elif active_direction == 5: #DR
        new_states[0] = y-y0-40
        new_states[1] = x-x0+40
        new_states[2] = normalize_angle(theta + np.pi/2)
    elif active_direction == 6: #DL
        new_states[0] = x-x0-40
        new_states[1] = x-x0-40
        new_states[2] = normalize_angle(theta + np.pi)
    else:                       #UL
        new_states[0] = y-y0+40
        new_states[1] = x-x0-40
        new_states[2] = normalize_angle(theta + np.pi*1.5)

    return new_states
