import numpy as np
from mushroom.utils.angles_utils import normalize_angle


class rototranslate():

    def __init__(self, small):
        self.small = small

    def __call__(self, inputs):
        #print 'ROTOTRANSLATE INPUTS:    ', inputs
        #print 'ROTOTRANSLATE INPUTS[0]: ', inputs[0]
        #print 'ROTOTRANSLATE INPUTS[1]: ', inputs[1]
        #print 'ROTOTRANSLATE INPUTS[2]: ', inputs[2]

        new_states = np.zeros(shape=(4,))
        active_direction = inputs[0]
        x = inputs[1][0]
        y = inputs[1][1]
        theta = inputs[1][2]
        theta_dot = inputs[1][3]
        x0 = inputs[2][0]
        y0 = inputs[2][1]

        if self.small:
            small_offset = 4
            large_offset = 7.5
        else:
            small_offset = 40
            large_offset = 75

        if active_direction == 0:   #R
            new_states[0] = x-x0+small_offset
            new_states[1] = y-y0+large_offset
            new_states[2] = normalize_angle(theta)
        elif active_direction == 1: #D
            new_states[0] = y0-y+small_offset
            new_states[1] = x-x0+large_offset
            new_states[2] = normalize_angle(theta + np.pi/2)
        elif active_direction == 2: #L
            new_states[0] = x0-x+small_offset
            new_states[1] = y0-y+large_offset
            new_states[2] = normalize_angle(theta +np.pi)
        elif active_direction == 3: #U
            new_states[0] = y-y0+small_offset
            new_states[1] = x0-x+large_offset
            new_states[2] = normalize_angle(theta+1.5*np.pi)
        elif active_direction == 4: #UR
            new_states[0] = x-x0+small_offset
            new_states[1] = y-y0+small_offset
            new_states[2] = normalize_angle(theta)
        elif active_direction == 5: #DR
            new_states[0] = y0-y+small_offset
            new_states[1] = x-x0+small_offset
            new_states[2] = normalize_angle(theta + np.pi/2)
        elif active_direction == 6: #DL
            new_states[0] = x0-x+small_offset
            new_states[1] = y0-y+small_offset
            new_states[2] = normalize_angle(theta + np.pi)
        else:                       #UL
            new_states[0] = y-y0+small_offset
            new_states[1] = x0-x+small_offset
            new_states[2] = normalize_angle(theta + np.pi*1.5)

        new_states[3] = theta_dot
        #print 'active direction     : ', active_direction
        #print 'non_translated states:', inputs[1]
        #print 'initial_point        :', inputs[2]
        #print 'translated states    :', new_stat   es
        return new_states

