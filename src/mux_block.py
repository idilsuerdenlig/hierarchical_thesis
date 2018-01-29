from block import Block
from control_block import ControlBlock
import numpy as np

class MuxBlock(Block):
    """
    This class implements the multiplexer object of a computational graph for hierarchical learning.

    """
    def __init__(self, name=None):

        self.block_lists = list()
        self.last_selection = None

        super(MuxBlock, self).__init__(name=name)


    def _call(self, inputs, reward, absorbing, last, learn_flag):

        selector = inputs[0]
        if selector < 4:
            selector = 0
        else:
            selector = 1

        state = inputs[1]
        selected_block_list = self.block_lists[selector]
        other_block_list = self.block_lists[int(not(selector))]
        alarms = list()

        if self.last_selection is not None and self.last_selection != selector:
            block_list = self.block_lists[self.last_selection]
            for block in block_list:
                if block.reward_connection is not None:
                    reward = block.reward_connection.last_output[0]
                else:
                    reward = None

                block(inputs=state, reward=reward, absorbing=absorbing, last=last, learn_flag=learn_flag, alarms=False)
                state = block.last_output

        state = inputs[1]
        for block in selected_block_list:
            if block.reward_connection is not None:
                reward = block.reward_connection.last_output[0]
            else:
                reward = None
            if not block.alarm_connections:
                alarms.append(True)
            else:
                for alarm_connection in block.alarm_connections:
                    alarms.append(alarm_connection.alarm_output)

            block(inputs=state, reward=reward, absorbing=absorbing, last=last, learn_flag=learn_flag, alarms=alarms)
            state = block.last_output

        for block in other_block_list:
            block.alarm_output = False

        self.last_selection = selector

        self.last_output = selected_block_list[-1].last_output


    def add_block_list(self, block_list):

        self.block_lists.append(block_list)

    def reset(self):

        for block_list in self.block_lists:
            for block in block_list:
                block.reset()

        self.last_output = None
        self.last_selection = None




