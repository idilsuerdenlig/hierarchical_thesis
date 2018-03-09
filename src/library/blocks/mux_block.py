from block import Block
import numpy as np
from memory_profiler import profile


class MuxBlock(Block):
    """
    This class implements the multiplexer object of a computational graph for hierarchical learning.

    """
    def __init__(self, name=None):

        self.block_lists = list()
        self.last_selection = None
        self.first = list()

        super(MuxBlock, self).__init__(name=name)


    def _call(self, inputs, reward, absorbing, last, learn_flag):
        selector = inputs[0]
        if selector < 4:
            selector = 0
            #print 'activating controller 1', absorbing, last
        else:
            #print 'activating controller 2', absorbing, last
            selector = 1

        state = inputs[1]

        selected_block_list = self.block_lists[selector]
        other_block_list = self.block_lists[int(not(selector))]
        alarms = list()

        if self.first[selector]:
            for block in selected_block_list:
                block.reset(inputs=inputs[1])
                self.first[selector] = False

        else:
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

        state = inputs[1]

        if self.last_selection is not None and self.last_selection != selector:
            for block in other_block_list:
                if block.reward_connection is not None:
                    reward = block.reward_connection.last_output[0]
                else:
                    reward = None
                block.last_call(inputs=state, reward=reward, absorbing=absorbing, learn_flag=learn_flag)
                state = block.last_output

        for block in other_block_list:
            block.alarm_output = False


        self.last_selection = selector

        self.last_output = selected_block_list[-1].last_output


    def add_block_list(self, block_list):

        self.block_lists.append(block_list)
        self.first.append(True)

    def reset(self, inputs):

        selector = inputs[0]
        if selector < 4:
            selector = 0
        else:
            selector = 1
        selected_block_list = self.block_lists[selector]

        state = inputs[1]
        for block in selected_block_list:
            block.reset(inputs=state)
            state = block.last_output
        self.first[selector] = False

        self.last_selection = selector

        self.last_output = state


    def init(self):
        for index in xrange(len(self.block_lists)):
            self.first[index] = True
        self.last_selection = None
        self.last_output = None
        for block_list in self.block_lists:
            for block in block_list:
                block.init()
