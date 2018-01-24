from block import Block

class MuxBlock(Block):
    """
    This class implements the multiplexer object of a computational graph for hierarchical learning.

    """
    def __init__(self, name=None):

        self.block_lists = list()

        super(MuxBlock, self).__init__(name=name)


    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):
        """
                whatever the block does when activated by the computational graph.
                if the state is absorbing, fit is called for controllers

        """
        selector = inputs[0]
        if selector < 4:
            selector = 0
            print 'activating controller 1', absorbing, last
        else:
            print 'activating controller 2', absorbing, last
            selector = 1

        state = inputs[1]

        selected_block_list = self.block_lists[selector]
        other_block_list = self.block_lists[int(not(selector))]
        alarms = list()

        for block in selected_block_list:
            if block.reward_connection is not None:
                reward = block.reward_connection.last_output
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

        '''

        if last:
            for block in other_block_list:
                if not block.alarm_connections:
                    alarms.append(True)
                else:
                    for alarm_connection in block.alarm_connections:
                        alarms.append(alarm_connection.alarm_output)
                block(inputs=state, reward=reward, absorbing=absorbing, last=last, learn_flag=learn_flag, alarms=alarms)
                state = block.last_output
        '''
        self.last_output = selected_block_list[-1].last_output

        return absorbing, last

    def add_block_list(self, block_list):

        self.block_lists.append(block_list)

    def reset(self, inputs):
        selector = inputs[0]
        if selector < 4:
            selector = 0
        else:
            selector = 1
        for block_list in self.block_lists:
            state = inputs[1]
            for block in block_list:
                block.reset(inputs=state)
                state = block.last_output
        selected_block_list = self.block_lists[selector]
        self.last_output = selected_block_list[-1].last_output
        #self.clock_counter = 0



