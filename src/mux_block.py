from block import Block

class MuxBlock(Block):
    """
    This class implements the multiplexer object of a computational graph for hierarchical learning.

    """
    def __init__(self):

        self.block_lists = list()

        super(MuxBlock, self).__init__(wake_time=1)


    def __call__(self, inputs, reward, absorbing, last, learn_flag):
        """
                whatever the block does when activated by the computational graph.
                if the state is absorbing, fit is called for controllers

        """
        selector = inputs[0]

        selected_block_list = self.block_lists[selector]

        for block in selected_block_list:
            absorbing, last = block(inputs=inputs[1], reward=None, absorbing=absorbing, learn_flag=learn_flag)
            inputs[1] = block.last_output

        self.last_output = inputs[1]

        return absorbing, last

    def add_block_list(self, block_list):

        self.block_lists.append(block_list)

    def reset(self, inputs):

        for block_list in self.block_lists:
            for block in block_list:
                block.reset(inputs=inputs[1])
                inputs[1] = block.last_output
        self.clock_counter = 0

