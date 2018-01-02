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
            absorbing, last = block(inputs=inputs, reward=None, absorbing=absorbing, learn_flag=learn_flag)
            inputs = block.last_output

        return absorbing, last



