from block import Block

class fBlock(Block):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, wake_time):
        """
        Constructor.

        Args:
            input_connections ([]block): the list of blocks that inputs the object block;
        """
        super(fBlock, self).__init__(wake_time=wake_time)

    def __call__(self, inputs, reward, absorbing, learn_flag):
        """
                whatever the block does when activated by the computational graph.
                if the state is absorbing, fit is called for controllers
        """
        if self.wake_time == self.clock_counter:
            self.last_output = self.function()

        return self.last_output





