from control_block import ControlBlock
import numpy as np
from M_block import MBlock

class ComputationalGraph(object):
    """
    This class implements the computational graph for hierarchical learning.

    """
    def __init__(self, blocks, order):
        """
        Constructor.

        Args:
            blocks: the list of blocks that the graph consists of;
            wake_time: list integers. When the internal clock of the called block reaches its wake_time it
                        updates its return
            fit_time: list of integers. When the internal fit counter of the control blocks reaches the fit_time
                        they produce a new control signal
        """
        self._blocks = blocks
        self._order = order
        self._model = self._blocks[self._order[0]]
        self.absorbing = False


    def call_blocks(self, learn_flag):
        """
        executes the blocks in the diagram in the provided order. Always starts from the model.

        """

        for index in self._order:
            block = self._blocks[index]
            inputs = list()
            for input_block in block.input_connections:
                if input_block.last_output is not None:
                    inputs.append(input_block.last_output)
            if block.reward_connection == None:
                reward = None
            else:
                reward = block.reward_connection.get_reward()
            self.absorbing, last = block(inputs=inputs, reward=reward, absorbing=self.absorbing, learn_flag=learn_flag)

        return last
