import numpy as np

class ComputationalGraph(object):
    """
    This class implements the computational graph for hierarchical learning.

    """
    def __init__(self, blocks, order, model):

        self.blocks = blocks
        self.order = order
        self.model = model
        self.state = list()
        self.reward = None
        self.absorbing = False
        self.last = False
        self.dataset_eval = list()
        self.step_counter = 0

    def call_blocks(self, learn_flag):
        """
        executes the blocks in the diagram in the provided order. Always starts from the model.

        """
        action = self.blocks[self.order[-1]].last_output
        self.state, self.reward, self.absorbing, _ = self.model.step(action)
        self.last = self.step_counter >= self.model.info.horizon or self.absorbing
        self.step_counter += 1
        self.blocks[0].last_output = self.state
        self.blocks[1].last_output = self.reward
        for index in self.order:
            block = self.blocks[index]
            inputs = list()
            for input_block in block.input_connections:
                if input_block.last_output is not None:
                    inputs.append(input_block.last_output)
            if block.reward_connection is None:
                reward = None
            else:
                reward = block.reward_connection.last_output
            block(inputs=inputs, reward=reward, absorbing=self.absorbing, last=self.last, learn_flag=learn_flag)
        return self.absorbing, self.last

    def reset(self):
        self.state = self.model.reset()
        self.blocks[0].last_output = self.state
        self.blocks[1].last_output = None
        for index in self.order:
            block = self.blocks[index]
            inputs = list()
            for input_block in block.input_connections:
                if input_block.last_output is not None:
                    inputs.append(input_block.last_output)
            block.reset(inputs=np.array(inputs))
        self.step_counter = 0

    def get_sample(self):
        state = self.blocks[0].last_output
        action = self.blocks[self.order[-1]].last_output
        rew_last = self.blocks[1].last_output
        abs = self.absorbing
        last = self.last
        return state, action, rew_last, abs, last
