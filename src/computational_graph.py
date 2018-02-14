import numpy as np
from topological_sort import topological_sort
class ComputationalGraph(object):
    """
    This class implements the computational graph for hierarchical learning.

    """
    def __init__(self, blocks, model):

        self.ordered = topological_sort(blocks)
        self.model = model
        self.state = list()
        self.reward = None
        self.absorbing = False
        self.first = True
        self.last = False
        self.dataset_eval = list()
        self.step_counter = 0

    def call_blocks(self, learn_flag):
        """
        executes the blocks in the diagram in the provided order. Always starts from the model.

        """
        action = self.ordered[-1].last_output
        self.state, self.reward, self.absorbing, _ = self.model.step(action)
        self.step_counter += 1
        self.last = self.step_counter >= self.model.info.horizon or self.absorbing
        self.ordered[0].last_output = self.state
        self.ordered[1].last_output = np.array([self.reward])
        #print 'ENV STATE, REW', self.state, self.reward
        for block in self.ordered:
            #print 'NAME  :',block.name
            #print 'ORDER :',index
            inputs = list()
            alarms = list()
            #print 'INPUTS: '
            for input_block in block.input_connections:
                #print input_block.name
                if input_block.last_output is not None:
                    inputs.append(input_block.last_output)
            if not block.alarm_connections:
                alarms.append(True)
            else:
                for alarm_connection in block.alarm_connections:
                    alarms.append(alarm_connection.alarm_output)
            if block.reward_connection is None:
                reward = None
            else:
                reward = block.reward_connection.last_output[0]
            block(inputs=inputs, reward=reward, absorbing=self.absorbing, last=self.last, learn_flag=learn_flag, alarms=alarms)
        return self.absorbing, self.last

    def reset(self):
        self.state = self.model.reset()
        self.ordered[0].last_output = self.state
        self.ordered[1].last_output = None
        #print 'ENV RESET STATE, REW', self.state, self.reward

        for block in self.ordered:
            inputs = list()
            for input_block in block.input_connections:
                if input_block.last_output is not None:
                    inputs.append(input_block.last_output)
            block.reset(inputs=inputs)
        self.step_counter = 0

    def get_sample(self):
        state = self.ordered[0].last_output
        action = self.ordered[-1].last_output
        rew_last = self.ordered[1].last_output
        abs = self.absorbing
        last = self.last
        return state, action, rew_last, abs, last

    def init(self):
        for block in self.ordered:
            block.init()
        self.step_counter = 0
