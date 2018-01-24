class Block(object):
    """
    This class implements the block object of a computational graph for hierarchical learning.

    """
    def __init__(self, name=None):
        """
        Constructor.

        Args:
            input_connections ([]block): the list of blocks that inputs the object block;
        """
        self.input_connections = list()
        self.reward_connection = None
        self.alarm_connections = list()
        #self.clock_counter = 0
        self.last_output = None
        #self.wake_time = wake_time
        self.name = name
        self.alarm_output = False

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):
        """
                whatever the block does when activated by the computational graph.
                if the state is absorbing, fit is called for controllers

        """
        raise NotImplementedError('Block is an abstract class')

    def add_input(self, block):
        self.input_connections.append(block)

    def add_alarm_connection(self, block):
        self.alarm_connections.append(block)

    def reset(self, inputs):
        raise NotImplementedError('Block is an abstract class')
