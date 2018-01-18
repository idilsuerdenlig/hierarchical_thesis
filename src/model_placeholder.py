from block import Block

class PlaceHolder(Block):

    def __init__(self, name=None, wake_time=1):
        super(PlaceHolder, self).__init__(name=name, wake_time=wake_time)

    def __call__(self, inputs, reward, absorbing, last, learn_flag):
        pass

    def reset(self,inputs):
        return self.last_output




