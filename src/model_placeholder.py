from block import Block

class PlaceHolder(Block):

    def __init__(self, wake_time=1):
        super(PlaceHolder, self).__init__(wake_time=1)

    def __call__(self, inputs, reward, absorbing, last, learn_flag):
        pass

    def reset(self,inputs):
        return self.last_output




