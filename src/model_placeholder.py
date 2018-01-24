from block import Block

class PlaceHolder(Block):

    def __init__(self, name=None):
        super(PlaceHolder, self).__init__(name=name)

    def __call__(self, inputs, reward, absorbing, last, learn_flag, alarms):
        pass

    def reset(self,inputs):
        pass
        #return self.last_output




