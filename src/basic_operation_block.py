from f_block import fBlock
import numpy as np


class addBlock(fBlock):

    def __init__(self, name=None, phi=None):
        def summation(inputs):
            res = np.zeros(1)
            for i in inputs:
                res += i
            return res
        self.phi = summation
        super(addBlock, self).__init__(phi=summation, name=name)



class squarednormBlock(fBlock):

    def __init__(self, name=None, phi=None):
        def squared_norm(inputs):
            res = np.zeros(1)
            for input in inputs:
                res += -input.dot(input)
            return res
        self.phi = squared_norm
        super(squarednormBlock, self).__init__(phi=squared_norm, name=name)
