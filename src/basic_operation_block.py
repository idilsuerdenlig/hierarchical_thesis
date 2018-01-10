from f_block import fBlock
import numpy as np


class addBlock(fBlock):

    def __init__(self, wake_time, phi=None):
        def summation(inputs):
            res = 0
            for i in inputs:
                res += i
            return res
        self.phi = summation
        super(addBlock, self).__init__(wake_time=wake_time, phi=summation)


class extractBlock(fBlock):

    def __init__(self, wake_time, phi=None):
        def extraction(inputs):
            res = inputs[0] - inputs[1]
            return res
        self.phi = extraction
        super(extractBlock, self).__init__(wake_time=wake_time, phi=extraction)


class squarednormBlock(fBlock):

    def __init__(self, wake_time, phi=None):
        def squared_norm(inputs):
            res = 0
            for input in inputs:
                res += -input.dot(input)
            return res
        self.phi = squared_norm
        super(squarednormBlock, self).__init__(wake_time=wake_time, phi=squared_norm)
