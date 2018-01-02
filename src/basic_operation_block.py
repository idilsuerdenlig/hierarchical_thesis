from f_block import fBlock
import numpy as np


class plusBlock(fBlock):

    def __init__(self, wake_time, phi=None):
        def summation(inputs):
            res = 0
            for i in inputs:
                res += i
            return res
        self.phi = summation
        super(plusBlock, self).__init__(wake_time=wake_time, phi=summation)


class minusBlock(fBlock):

    def __init__(self, wake_time, phi=None):
        def extraction(inputs):
            res = inputs[0] - inputs[1]
            return res
        self.phi = extraction
        super(minusBlock, self).__init__(wake_time=wake_time, phi=extraction)


class dotproductBlock(fBlock):

    def __init__(self, wake_time, phi=None):
        def dot_product(inputs):
            res = -(inputs.dot(inputs))
            return res
        self.phi = dot_product
        super(dotproductBlock, self).__init__(wake_time=wake_time, phi=dot_product)
