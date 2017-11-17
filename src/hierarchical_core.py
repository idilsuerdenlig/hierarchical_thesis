from tqdm import tqdm
from computational_graph import ComputationalGraph

import numpy as np


class HierarchicalCore(object):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, computational_graph, callbacks=None):
        """
        Constructor.

        Args:
            callbacks (list): list of callbacks to execute at the end of
                each learn iteration.
        """

        self.callbacks = callbacks if callbacks is not None else list()

        self.computational_graph = computational_graph
        self.computational_graph.initialize()




    def learn(self, n_iterations, render=False, quiet=False):

        for self.iteration in tqdm(xrange(n_iterations), dynamic_ncols=True,
                                   disable=quiet, leave=False):
            self.computational_graph.call_blocks(learn_flag = True)


    def evaluate(self, how_many=1, render=False, quiet=False):

        for self.iteration in tqdm(xrange(how_many), dynamic_ncols=True,
                                       disable=quiet, leave=False):
            self.computational_graph.call_blocks(learn_flag=False)



