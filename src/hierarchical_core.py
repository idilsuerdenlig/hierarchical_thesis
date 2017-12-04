from tqdm import tqdm
from computational_graph import ComputationalGraph

import numpy as np


class HierarchicalCore(object):
    """
    Implements the functions to run a generic computational graph.

    """
    def __init__(self, computational_graph, callbacks=None):

        self.computational_graph = computational_graph
        self.callbacks = callbacks if callbacks is not None else list()
        self._n_steps = None
        self._n_episodes = None

    def learn(self, n_steps=None, n_episodes=None, render=False, quiet=False):
        self._run(True, n_steps, n_episodes, render, quiet)

    def evaluate(self, n_steps=None, n_episodes=None, render=False, quiet=False):
        self._run(False, n_steps, n_episodes, render, quiet)

    def _run(self, learn_flag, n_steps, n_episodes, render, quiet):

        assert (n_episodes is not None and n_steps is None) or (n_episodes is None and n_steps is not None)
        if n_steps is not None:
            absorbing = True
            for step in tqdm(xrange(n_steps), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                if absorbing:
                    self.computational_graph.reset()

                absorbing = self.computational_graph.call_blocks(learn_flag=learn_flag)

        else:
            for episode in tqdm(xrange(n_episodes), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                self.computational_graph.reset()
                last = False
                while not last:
                    last = self.computational_graph.call_blocks(learn_flag=learn_flag)
