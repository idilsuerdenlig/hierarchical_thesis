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

        assert (n_episodes is not None and n_steps is None) or (n_episodes is None and n_steps is not None)

        if n_steps is not None:
            for step in tqdm(xrange(n_steps), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                self.computational_graph.call_blocks(learn_flag=True)

        else:
            for episode in tqdm(xrange(n_episodes), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                absorbing = self.computational_graph.episode_start()
                while not absorbing:
                    absorbing = self.computational_graph.call_blocks(learn_flag=True)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False):

        assert (n_episodes is not None and n_steps is None) or (n_episodes is None and n_steps is not None)

        if n_steps is not None:
            for step in tqdm(xrange(n_steps), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                self.computational_graph.call_blocks(learn_flag=False)
        else:
            for episode in tqdm(xrange(n_episodes), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                last = False
                while not last:
                    _,last = self.computational_graph.call_blocks(learn_flag=False)
