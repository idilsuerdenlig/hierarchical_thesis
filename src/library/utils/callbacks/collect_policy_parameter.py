import numpy as np
from mushroom.policy import DeterministicPolicy

class CollectPolicyParameter:
    """
    This callback can be used to collect the values of a parameter
    (e.g. learning rate) during a run of the agent.

    """
    def __init__(self, policy):

        self._policy = policy
        self._p = list()

    def __call__(self, **kwargs):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        if isinstance(self._policy, DeterministicPolicy):
            value = self._policy.distribution.get_parameters()
        else:
            value = self._policy.get_weights()
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._p.append(value)

    def get_values(self):

        return self._p

    def reset(self):

        self._p = list()
