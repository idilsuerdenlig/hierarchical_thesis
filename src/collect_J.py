import numpy as np

class CollectJ:
    """
    This callback can be used to collect the values of a parameter
    (e.g. learning rate) during a run of the agent.

    """
    def __init__(self, gpomdp):

        self._gpomdp = gpomdp
        self._p = list()

    def __call__(self, **kwargs):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        J = list()
        for sample in dataset:
            _, _, r, _, _, last = self._parse(sample)
            if last:
                J.append(r)
        value = self._gpomdp._compute_gradient(J=J)
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._p.append(value)

    def get_values(self):

        return self._p


   def _parse(self, sample):
        """
        Utility to parse the sample.

        Args:
             sample (list): the current episode step.

        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag. If provided, `state` is preprocessed with the features.

        """
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]
        last = sample[5]

        return state, action, reward, next_state, absorbing, last
