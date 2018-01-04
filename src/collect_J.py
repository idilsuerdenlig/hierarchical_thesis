import numpy as np

class CollectJ:

    def __init__(self, gpomdp, **kwargs):

        self._gpomdp = gpomdp
        self._p = list()
        self.dataset = kwargs

    def __call__(self, ):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        J = list()
        for sample in self.dataset:
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

        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]
        last = sample[5]

        return state, action, reward, next_state, absorbing, last
