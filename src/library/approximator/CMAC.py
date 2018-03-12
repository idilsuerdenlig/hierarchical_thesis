import numpy as np
from mushroom.features._implementations.tiles_features import TilesFeatures
#from memory_profiler import profile
class CMACApproximator:

    def __init__(self, tiles=None, weights=None, input_shape=None, output_shape=1,
                 **kwargs):

        assert len(input_shape) == 1 and len(output_shape) == 1
        assert tiles is not None

        self._phi = TilesFeatures(tiles)
        input_dim = input_shape[0]
        output_dim = output_shape[0]

        if weights is not None:
            self._w = weights.reshape((output_dim, -1))
        elif input_dim is not None:
            self._w = np.zeros((output_dim, self._phi.size))
        else:
            raise ValueError('You should specify the initial parameter vector'
                             ' or the input dimension')


    def fit(self, x, y, **fit_params):

        raise NotImplemetedError('fit not implemented')

    def predict(self, x, **predict_params):

        prediction = np.zeros((x.shape[0], self._w.shape[0]))
        for i, x_i in enumerate(x):
            offset = 0
            for tiling in self._phi._tiles:
                index = tiling(x_i)

                if index is not None:
                    prediction[i] = self._w.T[index + offset, :]

                offset += tiling.size


        return prediction

    @property
    def weights_size(self):
        return self._w.size

    def get_weights(self):
        return self._w.flatten()

    def set_weights(self, w):
        self._w = w.reshape(self._w.shape)

    def diff(self, state, action=None):

        if len(self._w.shape) == 1 or self._w.shape[0] == 1:
            return self._phi(state)
        else:
            n_phi = self._w.shape[1]
            n_outs = self._w.shape[0]

            if action is None:
                shape = (n_phi * n_outs, n_outs)
                df = np.zeros(shape)
                start = 0
                for i in xrange(n_outs):
                    stop = start + n_phi
                    df[start:stop, i] = self._phi(state)
                    start = stop
            else:
                shape = (n_phi * n_outs,)
                df = np.zeros(shape)
                start = action[0] * n_phi
                stop = start + n_phi
                df[start:stop] = self._phi(state)

            return df


