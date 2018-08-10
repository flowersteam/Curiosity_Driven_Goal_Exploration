import numpy as np

from latentgoalexplo.actors.meta_actors import *


class RbfController(AbstractActor, IController):
    """This controller generates time-bounded action sequences using radial basis functions.
    """

    def __init__(self, *args, n_timesteps, n_action_dims, n_rbf, sdev, **kwargs):

        try:
            import scipy.ndimage
            globals()['scipy.ndimage'] = scipy.ndimage
        except:
            raise ImportError("You need scipy.ndimage to use class {}".format(self.__class__.__name__))

        # The array containing the atoms is created by filtering a multidimensional array
        # containing indicators at centers of atoms.
        # We make it larger to convolve outside of support and we cut it after
        self._bfs_params = np.zeros([int(n_timesteps * 1.25), n_action_dims, n_rbf])
        width = n_timesteps // (n_rbf)
        centers = np.cumsum([width] * n_rbf) + int(width // 4)
        base = np.array(range(n_rbf))
        self._bfs_params[centers, :, base] = 1.
        self._bfs_params = scipy.ndimage.gaussian_filter1d(self._bfs_params,
                                                           sdev,
                                                           mode='constant',
                                                           axis=0)
        self._bfs_params /= self._bfs_params.max()

        self._bfs_params = self._bfs_params[:n_timesteps, :, :]

        self._action_sequence = None

    def reset(self):

        pass

    def act(self, parameters):

        self._action_sequence = np.einsum('ijk,jk->ij', self._bfs_params, parameters)

    def terminate(self):

        pass

    @property
    def action_sequence(self):

        return self._action_sequence

    @classmethod
    def test(cls):

        pass