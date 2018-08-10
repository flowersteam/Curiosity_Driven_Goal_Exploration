import numpy as np
import matplotlib.pyplot as plt

from latentgoalexplo.actors.meta_actors import *


class MatplotlibInteractiveRendering(AbstractActor):
    """Check you used the `%matplotlib notebook` magic
    """

    def __init__(self, renderer, *args, width=600, height=400, figsize=(5, 5), **kwargs):
        self._renderer = renderer(width=width, height=height, **kwargs)
        self._width = width
        self._height = height
        self._figsize = figsize

        self._fig = None
        self._ax = None
        self._imsh = None

    def reset(self):
        self._renderer.reset()
        self._fig = plt.figure(figsize=self._figsize)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._imsh = self._ax.imshow(np.random.uniform(0, 1, (self._height, self._width, 3)))
        plt.show()

    def act(self, **kwargs):
        self._renderer.act(**kwargs)
        self._imsh.set_array(self._renderer.rendering)
        self._fig.canvas.draw()

    def terminate(self):
        pass

    @classmethod
    def test(cls):
        pass


class MatplotlibInteractiveScatterRendering(AbstractActor):
    """This allows to render the ArmBall Environment
    """

    def __init__(self, *args, width=600, height=400, figsize=(5, 5), **kwargs):
        self._width = width
        self._height = height
        self._figsize = figsize

        self._fig = None
        self._ax = None
        self._imsh = None

    def reset(self):
        self._fig = plt.figure(figsize=self._figsize)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._imsh = self._ax.scatter(np.random.randn(1), np.random.randn(1))
        plt.show()

    def act(self, X, Y):
        self._imsh.remove()
        self._imsh = self._ax.scatter(X, Y, c=range(X.shape[0]))
        self._fig.canvas.draw()

    def terminate(self):
        pass

    @classmethod
    def test(cls):
        pass
