import numpy as np

from latentgoalexplo.actors.meta_actors import *


class FixedEpisodeDynamizer(AbstractActor, IEpisodicEnvironment, IEpisodicRewarding):
    """This actor allows to dynamize an environment for a fixed number of iterations.
    """

    def __init__(self, *args, static_env, n_iter, **kwargs):
        # assert issubclass(static_env, IStaticEnvironment)
        # assert issubclass(static_env, IRewarding)

        self._static_env = static_env(*args, **kwargs)
        self._n_iter = n_iter

        self._observation_sequence = None
        self._reward_sequence = None

    def reset(self):
        self._static_env.reset()
        self._observation_sequence = np.repeat(self._static_env.observation.reshape(1, -1),
                                               repeats=self._n_iter,
                                               axis=0)
        self._reward_sequence = np.array([self._static_env.reward] * self._n_iter)

    def act(self, action_sequence):
        for i, action in enumerate(action_sequence):
            self._static_env.act(action=action)

            self._observation_sequence[i] = self._static_env.observation

            self._reward_sequence[i] = self._static_env.reward

    def terminate(self):
        pass

    @property
    def observation_sequence(self):
        return self._observation_sequence

    @property
    def reward_sequence(self):
        return self._reward_sequence

    @classmethod
    def test(cls):
        pass