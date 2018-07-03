import numpy as np
import matplotlib.pyplot as plt
import itertools
import inspect

from explauto.utils import prop_choice

from meta_actors import *
import environments

import sys
import os
PACKAGE_PARENT = '../ExplorationAlgorithms/'
SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from learning_module import LearningModule


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
        self._imsh = self._ax.imshow(np.random.randn(self._height, self._width, 3))
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


class KnnRegressor(AbstractActor, ITrainable, IDataset):
    """ A wrap around sklearn knn regressor.
    """

    def __init__(self, *args, n_neighbors=5, metric='euclidean', weights='distance', **kwargs):

        try:
            from sklearn.neighbors import KNeighborsRegressor
            globals()['KNeighborsRegressor'] = KNeighborsRegressor
        except:
            raise ImportError("You need sklearn.neighbors to use class {}".format(self.__class__.__name__))

        self._model = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric, weights=weights)

        # Needed since sklearn reset data at each fit call.
        self._X = None
        self._y = None
        self._prediction = None
        self._performance = None

    def reset(self, X_train, y_train):  # Depending on the model, it may be initialized with train values.

        self._X = X_train
        self._y = y_train
        self._model.fit(self._X, self._y)
        self._prediction = self._y
        self._performance = self._model.score(self._X, self._y)

    def act(self, *args, X_pred=None, X_train=None, y_train=None, X_test=None, y_test=None):

        if X_train is not None and X_pred is not None and X_test is not None:
            raise Exception("Calling multiple modes at once is not possible.")

        if X_train is not None:
            self._X = np.concatenate([self._X, X_train])
            self._y = np.concatenate([self._y, y_train])
            self._model.fit(self._X, self._y)
        elif X_test is not None:
            self._performance = self._model.score(X_test, y_test)
        elif X_pred is not None:
            self._prediction = self._model.predict(X_pred)

    def terminate(self):

        pass

    @property
    def prediction(self):

        return self._prediction

    @property
    def performance(self):

        return self._performance

    @property
    def dataset(self):

        return self._X, self._y

    @classmethod
    def test(cls):

        pass


class GaussianDistribution(AbstractActor, IDistribution):
    """ This actor implements a gaussian distribution.
    """

    def __init__(self, *args, latent_sample_sdev, **kwargs):
        self._dim = None
        self._sample = None
        self._sdev = latent_sample_sdev

    def reset(self, X):
        self._dim = X.shape[1]
        self._sample = self._sdev * np.random.randn(1, self._dim)

    def act(self, n_points=1):
        assert n_points > 0.

        self._sample = self._sdev * np.random.randn(n_points, self._dim)

    def terminate(self):
        pass

    @property
    def sample(self):
        return self._sample

    @classmethod
    def test(self):
        pass


class SamplerVAELatents(AbstractActor, IDistribution):
    """ This actor implements a gaussian distribution.
    """

    def __init__(self, *args, latent_sample_sdev, **kwargs):

        self._dim = None
        self._sample = None
        self._sdev = latent_sample_sdev

    def reset(self, X):

        self._dim = X.shape[1]

    def act(self, latents_values, indices, n_points=1):

        assert n_points > 0.

        self._sample = np.array([latents_values] * n_points)
        self._sample[:, [indices]] = self._sdev * np.random.randn(n_points, len(indices))

    def terminate(self):

        pass

    @property
    def sample(self):

        return self._sample

    @classmethod
    def test(self):

        pass


class RandomParameterizationExploration(AbstractActor, IExplorer):
    """Random Parameterization Exploration
    """

    def __init__(self, static_env, **kwargs):
        # assert issubclass(static_env, IStaticEnvironment)

        self._env = static_env(**kwargs)

        self._actions = None
        self._outcomes = None
        self._outcomes_states = None

    def reset(self):
        self._actions = []
        self._outcomes = []
        self._outcomes_states = []
        self._env.reset()

    def act(self, n_iter=1, **kwargs):
        assert n_iter > 0

        for i in range(n_iter):
            action = np.random.uniform(low=self._env.action_space[:, 0], high=self._env.action_space[:, 1])

            self._env.reset()
            self._env.act(action=action, **kwargs)

            outcome = self._env.observation

            self._actions.append(action)
            self._outcomes.append(outcome)

            outcome_state = self._env.hidden_state
            self._outcomes_states.append(outcome_state)

    def terminate(self):
        pass

    @property
    def environment(self):
        return self._env

    @property
    def actions(self):
        return self._actions

    @property
    def outcomes(self):
        return self._outcomes

    @property
    def outcomes_states(self):
        return self._outcomes_states

    @classmethod
    def test(cls):
        pass


class ActiveGoalExplorationUgl(AbstractActor, IExplorer):
    """Random Goal Exploration with unsupervised goal space learning.
    """

    def __init__(self, *args, static_env, representation, interest_model, n_explore, explo_ratio, n_modules,
                 explo_noise_sdev, win_size, s_bound=3., **kwargs):

        assert issubclass(static_env, IStaticEnvironment)
        # assert issubclass(representation, IRepresentation)
        # assert issubclass(representation, ITrainable)

        self._env = static_env(**kwargs)
        self._rep = representation(**kwargs)
        self._n_explore = n_explore
        self._explo_ratio = explo_ratio
        # We divide the explo noise by 2 to match explauto implementation with respect to our implementation
        self._explo_noise_sdev = explo_noise_sdev / 2

        self._n_latents = self._rep._n_latents

        assert (self._n_latents % n_modules == 0)

        self._interest_model = interest_model
        self._n_modules = n_modules
        self._win_size = win_size
        self._latents_env_config = dict(
                m_mins=[-1.] * len(self._env.action_space),
                m_maxs=[1.] * len(self._env.action_space),
                s_mins=[-s_bound] * self._n_latents,
                s_maxs=[s_bound] * self._n_latents
        )
        self._latents_env = environments.ExplautoEnv(**self._latents_env_config)
        self._learning_modules = []
        self._interests_evolution = []
        self._explo_evolution = []

        self._goals_states = []
        self._actions = None
        self._outcomes = None
        self._outcomes_train = None
        self._outcomes_reps = None
        self._outcomes_states = None
        self._attainable_points = None
        self._attainable_reps = None

    def reset(self, actions, outcomes, outcomes_states, outcomes_train, n_points=40):

        self._actions = actions
        self._outcomes = outcomes
        self._outcomes_states = outcomes_states
        self._outcomes_train = outcomes_train

        X = np.array(self._outcomes)
        y = np.array(self._actions)

        self._typical_img = self._env._observer.typical_img

        # Train the representation
        self._rep.reset(X_train=outcomes_train, y_train=outcomes_train, typical_img=self._typical_img)
        # Represent the set of outcomes
        self._rep.act(X_pred=X)
        self._outcomes_reps = [self._rep.representation[i] for i in range(self._rep.representation.shape[0])]

        # Define motor and sensory spaces:
        m_ndims = self._latents_env.conf.m_ndims  # number of motor parameters
        m_space = range(m_ndims)
        for i in range(self._n_modules):
            module_id = "mod" + str(i)
            s_mod = self._rep.sorted_latents[
                    i * self._n_latents // self._n_modules:(i + 1) * self._n_latents // self._n_modules] + m_ndims
            module = LearningModule(module_id, m_space, s_mod, self._latents_env.conf, explo_noise=self._explo_noise_sdev,
                                    win_size=self._win_size, interest_model=self._interest_model)
            self._learning_modules.append(module)

        for i, m in enumerate(actions):
            s = self._outcomes_reps[i]
            for module in self._learning_modules:
                module.update_sm(m, module.get_s(np.concatenate([m, s])))

        self._env.reset()

        if self._rep._network_type == 'fc':
            # We store the representation of attainable points
            attainable_space = [np.linspace(-1.0, 1.0, n_points) for i in range(2)]
            self._attainable_points = []
            for idx, coor in enumerate(itertools.product(*attainable_space)):
                self._env._observer.act(np.concatenate([[0, 0, 0, 0, 0, 0, 0], np.array(coor)]))
                self._attainable_points.append(self._env._observer.rendering)
            self._attainable_points = np.array(self._attainable_points)
            self._rep.act(X_pred=self._attainable_points)
            self._attainable_reps = self._rep.representation
        self._sorted_latents = self._rep._sorted_latents
        self._kld_latents = self._rep._kld_latents

    def load_representation(self, actions, outcomes, outcomes_states, outcomes_train, model_path, n_points=40):

        self._actions = actions
        self._outcomes = outcomes
        self._outcomes_states = outcomes_states
        self._outcomes_train = outcomes_train

        X = np.array(self._outcomes)
        y = np.array(self._actions)

        self._typical_img = self._env._observer.typical_img

        # Load the representation with pre-trained weights
        self._rep.load_model(model_path, typical_img=self._typical_img)
        self._rep.estimate_kld(outcomes_train, outcomes_train)
        # Represent the set of outcomes
        self._rep.act(X_pred=X)
        self._outcomes_reps = [self._rep.representation[i] for i in range(self._rep.representation.shape[0])]

        # Define motor and sensory spaces:
        m_ndims = self._latents_env.conf.m_ndims  # number of motor parameters
        m_space = range(m_ndims)
        for i in range(self._n_modules):
            module_id = "mod" + str(i)
            s_mod = self._rep.sorted_latents[
                    i * self._n_latents // self._n_modules:(i + 1) * self._n_latents // self._n_modules] + m_ndims
            module = LearningModule(module_id, m_space, s_mod, self._latents_env.conf, explo_noise=self._explo_noise_sdev,
                                    win_size=self._win_size, interest_model=self._interest_model)
            self._learning_modules.append(module)

        for i, m in enumerate(actions):
            s = self._outcomes_reps[i]
            for module in self._learning_modules:
                module.update_sm(m, module.get_s(np.concatenate([m, s])))

        self._env.reset()

        if self._rep._network_type == 'fc':
            # We store the representation of attainable points
            attainable_space = [np.linspace(-1.0, 1.0, n_points) for i in range(2)]
            self._attainable_points = []
            for idx, coor in enumerate(itertools.product(*attainable_space)):
                self._env._observer.act(np.concatenate([[0, 0, 0, 0, 0, 0, 0], np.array(coor)]))
                self._attainable_points.append(self._env._observer.rendering)
            self._attainable_points = np.array(self._attainable_points)
            self._rep.act(X_pred=self._attainable_points)
            self._attainable_reps = self._rep.representation
        self._sorted_latents = self._rep._sorted_latents
        self._kld_latents = self._rep._kld_latents

    def act(self, n_iter=1, **kwargs):

        assert n_iter > 0

        # Steps of (4 exploring and 1 exploiting iterations):
        for step in range(n_iter // (self._n_explore + 1)):
            # Compute the interest of modules
            interests = [module.interest() for module in self._learning_modules]
            self._interests_evolution.append(interests)
            # Choose the babbling module (probabilities proportional to interests, with epsilon of random choice):
            choice = prop_choice(interests, eps=self._explo_ratio)
            babbling_module = self._learning_modules[choice]
            # The babbling module picks a random goal in its sensory space and returns 4 noisy motor commands:
            m_list = babbling_module.produce(n=self._n_explore)
            goal = babbling_module.s
            _, indexes = babbling_module.sensorimotor_model.model.imodel.fmodel.dataset.nn_y(goal)
            self._goals_states.append(self._outcomes_states[indexes[0]])
            for m in m_list:
                # We perform the actions and observe outcomes
                self._env.reset()
                self._env.act(action=m, **kwargs)
                self._actions.append(m)
                outcome = self._env.observation
                # We represent the raw outcome
                self._rep.act(X_pred=outcome)
                s = self._rep.representation.ravel()
                # self._outcomes.append(outcome)
                self._outcomes_reps.append(s)
                self._outcomes_states.append(self._env.hidden_state)
                # Update each sensorimotor models:
                for module in self._learning_modules:
                    module.update_sm(m, module.get_s(np.concatenate([m, s])))
            # Choose the best motor command to reach current goal (with no noise):
            m = babbling_module.infer(babbling_module.expl_dims, babbling_module.inf_dims,
                                      babbling_module.x, n=1, explore=False)
            # We perform the action and observe outcomes
            self._env.reset()
            self._env.act(action=m, **kwargs)
            self._actions.append(m)
            outcome = self._env.observation
            # We represent the raw outcome
            self._rep.act(X_pred=outcome)
            s = self._rep.representation.ravel()
            # self._outcomes.append(outcome)
            self._outcomes_reps.append(s)
            self._outcomes_states.append(self._env.hidden_state)
            # Update the interest of the babbling module:
            babbling_module.update_im(m, babbling_module.get_s(np.concatenate([m, s])))
            # Update each sensorimotor models:
            for module in self._learning_modules:
                module.update_sm(m, module.get_s(np.concatenate([m, s])))
            explos_modules = [int(100. * (self._n_explore + 1) * module.im.n_points() / float(module.sm.t)) for module in
                              self._learning_modules]
            self._explo_evolution.append(explos_modules)

    def terminate(self):

        pass

    @property
    def environment(self):

        return self._env

    @property
    def actions(self):

        return self._actions

    @property
    def outcomes(self):

        return self._outcomes

    @property
    def outcomes_reps(self):

        return self._outcomes_reps

    @property
    def outcomes_states(self):

        return self._outcomes_states

    @property
    def attainable_reps(self):

        return self._attainable_reps

    @property
    def learning_modules(self):

        return self._learning_modules

    @property
    def interests_evolution(self):

        return self._interests_evolution

    @property
    def explo_evolution(self):

        return self._explo_evolution

    @property
    def goals_states(self):

        return self._goals_states

    @classmethod
    def test(cls):

        pass


if __name__ == '__main__':
    from armballs import *

    # We perform Bootstrap
    a = RandomParameterizationExploration(static_env=MyArmBalls, object_size=0.1, stick_length=0.4,
                                          stick_handle_tol=0.05, n_rbf=5, sdev=5., n_timesteps=50,
                                          width=64, height=64, rgb=False, render=False, env_noise=0.1)
    a.reset()
    a.act(n_iter=10, render=False)
