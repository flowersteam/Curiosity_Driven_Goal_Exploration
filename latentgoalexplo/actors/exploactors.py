import numpy as np
import itertools

from explauto.utils import prop_choice

from latentgoalexplo.actors.meta_actors import *

from latentgoalexplo.curiosity.learning_module import LearningModule
from latentgoalexplo.environments.explautoenv import ExplautoEnv


class RandomParameterizationExploration(AbstractActor, IExplorer):
    """Random Parameterization Exploration
    """

    def __init__(self, static_env, **kwargs):

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
        self._latents_env = ExplautoEnv(**self._latents_env_config)
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

    def reset(self, actions, outcomes, outcomes_states, outcomes_train):

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

        self._sorted_latents = self._rep._sorted_latents
        self._kld_latents = self._rep._kld_latents

    def load_representation(self, actions, outcomes, outcomes_states, outcomes_train, model_path):

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

        self._sorted_latents = self._rep._sorted_latents
        self._kld_latents = self._rep._kld_latents

    def use_representation(self, actions, outcomes, outcomes_states, outcomes_train, representation):

        self._actions = actions
        self._outcomes = outcomes
        self._outcomes_states = outcomes_states
        self._outcomes_train = outcomes_train

        X = np.array(self._outcomes)
        y = np.array(self._actions)

        self._typical_img = self._env._observer.typical_img

        # Load the representation with pre-trained weights
        self._rep = representation
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
