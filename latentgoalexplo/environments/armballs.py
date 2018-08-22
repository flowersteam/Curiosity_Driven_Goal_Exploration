import numpy as np
import gizeh

from latentgoalexplo.actors.meta_actors import *
from latentgoalexplo.common.rendering import MatplotlibInteractiveRendering
from latentgoalexplo.actors.controllers import RbfController
from latentgoalexplo.environments.utils import FixedEpisodeDynamizer


class ArmBalls(AbstractActor, IStaticEnvironment, IRewarding):
    """The Armball environment.
    """

    def __init__(self, *args, object_initial_pose=np.array([0.6, 0.6]), object_size=0.2,
                 object_rewarding_pose=np.array([-0.6, -0.6]),
                 arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]),
                 distract_initial_pose=np.array([0.7, -0.45]), distract_size=0.15,
                 distract_noise=0.2, stochastic=False, **kwargs):

        assert arm_lengths.size < 8, "The number of joints must be inferior to 8"
        assert arm_lengths.sum() == 1., "The arm length must sum to 1."

        # We set the parameters
        self._n_joints = arm_lengths.size
        self._arm_lengths = arm_lengths
        self._stochastic = stochastic
        self._object_initial_pose = object_initial_pose
        self._object_rewarding_pose = object_rewarding_pose
        self._object_size = object_size
        self._distract_size = distract_size
        self._distract_initial_pose = distract_initial_pose
        self._distract_noise = distract_noise
        self._actual_arm_pose = np.zeros(self._arm_lengths.shape)
        self._hand_pos = np.zeros(2)
        self._object_handled = False

        # We set the space
        self.observation_space = np.array([[-1, 1]] * (len(self._arm_lengths) + 6))
        self.action_space = np.array([[-1, 1]] * self._n_joints)

        # We set to None to rush error if reset not called
        self._reward = None
        self._observation = None

    def reset(self):

        # We reset the simulation
        if self._stochastic:
            self._object_initial_pose = np.random.uniform(-0.9, 0.9, 2)
        self._actual_object_pose = self._object_initial_pose.copy()
        self._actual_distract_pose = self._distract_initial_pose.copy()
        self._actual_arm_pose = np.zeros(self._arm_lengths.shape)
        self._object_handled = False
        angles = np.cumsum(self._actual_arm_pose)
        angles_rads = np.pi * angles
        self._hand_pos = np.array([np.sum(np.cos(angles_rads) * self._arm_lengths),
                                   np.sum(np.sin(angles_rads) * self._arm_lengths)])
        self._observation = np.concatenate([self._actual_arm_pose, self._hand_pos, self._actual_distract_pose,
                                            self._actual_object_pose])

        # We compute the initial reward.
        self._reward = np.linalg.norm(self._actual_object_pose - self._object_rewarding_pose, ord=2)

    def act(self, action=np.array([0., 0., 0., 0., 0., 0., 0.])):
        """Perform an agent action in the Environment
        """

        assert action.shape == self.action_space.shape[0:1]
        assert (action >= self.action_space[:, 0]).all()
        assert (action <= self.action_space[:, 1]).all()

        # We compute the position of the end effector
        self._actual_arm_pose = action
        angles = np.cumsum(self._actual_arm_pose)
        angles_rads = np.pi * angles
        self._hand_pos = np.array([np.sum(np.cos(angles_rads) * self._arm_lengths),
                                   np.sum(np.sin(angles_rads) * self._arm_lengths)])

        # We check if the object is handled and we move it.
        if np.linalg.norm(self._hand_pos - self._actual_object_pose, ord=2) < self._object_size:
            self._object_handled = True
        if self._object_handled:
            self._actual_object_pose = self._hand_pos

        # We move the distractor
        self._actual_distract_pose = self._actual_distract_pose + np.random.randn(2) * self._distract_noise
        self._actual_distract_pose = np.clip(self._actual_distract_pose, -.95, 0.95)

        # We update observation and reward
        self._observation = np.concatenate([self._actual_arm_pose, self._hand_pos, self._actual_distract_pose,
                                            self._actual_object_pose])
        self._reward = np.linalg.norm(self._actual_object_pose - self._object_rewarding_pose, ord=2)

    def terminate(self):

        # We reset stuffs to None to generate errors if called
        self._n_joints = None
        self._arm_lengths = None
        self._object_initial_pose = None
        self._object_rewarding_pose = None
        self._object_size = None
        self._actual_object_pose = None
        self._actual_arm_pose = None
        self._object_handled = None
        self._reward = None
        self._observation = None

    @property
    def observation(self):

        return self._observation

    @property
    def reward(self):

        return self._reward

    @classmethod
    def test(cls):
        pass


class ArmBallsRenderer(AbstractActor, IRenderer):
    """This allows to render the ArmBall Environment
    """

    def __init__(self, *args, width=600, height=400, rgb=True, render_arm=True,
                 arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]), object_size=0.2,
                 distract_size=0.15, env_noise=0., distract_first=False, interpolate=True, **kwargs):

        self._width = width
        self._height = height
        self._rgb = rgb
        self._env_noise = env_noise
        self._interpolate = interpolate
        self._distract_first = distract_first

        self._arm_lengths = arm_lengths
        self._object_size = object_size
        self._distract_size = distract_size
        self._render_arm = render_arm

        # We set the spaces
        self.action_space = np.array([[-1, 1]] * (len(self._arm_lengths) + 6))

        self._rendering = None
        self._typical_img = None

    def reset(self):

        if self._rgb:
            self._rendering = np.zeros([self._height, self._width, 3])
            self._rendering[0] = 1
        else:
            self._rendering = np.zeros([self._height, self._width])
            self._rendering[0] = 1

        self.act(observation=np.concatenate([np.zeros(len(self._arm_lengths)), [0, 1, 0.2, 0.2, -0.1, -0.1]]))
        self._typical_img = self._rendering

    def act(self, observation=np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., -0.6, 0.4, .6, .6]),
            render_goal=False, goal=None, render_hand=False, render_object=True, render_distract=True):

        assert len(observation) == len(self._arm_lengths) + 2 + 2 + 2

        # We retrieve arm and object pose
        arm_pose = observation[:len(self._arm_lengths)]
        distract_pose = observation[-4: -2]
        object_pose = observation[-2:]

        # World parameters
        world_size = 2.
        arm_angles = np.cumsum(arm_pose)
        arm_angles = np.pi * arm_angles
        arm_points = np.array([np.cumsum(np.cos(arm_angles) * self._arm_lengths),
                               np.cumsum(np.sin(arm_angles) * self._arm_lengths)])
        hand_pos = np.array([np.sum(np.cos(arm_angles) * self._arm_lengths),
                             np.sum(np.sin(arm_angles) * self._arm_lengths)])

        # Screen parameters
        screen_width = self._width
        screen_height = self._height
        screen_center_w = np.ceil(self._width / 2)
        screen_center_h = np.ceil(self._height / 2)

        # Ratios
        world2screen = min(screen_width / world_size, screen_height / world_size)

        # Instantiating surface
        surface = gizeh.Surface(width=screen_width, height=screen_height)

        # Drawing object
        if render_object:
            if self._distract_first is False:
                objt = gizeh.circle(r=self._object_size * world2screen,
                                    xy=(screen_center_w + object_pose[0] * world2screen,
                                        screen_center_h + object_pose[1] * world2screen),
                                    fill=(1, 1, 0))
                objt.draw(surface)
            else:
                objt = gizeh.circle(r=self._object_size * world2screen,
                                    xy=(screen_center_w + object_pose[0] * world2screen,
                                        screen_center_h + object_pose[1] * world2screen),
                                    fill=(1, 0, 0))
                objt.draw(surface)

        # Drawing distractor
        if render_distract:
            if self._distract_first is False:
                objt = gizeh.circle(r=self._distract_size * world2screen,
                                    xy=(screen_center_w + distract_pose[0] * world2screen,
                                        screen_center_h + distract_pose[1] * world2screen),
                                    fill=(0, 0, 1))
                objt.draw(surface)
            else:
                objt = gizeh.circle(r=self._distract_size * world2screen,
                                    xy=(screen_center_w + distract_pose[0] * world2screen,
                                        screen_center_h + distract_pose[1] * world2screen),
                                    fill=(0, 1, 1))
                objt.draw(surface)

        # Drawing goal
        if render_goal == True:
            objt = gizeh.circle(r=self._object_size * world2screen / 4,
                                xy=(screen_center_w + goal[0] * world2screen,
                                    screen_center_h + goal[1] * world2screen),
                                fill=(1, 0, 0))
            objt.draw(surface)

        # Drawing hand
        if render_hand == True:
            objt = gizeh.circle(r=self._object_size * world2screen / 2,
                                xy=(screen_center_w + hand_pos[0] * world2screen,
                                    screen_center_h + hand_pos[1] * world2screen),
                                fill=(1, 0, 0))
            objt.draw(surface)

        # Drawing arm
        if self._render_arm:
            screen_arm_points = arm_points * world2screen
            screen_arm_points = np.concatenate([[[0., 0.]], screen_arm_points.T], axis=0) + \
                                np.array([screen_center_w, screen_center_h])
            arm = gizeh.polyline(screen_arm_points, stroke=(0, 1, 0), stroke_width=3.)
            arm.draw(surface)

        if self._rgb:
            self._rendering = surface.get_npimage().astype(np.float32)
            self._rendering -= self._rendering.min()
            self._rendering /= self._rendering.max()
            if self._env_noise > 0:
                self._rendering = np.random.normal(self._rendering, self._env_noise)
                self._rendering -= self._rendering.min()
                self._rendering /= self._rendering.max()
        else:
            self._rendering = surface.get_npimage().astype(np.float32).sum(axis=-1)
            self._rendering -= self._rendering.min()
            self._rendering /= self._rendering.max()
            if self._env_noise > 0:
                self._rendering = np.random.normal(self._rendering, self._env_noise)
                self._rendering -= self._rendering.min()
                self._rendering /= self._rendering.max()
            # Added by Adrien, makes training easier
            # self._rendering = -self._rendering + 1
            if not self._interpolate:
                self._rendering[self._rendering < 0.5] = 0
                self._rendering[self._rendering >= 0.5] = 1

    def terminate(self):

        pass

    @property
    def rendering(self):

        return self._rendering

    @property
    def observation_space(self):

        return self._observation_space

    @property
    def typical_img(self):

        return self._typical_img

    @classmethod
    def test(cls):

        pass


class MyArmBalls(AbstractActor, IStaticEnvironment):
    """
    This is an example of a static environment that could be implemented
    """

    def __init__(self, *args, arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]),
                 object_size=0.2, distract_size=0.15, distract_noise=0.1,
                 n_rbf=5, sdev=5., n_timesteps=150, render=False, **kwargs):

        self._arm_lengths = arm_lengths
        self._object_size = object_size
        self._distract_size = distract_size
        self._distract_noise = distract_noise
        self._n_rbf = n_rbf
        self._n_timesteps = n_timesteps
        self._sdev = sdev

        # We set the spaces
        self.observation_space = np.array([[-1, 1]] * 4)
        self.action_space = np.array([[-1, 1]] * arm_lengths.shape[0] * n_rbf)

        self._dynamic_environment = FixedEpisodeDynamizer(static_env=ArmBalls, n_iter=n_timesteps,
                                                          arm_lengths=arm_lengths, object_size=object_size,
                                                          distract_size=distract_size, distract_noise=distract_noise)
        self._controller = RbfController(n_action_dims=len(arm_lengths), n_rbf=n_rbf,
                                         n_timesteps=n_timesteps, sdev=sdev)
        #self.render_interval = render_interval
        if render:
            self._renderer = MatplotlibInteractiveRendering(ArmBallsRenderer, width=500, height=500,
                                                            rgb=False, object_size=object_size,
                                                            distract_size=distract_size)
            self._renderer.reset()

        self._observation = None
        self._hidden_state = None

    def reset(self):

        self._dynamic_environment.reset()
        self.n_iters = 0

        obs = self._dynamic_environment.observation_sequence

        self._observation = obs[-1, -4:]
        self._hidden_state = obs[-1, -2:]

    def act(self, action, render=True, **kwargs):

        parameterization = action.reshape(self._arm_lengths.shape[0], self._n_rbf)
        self._controller.act(parameterization)
        action_sequence = np.clip(self._controller.action_sequence, a_min=-1, a_max=1)
        self._dynamic_environment.act(action_sequence)
        self._observation = self._dynamic_environment.observation_sequence[-1, -4:]
        self._hidden_state = self._dynamic_environment.observation_sequence[-1, -2:]
        self.n_iters += 1
        
        if render:
            for i in range(self._n_timesteps):
                self._renderer.act(observation=self._dynamic_environment.observation_sequence[i], **kwargs)

    def terminate(self):

        pass

    @property
    def observation(self) -> np.ndarray:

        return self._observation

    @property
    def hidden_state(self) -> np.ndarray:

        return self._hidden_state

    @classmethod
    def test(cls):

        pass


class MyArmBallsObserved(AbstractActor, IStaticEnvironment):
    """
    ArmBalls with observations given as images
    """

    def __init__(self, arm_lengths=np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]),
                 object_size=0.2, distract_size=0.15, distract_noise=0.1,
                 n_rbf=5, sdev=5., n_timesteps=150, env_noise=0, render=False, render_interval=500,
                 rgb=False, render_arm=False, distract_first=False, **kwargs):

        self._arm_lengths = arm_lengths
        self._object_size = object_size
        self._distract_size = distract_size
        self._distract_noise = distract_noise
        self._n_rbf = n_rbf
        self._n_timesteps = n_timesteps
        self._sdev = sdev

        # We set the spaces
        self.action_space = np.array([[-1, 1]] * arm_lengths.shape[0] * n_rbf)

        self._dynamic_environment = FixedEpisodeDynamizer(static_env=ArmBalls, n_iter=n_timesteps, arm_lengths=arm_lengths,
                                                          object_size=object_size, distract_size=distract_size,
                                                          distract_noise=distract_noise, **kwargs)
        self._controller = RbfController(n_action_dims=len(arm_lengths), n_rbf=n_rbf,
                                         n_timesteps=n_timesteps, sdev=sdev)
        self.render_interval = render_interval
        if render:
            self._renderer = MatplotlibInteractiveRendering(ArmBallsRenderer, width=500, height=500, rgb=rgb,
                                                            arm_lengths=arm_lengths, object_size=object_size,
                                                            distract_size=distract_size, env_noise=env_noise,
                                                            distract_first=distract_first)
            self._renderer.reset()

        self._observer = ArmBallsRenderer(rgb=rgb, render_arm=render_arm, object_size=0.17, distract_size=0.15,
                                          env_noise=env_noise, distract_first=distract_first, **kwargs)
        self._observer.reset()

        self._observation = None
        self._explored_states = None
        self._hidden_state = None

    def reset(self):

        self._dynamic_environment.reset()
        self._observer.reset()
        self.n_iters = 0

        obs = self._dynamic_environment.observation_sequence
        self._observer.act(observation=obs[-1])

        self._observation = self._observer.rendering
        self._hidden_state = obs[-1, -2:]

    def act(self, action, render=True, **kwargs):

        parameterization = action.reshape(self._arm_lengths.shape[0], self._n_rbf)

        self._controller.act(parameterization)
        action_sequence = np.clip(self._controller.action_sequence, a_min=-1, a_max=1)
        self._dynamic_environment.act(action_sequence)
        env_state = self._dynamic_environment.observation_sequence[-1, :]
        self._observer.act(observation=env_state)
        self._observation = self._observer.rendering
        self._hidden_state = env_state[-2:]
        self.n_iters += 1

        if render and self.n_iters % self.render_interval == 0:
            for i in range(self._n_timesteps):
                self._renderer.act(observation=self._dynamic_environment.observation_sequence[i], **kwargs)

    def terminate(self):

        pass

    @property
    def observation(self) -> np.ndarray:

        return self._observation

    @property
    def hidden_state(self):

        return self._hidden_state

    @classmethod
    def test(cls):

        pass
