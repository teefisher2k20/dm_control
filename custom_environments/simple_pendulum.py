"""Simple pendulum swingup task following Suite-style patterns."""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np


_DEFAULT_TIME_LIMIT = 10
_SWINGUP_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    model_path = os.path.join(os.path.dirname(__file__), 'simple_pendulum.xml')
    with open(model_path, 'r') as f:
        xml_string = f.read()
    return xml_string, {}


@SUITE.add('benchmarking')
def balance(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Pendulum Balance task (start near upright)."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Balance(swing_up=False, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def swingup(time_limit=_SWINGUP_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Pendulum Swing-Up task (start hanging down)."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Balance(swing_up=True, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def swingup_sparse(time_limit=_SWINGUP_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the sparse reward variant of the Pendulum Swing-Up task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Balance(swing_up=True, sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Pendulum domain."""

    def pole_angle(self):
        """Returns the angle of the pole in radians."""
        return self.named.data.qpos['hinge'][0]

    def pole_angle_cosine(self):
        """Returns the cosine of the pole angle."""
        return np.cos(self.pole_angle())

    def pole_angular_velocity(self):
        """Returns the angular velocity of the pole."""
        return self.named.data.qvel['hinge'][0]

    def tip_position(self):
        """Returns the height of the pole tip."""
        return self.named.data.geom_xpos['tip', 'z']


class Balance(base.Task):
    """A Pendulum task to balance the pole upright or swing it up."""

    _ANGLE_COSINE_RANGE = (0.995, 1.0)  # Near upright

    def __init__(self, swing_up, sparse, random=None):
        """Initializes an instance of `Balance`.

        Args:
            swing_up: A `bool`, which if `True` starts the pole pointing
                downward. Otherwise, starts near upright.
            sparse: A `bool`, whether to return a sparse or smooth reward.
            random: Optional, either a `numpy.random.RandomState` instance, an
                integer seed for creating a new `RandomState`, or None to select
                a seed automatically (default).
        """
        self._sparse = sparse
        self._swing_up = swing_up
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
            physics: An instance of `Physics`.
        """
        if self._swing_up:
            # Start hanging down with small perturbation
            physics.named.data.qpos['hinge'] = np.pi + 0.01 * self.random.randn()
        else:
            # Start near upright
            physics.named.data.qpos['hinge'] = self.random.uniform(-0.1, 0.1)
        
        # Add small initial velocity
        physics.named.data.qvel['hinge'] = 0.01 * self.random.randn()
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the physics state."""
        obs = collections.OrderedDict()
        obs['angle_cos'] = np.array([physics.pole_angle_cosine()])
        obs['angle_sin'] = np.array([np.sin(physics.pole_angle())])
        obs['velocity'] = np.array([physics.pole_angular_velocity()])
        obs['tip_height'] = np.array([physics.tip_position()])
        return obs

    def _get_reward(self, physics, sparse):
        """Returns the reward for the current state."""
        if sparse:
            # Binary reward: 1 if upright, 0 otherwise
            angle_cos = physics.pole_angle_cosine()
            upright = 1.0 if angle_cos > self._ANGLE_COSINE_RANGE[0] else 0.0
            small_velocity = 1.0 if abs(physics.pole_angular_velocity()) < 0.5 else 0.0
            return upright * small_velocity
        else:
            # Smooth reward based on angle and velocity
            upright = (physics.pole_angle_cosine() + 1.0) / 2.0  # 0 when down, 1 when up
            centered = rewards.tolerance(
                physics.pole_angle_cosine(), 
                bounds=(0.99, 1.0), 
                margin=1.0,
                sigmoid='linear',
                value_at_margin=0.0
            )
            small_velocity = rewards.tolerance(
                physics.pole_angular_velocity(),
                margin=5.0,
                sigmoid='quadratic',
                value_at_margin=0.0
            )
            small_control = rewards.tolerance(
                physics.control(),
                margin=5.0,
                value_at_margin=0.0,
                sigmoid='quadratic'
            )[0]
            
            return upright * (1.0 + centered + small_velocity + small_control) / 4.0

    def get_reward(self, physics):
        """Returns a sparse or smooth reward, as specified in the constructor."""
        return self._get_reward(physics, sparse=self._sparse)
