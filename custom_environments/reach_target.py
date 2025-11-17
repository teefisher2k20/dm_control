"""Custom Composer-style environment with modular components."""

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import entity as entity_module
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base as walker_base
from dm_control.utils import rewards
import numpy as np


class SimpleArena(entity_module.Entity):
    """A simple flat arena with walls."""

    def _build(self, size=10, wall_height=1.0, name='simple_arena'):
        """Builds the arena.

        Args:
            size: Size of the arena floor (square).
            wall_height: Height of the walls.
            name: Name of the arena.
        """
        self._mjcf_root = mjcf.RootElement(model=name)
        
        # Add lighting
        self._mjcf_root.worldbody.add('light', pos=(0, 0, 4), dir=(0, 0, -1))
        
        # Add floor
        self._floor = self._mjcf_root.worldbody.add(
            'geom',
            type='plane',
            size=(size, size, 0.1),
            rgba=(0.9, 0.9, 0.9, 1.0),
            name='floor'
        )
        
        # Add walls
        wall_thickness = 0.1
        for i, (x, y, size_x, size_y) in enumerate([
            (size, 0, wall_thickness, size),  # Right wall
            (-size, 0, wall_thickness, size),  # Left wall
            (0, size, size, wall_thickness),  # Front wall
            (0, -size, size, wall_thickness),  # Back wall
        ]):
            self._mjcf_root.worldbody.add(
                'geom',
                type='box',
                pos=(x, y, wall_height / 2),
                size=(size_x, size_y, wall_height / 2),
                rgba=(0.7, 0.7, 0.7, 1.0),
                name=f'wall_{i}'
            )
        
        # Spawn site for entities
        self._spawn_site = self._mjcf_root.worldbody.add(
            'site', name='spawn', size=(0.01, 0.01, 0.01), pos=(0, 0, 0)
        )

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def spawn_site(self):
        return self._spawn_site

    def regenerate(self, random_state):
        """Regenerates the arena (for procedural generation)."""
        # In this simple arena, we don't change anything
        pass


class SimpleWalker(walker_base.Walker):
    """A simple point mass walker that can move in 2D."""

    def _build(self, name='simple_walker'):
        """Builds the walker."""
        self._mjcf_root = mjcf.RootElement(model=name)
        
        # Create the main body
        self._body = self._mjcf_root.worldbody.add(
            'body', name='walker_body', pos=(0, 0, 0.5)
        )
        
        # Add slide joints for X and Y movement
        self._joint_x = self._body.add(
            'joint', name='slide_x', type='slide', axis=(1, 0, 0), 
            damping=0.5, limited=False
        )
        self._joint_y = self._body.add(
            'joint', name='slide_y', type='slide', axis=(0, 1, 0),
            damping=0.5, limited=False
        )
        
        # Add a sphere geom for the body
        self._body.add(
            'geom',
            type='sphere',
            size=(0.2,),
            rgba=(0.3, 0.3, 0.8, 1.0),
            name='body_geom',
            mass=1.0
        )
        
        # Add actuators for X and Y movement
        self._actuator_x = self._mjcf_root.actuator.add(
            'motor',
            joint=self._joint_x,
            gear=(5,),
            name='move_x',
            ctrllimited=True,
            ctrlrange=(-10, 10)
        )
        self._actuator_y = self._mjcf_root.actuator.add(
            'motor',
            joint=self._joint_y,
            gear=(5,),
            name='move_y',
            ctrllimited=True,
            ctrlrange=(-10, 10)
        )

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def root_body(self):
        return self._body

    @property
    def actuators(self):
        """Returns all actuators."""
        return [self._actuator_x, self._actuator_y]

    @property
    def observable_joints(self):
        """Returns joints that should be observable."""
        return [self._joint_x, self._joint_y]

    def _build_observables(self):
        return SimpleWalkerObservables(self)


class SimpleWalkerObservables(walker_base.WalkerObservables):
    """Observables for the simple walker."""

    @composer.observable
    def position(self):
        """Returns the position of the walker."""
        return observable.MJCFFeature('xpos', self._entity.root_body)

    @composer.observable
    def velocity(self):
        """Returns the velocity of the walker."""
        return observable.MJCFFeature('cvel', self._entity.root_body)


class ReachTarget(composer.Task):
    """A task where the walker must reach a target location."""

    def __init__(self, walker, arena, target_distance=5.0,
                 physics_timestep=0.005, control_timestep=0.025):
        """Initializes the task.

        Args:
            walker: A `SimpleWalker` instance.
            arena: A `SimpleArena` instance.
            target_distance: Distance of the target from the origin.
            physics_timestep: Physics simulation timestep.
            control_timestep: Agent control timestep.
        """
        self._arena = arena
        self._walker = walker
        self._target_distance = target_distance
        
        # Attach walker to arena
        self._arena.spawn_site.attach(self._walker.mjcf_model)
        
        # Add target sphere to arena
        self._target_site = self._arena.mjcf_model.worldbody.add(
            'site',
            name='target',
            type='sphere',
            size=(0.3,),
            rgba=(0.8, 0.3, 0.3, 0.5),
            pos=(target_distance, 0, 0.3)
        )
        
        # Enable observables
        self._walker.observables.position.enabled = True
        self._walker.observables.velocity.enabled = True
        
        # Set timesteps (control_timestep must be >= physics_timestep)
        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)
        
        self._target_pos = np.array([target_distance, 0, 0])

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """Modifies MJCF before compilation."""
        self._arena.regenerate(random_state)
        
        # Randomize target position
        angle = random_state.uniform(0, 2 * np.pi)
        distance = random_state.uniform(3.0, 7.0)
        self._target_pos = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0
        ])
        self._target_site.pos = np.array([self._target_pos[0], self._target_pos[1], 0.3])

    def initialize_episode(self, physics, random_state):
        """Sets initial physics state."""
        # Reset walker to origin with small random offset
        walker_body = physics.bind(self._walker.root_body)
        offset = random_state.uniform(-0.5, 0.5, size=2)
        walker_body.xpos = np.array([offset[0], offset[1], 0.5])
        walker_body.xquat = np.array([1, 0, 0, 0])

    def before_step(self, physics, action, random_state):
        """Applies action before physics step."""
        # Action directly maps to the actuators
        physics.set_control(action)

    def get_reward(self, physics):
        """Returns reward based on distance to target."""
        walker_pos = physics.bind(self._walker.root_body).xpos[:2]  # Only X, Y
        target_pos_2d = self._target_pos[:2]  # Only X, Y
        distance = np.linalg.norm(walker_pos - target_pos_2d)
        
        # Reward for being close to target
        reward = rewards.tolerance(
            distance,
            bounds=(0, 0.5),
            margin=self._target_distance,
            sigmoid='linear',
            value_at_margin=0.0
        )
        return reward

    def should_terminate_episode(self, physics):
        """Check if episode should terminate."""
        # Terminate if walker reaches target
        walker_pos = physics.bind(self._walker.root_body).xpos[:2]  # Only X, Y
        target_pos_2d = self._target_pos[:2]  # Only X, Y
        distance = np.linalg.norm(walker_pos - target_pos_2d)
        return distance < 0.5
