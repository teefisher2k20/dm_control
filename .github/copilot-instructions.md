# dm_control: MuJoCo-Based Reinforcement Learning Framework

## Architecture Overview

`dm_control` is a layered system for physics-based simulation and RL environments:

- **Core bindings** (`dm_control.mujoco`): Low-level Python bindings to MuJoCo physics engine, auto-generated from C headers via `dm_control/autowrap/`
- **Suite environments** (`dm_control.suite`): Pre-built RL tasks (cartpole, cheetah, humanoid) with MJCF XML models in same directory
- **Composer** (`dm_control.composer`): Modular environment construction using Entity/Arena/Task pattern
- **MJCF library** (`dm_control.mjcf`): Python object model for composing and modifying MuJoCo XML models
- **Locomotion** (`dm_control.locomotion`): Reusable components for locomotion tasks (walkers, arenas, tasks)

## Key Patterns

### Environment Construction

**Suite-style** (simple, monolithic):
```python
physics = Physics.from_xml_string(*get_model_and_assets())
task = Balance(swing_up=True, random=random)
env = control.Environment(physics, task, time_limit=10)
```

**Composer-style** (modular, composable):
```python
walker = cmu_humanoid.CMUHumanoidPositionControlled()
arena = corr_arenas.WallsCorridor(wall_gap=4., corridor_width=10)
task = corr_tasks.RunThroughCorridor(walker=walker, arena=arena)
env = composer.Environment(task=task, time_limit=30)
```

### MJCF Composition

PyMJCF enables composing models without name collisions:
```python
mjcf_model = mjcf.RootElement()  # Root <mujoco> element
body = mjcf_model.worldbody.add('body', name='my_body')
body.add('geom', type='box', size=[0.1, 0.1, 0.1])
site.attach(arm.mjcf_model)  # Attach sub-models, auto-prefixes names
```

Parse existing XML: `mjcf.from_path(filename)` or `mjcf.from_xml_string(xml_string)`

### Task Hierarchy

- **Suite tasks**: Inherit from `dm_control.suite.base.Task` (which extends `control.Task`)
- **Composer tasks**: Inherit from `dm_control.composer.task.Task` (abstract base with hooks)
- Tasks define `get_observation()`, `get_reward()`, `initialize_episode()`, action/observation specs

### Entity System (Composer)

Entities (`dm_control.composer.entity.Entity`) are reusable components with:
- `mjcf_model`: The PyMJCF model defining the entity's structure
- Lifecycle hooks: `initialize_episode_mjcf`, `after_compile`, `initialize_episode`, `before_step`, `after_step`
- `observables`: Dictionary of observable properties for RL agents

Arenas extend Entity and define the environment space. Walkers are detached entities that move within arenas.

## Build & Test

**Installation**: `pip install dm_control` (editable mode NOT supported due to auto-generated bindings)

**Testing**: Uses `unittest` framework. Run with:
```bash
python -m pytest dm_control/
python module_name_test.py  # Individual test files
```

Tests follow `*_test.py` naming convention.

**Build bindings**: The `autowrap.py` script generates low-level ctypes bindings from MuJoCo headers during setup. See `setup.py` `BuildMJBindingsCommand`.

## Rendering

Three OpenGL backends (auto-selected or via `MUJOCO_GL` env var):
- **GLFW**: Windowed rendering (required for `dm_control.viewer`)
- **EGL**: Headless hardware rendering (set `MUJOCO_EGL_DEVICE_ID` for GPU selection)
- **OSMesa**: Software rendering

Rendering methods:
- `physics.render()`: Returns numpy pixel array
- `viewer.launch(environment_loader=env_fn)`: Interactive GUI

## File Conventions

- MJCF XML models live alongside Python task definitions (e.g., `suite/cartpole.py` + `suite/cartpole.xml`)
- Common assets in `suite/common/` (skybox, materials, visual settings)
- Third-party models in `dm_control/third_party/` (Kinova arm, ANT quadruped)
- Assets referenced via `common.ASSETS` dict passed to `Physics.from_xml_string(xml, assets)`

## Physics Interaction

Access named elements via `physics.named.data.*` or `physics.named.model.*`:
```python
physics.named.data.qpos['joint_name']  # Joint positions
physics.named.data.geom_xpos[['geom1', 'geom2'], 'z']  # Geom z-positions
```

Modify state in `reset_context`: `with physics.reset_context(): physics.data.qpos[:] = ...`

## Common Gotchas

- **No editable installs**: `-e` flag breaks auto-generated bindings imports
- **Name collisions**: Use PyMJCF's `attach()` for automatic name prefixing when composing models
- **Timesteps**: Composer tasks require `physics_timestep` and `control_timestep` (must be integer multiples)
- **Tagged tasks**: Suite uses `SUITE = containers.TaggedTasks()` decorator pattern for task registry
- **Hook optimization**: Composer scans and memoizes non-trivial entity hooks to avoid overhead

## Creating Custom Environments

### Suite-Style (Simple, Single File)

1. **Define Physics extensions** (optional):
```python
class Physics(mujoco.Physics):
    def cart_position(self):
        return self.named.data.qpos['slider'][0]
```

2. **Create Task class** inheriting from `suite.base.Task`:
```python
class Balance(base.Task):
    def initialize_episode(self, physics):
        # Set initial state
        physics.named.data.qpos['slider'] = self.random.uniform(-.1, .1)
        super().initialize_episode(physics)
    
    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.bounded_position()
        obs['velocity'] = physics.velocity()
        return obs
    
    def get_reward(self, physics):
        upright = (physics.pole_angle_cosine() + 1) / 2
        return upright.mean()
```

3. **Register with tagged tasks**:
```python
SUITE = containers.TaggedTasks()

@SUITE.add('benchmarking')
def my_task(time_limit=10, random=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = MyTask(random=random)
    return control.Environment(physics, task, time_limit=time_limit)
```

### Composer-Style (Modular, Reusable)

1. **Create Entity components** (walkers, props, arenas):
```python
class MyArena(entity.Entity):
    def _build(self, name=None):
        self._mjcf_root = mjcf.from_path('arena.xml')
    
    @property
    def mjcf_model(self):
        return self._mjcf_root
```

2. **Define Task with lifecycle hooks**:
```python
class MyTask(composer.Task):
    def __init__(self, walker, arena, physics_timestep=0.005, control_timestep=0.025):
        self._arena = arena
        self._walker = walker
        self._arena.attach(self._walker)  # Compose entities
        
        # Enable observables
        for obs in self._walker.observables.proprioception:
            obs.enabled = True
        
        self.set_timesteps(physics_timestep, control_timestep)
    
    @property
    def root_entity(self):
        return self._arena
    
    def initialize_episode_mjcf(self, random_state):
        # Modify MJCF before compilation
        self._arena.regenerate(random_state)
    
    def initialize_episode(self, physics, random_state):
        # Set initial physics state
        self._walker.reinitialize_pose(physics, random_state)
    
    def before_step(self, physics, action, random_state):
        self._walker.apply_action(physics, action, random_state)
    
    def after_step(self, physics, random_state):
        # Check termination conditions
        pass
    
    def get_reward(self, physics):
        walker_xvel = physics.bind(self._walker.root_body).subtree_linvel[0]
        return rewards.tolerance(walker_xvel, (3.0, 3.0), margin=3.0)
    
    def should_terminate_episode(self, physics):
        return False
```

3. **Compose and instantiate**:
```python
walker = cmu_humanoid.CMUHumanoidPositionControlled()
arena = MyArena()
task = MyTask(walker=walker, arena=arena)
env = composer.Environment(task=task, time_limit=30)
```

## Extending Existing Environments

### Modify Suite Tasks

**Override specific methods**:
```python
class HarderBalance(cartpole.Balance):
    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Add wind disturbance
        physics.named.data.xfrc_applied['cart', 'force'] = self.random.randn(6)
    
    def get_reward(self, physics):
        # Make reward more sparse
        base_reward = super().get_reward(physics)
        return base_reward if base_reward > 0.9 else 0.0
```

### Extend Composer Entities

**Add observables to entities**:
```python
class MyWalker(cmu_humanoid.CMUHumanoid):
    @property
    def observables(self):
        observables = super().observables
        # Add custom observable
        observables.add_observable('my_sensor', 
                                   lambda physics: physics.data.qpos[:])
        return observables
```

### Modify MJCF Models Programmatically

**Add elements at runtime**:
```python
def initialize_episode_mjcf(self, random_state):
    # Add random obstacles
    for i in range(5):
        pos = random_state.uniform(-2, 2, size=3)
        self._arena.mjcf_model.worldbody.add(
            'geom', type='sphere', size=[0.1], pos=pos)
```

## Key Extension Points

- **Physics**: Custom named accessors for domain-specific quantities
- **Task.initialize_episode**: Set initial state with `self.random` for reproducibility
- **Task.get_observation**: Return `OrderedDict` with observation components
- **Task.get_reward**: Use `dm_control.utils.rewards` for smooth reward shaping
- **Task hooks**: `before_step`, `after_step`, `initialize_episode_mjcf`, `after_compile`
- **Entity.observables**: Use `@define.observable` decorator for entity-specific observations
- **Arena.regenerate**: Procedurally modify environment structure per episode

## Dependencies

Core: `mujoco`, `dm-env`, `numpy`, `lxml` (XML parsing), `absl-py`  
Rendering: `glfw`, `pyopengl`  
Locomotion: `labmaze` (maze generation)  
See `requirements.txt` for version-specific pins (numpy/scipy vary by Python version)
