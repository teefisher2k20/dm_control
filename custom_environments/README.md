# Custom dm_control Environments

This directory contains custom environments built using the dm_control framework, demonstrating both **Suite-style** and **Composer-style** patterns from the how-to guide.

## Environments

### 1. Simple Pendulum (Suite-Style)

A simple pendulum swingup task following the monolithic Suite pattern.

**Files:**
- `simple_pendulum.xml` - MJCF model defining the physics
- `simple_pendulum.py` - Task implementation with Physics extensions
- `demo_suite_pendulum.py` - Demo script

**Tasks:**
- `balance()` - Start near upright, maintain balance
- `swingup()` - Start hanging down, swing up to upright position
- `swingup_sparse()` - Sparse reward variant

**Features:**
- Custom Physics class with domain-specific accessors
- Multiple task variants from same base class
- Suite-style task registration with `@SUITE.add()`
- Smooth and sparse reward shaping

**Run:**
```bash
python demo_suite_pendulum.py
```

**Visualize:**
```bash
python demo_with_viewer.py  # Select option 1
```

### 2. Reach Target (Composer-Style)

A modular environment where a 2D point mass must reach a randomly positioned target.

**Files:**
- `reach_target.py` - Modular components (Arena, Walker, Task)
- `demo_composer_reach.py` - Demo script

**Components:**
- `SimpleArena` - Flat arena with walls
- `SimpleWalker` - 2D point mass with slide joints
- `ReachTarget` - Task with target reaching objective
- `SimpleWalkerObservables` - Custom observable properties

**Features:**
- Entity/Arena/Task separation
- Procedural target generation via `initialize_episode_mjcf()`
- Walker observables (position, velocity)
- Modular, reusable components
- Proper timestep management

**Run:**
```bash
python demo_composer_reach.py
```

**Visualize:**
```bash
python demo_with_viewer.py  # Select option 2
```

## Key Patterns Demonstrated

### Suite-Style
✅ Custom Physics class with named accessors  
✅ Task inheritance from `base.Task`  
✅ `initialize_episode()` for episode initialization  
✅ `get_observation()` returning OrderedDict  
✅ `get_reward()` with smooth and sparse variants  
✅ Task registration with `@SUITE.add()`  

### Composer-Style
✅ Modular Entity components  
✅ Arena/Walker/Task separation  
✅ Walker extending `walker_base.Walker`  
✅ `initialize_episode_mjcf()` for MJCF modification  
✅ `before_step()` for action application  
✅ Observable system for RL agents  
✅ Proper timestep configuration  
✅ `should_terminate_episode()` for early termination  

## Installation

```bash
# Install dm_control
pip install dm_control

# Run demos from this directory
python demo_suite_pendulum.py
python demo_composer_reach.py

# Visualize with interactive viewer (requires GLFW)
python demo_with_viewer.py
```

## Architecture Insights

**Suite-style** is best for:
- Simple, self-contained tasks
- Rapid prototyping
- Tasks with single XML model
- When modularity isn't needed

**Composer-style** is best for:
- Complex, multi-entity environments
- Reusable components
- Procedural generation
- Multi-agent scenarios
- When entities need to be composed dynamically

## Common Gotchas Fixed

1. **Timesteps**: Control timestep must be ≥ physics timestep and an integer multiple
2. **MJCF modifications**: Must happen in `initialize_episode_mjcf()`, not `initialize_episode()`
3. **Walker abstract methods**: Must implement `actuators` and `observable_joints` properties
4. **Joint actuation**: Freejoint requires special handling; slide/hinge joints are simpler
5. **Array shapes**: Be careful with numpy array concatenation and broadcasting

## Extending These Examples

### Adding Observables to Walker:
```python
@composer.observable
def custom_sensor(self):
    return observable.MJCFFeature('sensordata', self._entity.mjcf_model.sensor['my_sensor'])
```

### Modifying Rewards:
```python
def get_reward(self, physics):
    base_reward = super().get_reward(physics)
    bonus = rewards.tolerance(...)
    return base_reward + bonus
```

### Procedural Arena Generation:
```python
def regenerate(self, random_state):
    # Clear old obstacles
    # Add new random obstacles to self._mjcf_root.worldbody
    pass
```

## References

- Main documentation: [dm_control README](../README.md)
- How-to guide: [.github/copilot-instructions.md](../.github/copilot-instructions.md)
- Suite examples: `dm_control/suite/`
- Composer examples: `dm_control/locomotion/examples/`
