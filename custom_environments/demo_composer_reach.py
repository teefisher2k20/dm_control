"""Example script to run the Composer-style reach target environment."""

import sys
import os

# Add custom_environments to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dm_control import composer
from reach_target import SimpleArena, SimpleWalker, ReachTarget


def run_environment(env, num_episodes=3, steps_per_episode=200):
    """Run the environment for several episodes."""
    print(f"\nRunning environment for {num_episodes} episodes...")
    print(f"Action spec: {env.action_spec()}")
    print(f"Observation spec keys: {list(env.observation_spec().keys())}")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        time_step = env.reset()
        total_reward = 0.0
        
        for step in range(steps_per_episode):
            # Random action
            action = np.random.uniform(
                env.action_spec().minimum,
                env.action_spec().maximum,
                size=env.action_spec().shape
            )
            
            time_step = env.step(action)
            total_reward += time_step.reward
            
            if step % 40 == 0:
                # Get walker position if available
                obs = time_step.observation
                walker_pos = None
                for key in obs.keys():
                    if 'position' in key:
                        walker_pos = obs[key][:2] if len(obs[key]) >= 2 else obs[key]
                        break
                
                if walker_pos is not None:
                    print(f"  Step {step}: reward={time_step.reward:.3f}, "
                          f"position=({walker_pos[0]:.2f}, {walker_pos[1]:.2f})")
                else:
                    print(f"  Step {step}: reward={time_step.reward:.3f}")
            
            if time_step.last():
                print(f"  Episode terminated at step {step}")
                break
        
        print(f"Episode {episode + 1} total reward: {total_reward:.3f}")


def main():
    print("=" * 60)
    print("Composer-Style Reach Target Environment Demo")
    print("=" * 60)
    
    # Create components
    print("\nCreating environment components...")
    walker = SimpleWalker()
    arena = SimpleArena(size=8, wall_height=1.0)
    task = ReachTarget(
        walker=walker,
        arena=arena,
        target_distance=5.0,
        physics_timestep=0.005,
        control_timestep=0.025
    )
    
    # Create environment
    print("Building composer environment...")
    env = composer.Environment(
        task=task,
        time_limit=20,
        random_state=42,
        strip_singleton_obs_buffer_dim=True
    )
    
    print("Environment created successfully!")
    print(f"Physics timestep: {task.physics_timestep}")
    print(f"Control timestep: {task.control_timestep}")
    
    # Run episodes
    run_environment(env, num_episodes=3, steps_per_episode=200)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNote: The walker is a simple point mass that tries to reach")
    print("a randomly positioned target in the arena using 2D forces.")


if __name__ == '__main__':
    main()
