"""Example script to run the Suite-style simple pendulum environment."""

import sys
import os

# Add custom_environments to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from simple_pendulum import balance, swingup, swingup_sparse


def run_environment(env, num_episodes=3, steps_per_episode=100):
    """Run the environment for several episodes."""
    print(f"\nRunning environment for {num_episodes} episodes...")
    print(f"Action spec: {env.action_spec()}")
    print(f"Observation spec: {env.observation_spec()}")
    
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
            
            if step % 20 == 0:
                print(f"  Step {step}: reward={time_step.reward:.3f}, "
                      f"angle_cos={time_step.observation['angle_cos'][0]:.3f}")
            
            if time_step.last():
                break
        
        print(f"Episode {episode + 1} total reward: {total_reward:.3f}")


def main():
    print("=" * 60)
    print("Suite-Style Simple Pendulum Environment Demo")
    print("=" * 60)
    
    # Test balance task (start near upright)
    print("\n\n1. BALANCE TASK (start near upright)")
    print("-" * 60)
    env = balance(time_limit=10, random=42)
    run_environment(env, num_episodes=2, steps_per_episode=50)
    
    # Test swingup task (start hanging down)
    print("\n\n2. SWINGUP TASK (start hanging down)")
    print("-" * 60)
    env = swingup(time_limit=15, random=42)
    run_environment(env, num_episodes=2, steps_per_episode=100)
    
    # Test sparse reward variant
    print("\n\n3. SWINGUP SPARSE REWARD")
    print("-" * 60)
    env = swingup_sparse(time_limit=15, random=42)
    run_environment(env, num_episodes=2, steps_per_episode=100)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
