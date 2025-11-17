"""
Demo script showing how to visualize custom environments with dm_control viewer.
"""

from dm_control import viewer
import numpy as np

# Import both custom environments
from simple_pendulum import balance, swingup
from reach_target import create_reach_target_env


def pendulum_policy(time_step):
    """Simple policy for pendulum - random actions."""
    return np.random.uniform(-1, 1, size=(1,))


def reach_policy(time_step):
    """Simple policy for reach target - move toward target."""
    # Random exploration
    return np.random.uniform(-1, 1, size=(2,))


if __name__ == '__main__':
    print("Choose an environment to visualize:")
    print("1. Simple Pendulum (swingup)")
    print("2. Reach Target (composer)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        print("\nLaunching Simple Pendulum with viewer...")
        print("Controls: Double-click to pause/unpause, mouse to rotate view")
        env = swingup(time_limit=10, random=np.random.RandomState(42))
        viewer.launch(env, policy=pendulum_policy)
    
    elif choice == '2':
        print("\nLaunching Reach Target with viewer...")
        print("Controls: Double-click to pause/unpause, mouse to rotate view")
        env = create_reach_target_env(time_limit=30)
        viewer.launch(env, policy=reach_policy)
    
    else:
        print("Invalid choice. Please run again and enter 1 or 2.")
