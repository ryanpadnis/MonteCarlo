"""
Openai gym caller for 3 sepearte envs
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
# --- Continuous Action Space Environments ---
# Pendulum-v1:               Swing up & balance a pendulum | Box(1,) actions  | Box(3,) obs
# MountainCarContinuous-v0:  Drive a car up a hill         | Box(1,) actions  | Box(2,) obs
# LunarLanderContinuous-v2:  Land a rocket on a pad        | Box(2,) actions  | Box(8,) obs

ENV_NAMES = [
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "LunarLander-v3",  # render_mode="rgb_array" or "human"
]


if __name__ == "__main__":
    env_name = ENV_NAMES[0]  # Change index to switch env
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()

    for _ in range(20):
        action = env.action_space.sample()  # random agent — swap with model.predict() later
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)  # ~50fps
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


