from envs import create_env
from internal_control.PID import PIDController
import numpy as np
import torch
from numpy.typing import NDArray
from enum import Enum


device = "cuda" if torch.cuda.is_available() else "cpu"

class LegAssociation(Enum):
    FL = 0
    FR = 1
    RL = 2
    RR = 3

env = create_env()

obs = env.reset()

episode_reward = 0.0

for t in range(1000):

    # action = env.action_space.sample()

    pid_controller = PIDController()

    q = env.mjData.qpos[7:19]
    dq = env.mjData.qvel[6:18]

    action = pid_controller.get_action(q=q,dq=dq)

    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    env.render()

    if terminated or truncated:
        break

print("Episode reward:",episode_reward)

env.close()