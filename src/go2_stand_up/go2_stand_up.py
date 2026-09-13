from envs import create_env,FULL_STATE_OBS
from internal_control.PID import PIDController
from policies.PPO import PPO,RolloutBuffer
import numpy as np
import torch
from numpy.typing import NDArray
from enum import Enum


device = "cuda" if torch.cuda.is_available() else "cpu"

def flatten_obs(obs,keys = FULL_STATE_OBS):
    return np.concatenate([np.atleast_1d(obs[key]) for key in keys])

class LegAssociation(Enum):
    FL = 0
    FR = 1
    RL = 2
    RR = 3

env = create_env()

obs = env.reset()

episode_reward = 0.0

OBS_DIM = 37
ACTION_DIM = 12
MAX_STEPS = 2_000_000
EPISODE_LENGTH = 1000
SAVE_INTERVAL = 50_000

action_scale = 0

agent = PPO(
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    device=device,
    lr=1e-4,
    gamma=0.98,
    gae_lambda=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.005,
    max_grad_norm=0.5,
    ppo_epochs=5,
    minibatch_size=256)

buffer = RolloutBuffer(
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    rollout_steps=4096,
    device=device
)

obs = flatten_obs(env.reset())
print(obs.shape)

class Episode:
    def __init__(self,returns, length_counter, number):
        self.returns = returns
        self.length_counter = length_counter
        self.number = number

episode = Episode(0,0,0)

latest_losses = {}


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