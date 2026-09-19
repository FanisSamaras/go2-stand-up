from envs import create_env,FULL_STATE_OBS
import numpy as np
import torch
from numpy.typing import NDArray
from enum import Enum
from internal_control.PID_alone import PIDController

device = "cuda" if torch.cuda.is_available() else "cpu"

def flatten_obs(obs,keys = FULL_STATE_OBS):
    return np.concatenate([np.atleast_1d(obs[key]) for key in keys])

def early_exit(terminated,truncated,steps,max_steps=1000):
    return bool(terminated or truncated or (steps >= max_steps))

class LegAssociation(Enum):
    FL = 0
    FR = 1
    RL = 2
    RR = 3

env = create_env()

pid_controller = PIDController()

obs = flatten_obs(env.reset())
print(obs.shape)

def test():
    obs = flatten_obs(env.reset())
    for _ in range(1000):
        q = env.mjData.qpos[7:19]
        dq = env.mjData.qvel[6:18]
        action = pid_controller.get_action(q=q,dq=dq)
        next_obs, reward, terminated, truncated, info = env.step(action=action)
        done = truncated or terminated
        obs = next_obs
        obs = flatten_obs(obs=obs)
        env.render()


if __name__ == "__main__":
    test()