from envs import create_env,FULL_STATE_OBS
from internal_control.PID import PIDController
from policies.PPO import PPO,RolloutBuffer
import numpy as np
import torch
from numpy.typing import NDArray
from enum import Enum
from internal_control.PID import PIDController

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

obs = env.reset()

episode_reward = 0.0

OBS_DIM = 37
ACTION_DIM = 12
MAX_STEPS = 2_000_000
EPISODE_LENGTH = 1000
SAVE_INTERVAL = 50_000

action_scale = 0.5

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

pid_controller = PIDController()

obs = flatten_obs(env.reset())
print(obs.shape)

class Episode:
    def __init__(self,returns, length_counter, number):
        self.returns = returns
        self.length_counter = length_counter
        self.number = number

def train():
    obs = flatten_obs(env.reset())

    episode = Episode(0.0,0,0)

    latest_losses = {}

    time = 0
    for step in range(1,MAX_STEPS + 1):
        rl_action, log_prob, value = agent.select_action(obs)

        q_desired = action_scale * rl_action + pid_controller.q_nominal # nominal MAYBE

        q = env.mjData.qpos[7:19]
        dq = env.mjData.qvel[6:18]
    
        # Compute real motor torque commands
        action = pid_controller.kp * (q_desired - q) - pid_controller.kd * dq

        # Advance clock
        time += env.simulation_dt

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = early_exit(terminated,truncated,episode.length_counter,2000)

        buffer.add(
            obs=obs,
            action=rl_action,
            log_prob=log_prob,
            reward=reward,
            done=float(done),
            value=value
        )

        obs = next_obs
        obs = flatten_obs(obs)
        episode.returns += reward
        episode.length_counter += 1

        if done:
            print(
                f"Episode {episode.number:05d} |"
                f"Step {step:08d} |"
                f"Return {episode.returns:8.2f} |"
                f"Length {episode.length_counter} |"
                f"KL {latest_losses.get('approx_kl', 0.0):.4f}"
            )

            obs = flatten_obs(env.reset())

            episode.returns = 0.0
            episode.length_counter = 0
            episode.number +=1
    
        if buffer.is_full():

            last_value = agent.value(obs)
            latest_losses = agent.update(buffer, last_value)
            buffer.clear()   
            print(
                f"[PPO Update] Step {step:08d} | "
                f"policy_loss {latest_losses['policy_loss']:.4f} | "
                f"value_loss {latest_losses['value_loss']:.4f} | "
                f"entropy {latest_losses['entropy']:.4f} | "
                f"KL {latest_losses['approx_kl']:.4f}"
            )

        if step % SAVE_INTERVAL == 0:
            checkpoint_path = f"./src/policies/checkpoint/ppo_go2_step_{step}.pt"
            agent.save(checkpoint_path)
            print(f"[Checkpoint] Saved for steps {step}")

def load_test():
    time = 0
    agent.load("./src/policies/checkpoint/ppo_go2_step_200000.pt")
    obs = flatten_obs(env.reset())
    for _ in range(1,2000*5+1):
        rl_action, log_prob, value = agent.select_action(obs)
        q_desired = action_scale * rl_action + pid_controller.q_nominal # nominal MAYBE
        q = env.mjData.qpos[7:19]
        dq = env.mjData.qvel[6:18]
        # Compute real motor torque commands
        action = pid_controller.kp * (q_desired - q) - pid_controller.kd * dq
        # Advance clock
        time += env.simulation_dt

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = truncated or terminated
        obs = next_obs
        obs = flatten_obs(obs)
        env.render()
        if done:
            break

if __name__ == "__main__":
    load_test()
    env.close()