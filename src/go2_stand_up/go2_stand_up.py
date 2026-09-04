from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU
import mujoco
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

_orignial_imu_init = IMU.__init__

def patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model,mj_data)
    _orignial_imu_init(self, mj_model,mj_data,*args,**kwargs)
    self.step()

IMU.__init__ = patched_imu_init

FULL_STATE_OBS = (
    "base_pos",
    "base_ori_quat_wxyz",
    "base_lin_vel",
    "base_ang_vel",
    "qpos_js",
    "qvel_js"
)

PROPRIOCEPTIVE_OBS = (
    "gravity_vector:base",
    "imu_acc",
    "imu_gyro",
    "qpos_js",
    "qvel_js"
)

OBSERVATION_VARIANT = "full_state"

if OBSERVATION_VARIANT == "full_state":
    state_obs_name = FULL_STATE_OBS
elif OBSERVATION_VARIANT == "proprioceptive":
    state_obs_name = PROPRIOCEPTIVE_OBS
else:
    raise ValueError(
        f"Unknown observation variant: {OBSERVATION_VARIANT}"
    )

imu_kwargs = {
    "accel_name": "imu_acc",
    "gyro_name" : "imu_gyro",
    "imu_site_name" : "imu"
}

env = QuadrupedEnv(
    robot="go2",
    scene="flat",
    state_obs_names=state_obs_name,
    sensors=(IMU,),
    sensors_kwargs=(imu_kwargs,),
)

obs = env.reset()

episode_reward = 0.0

for t in range(1000):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    if terminated or truncated:
        break

print("Episode reward:",episode_reward)

env.close()