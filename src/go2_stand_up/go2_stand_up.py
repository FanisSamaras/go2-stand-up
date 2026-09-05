from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU
import mujoco
import numpy as np
import torch
from numpy.typing import NDArray

device = "cuda" if torch.cuda.is_available() else "cpu"

_orignial_imu_init = IMU.__init__

def patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model,mj_data)
    _orignial_imu_init(self, mj_model,mj_data,*args,**kwargs)
    self.step()

IMU.__init__ = patched_imu_init

def _compute_reward(self):
    return 1

QuadrupedEnv._compute_reward = _compute_reward

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
    legs_order=("FL","FR","RL","RR")
)

obs = env.reset()

episode_reward = 0.0

class PIDController:
    '''
    Simple PID controller based around a single q_nominal position trying to stabilize around this position
    \n Leg order FL, FR, RL, RR
    '''
    def __init__(self) -> None:
        self.q_nominal = np.array([0.0, 0.9, -1.8, #FL                          
                                   0.0, 0.9, -1.8, #FR
                                   0.0, 0.9, -1.8, #RL
                                   0.0, 0.9, -1.8] #RR
                                   ,dtype=np.float32)
        self.kp = np.array([20,35,45] * 4, dtype=np.float32)
        self.kd = np.sqrt(self.kp)

    def get_action(self,q,dq) -> NDArray:
        action = self.kp * (self.q_nominal - q) - self.kd * dq
        return action
        

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