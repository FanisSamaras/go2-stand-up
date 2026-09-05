from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU
import mujoco
import numpy as np
import torch
from numpy.typing import NDArray
from enum import Enum


device = "cuda" if torch.cuda.is_available() else "cpu"

_orignial_imu_init = IMU.__init__

def patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model,mj_data) #pyright: ignore[reportAttributeAccessIssue]
    _orignial_imu_init(self, mj_model,mj_data,*args,**kwargs)
    self.step()

class LegAssociation(Enum):
    FL = 0
    FR = 1
    RL = 2
    RR = 3


class NewEnv(QuadrupedEnv):
    def __init__(self, robot, state_obs_names = ..., scene = 'flat', sim_dt = 0.002, base_vel_command_type = 'forward', ref_base_lin_vel = 0.5, ref_base_ang_vel = 0, ground_friction_coeff = 1, legs_order = ('FL', 'FR', 'RL', 'RR'), sensors = None, sensors_kwargs = None, external_disturbances_kwargs = None,leg_pair = (LegAssociation.FL.value,LegAssociation.RR.value)):
        super().__init__(robot, state_obs_names, scene, sim_dt, base_vel_command_type, ref_base_lin_vel, ref_base_ang_vel, ground_friction_coeff, legs_order, sensors, sensors_kwargs, external_disturbances_kwargs)
        self.leg_pair = legs_order[leg_pair[0]],legs_order[leg_pair[1]]

    def _compute_reward(self):
        counter = 0
        linear_velocity_error = env.base_lin_vel_err(frame="base")
        angular_velocity_error = env.base_ang_vel_err(frame="base")
        feet_contact_dictionary = env.feet_contact_state(frame="base",ground_reaction_forces=False)[0]
        for value in feet_contact_dictionary:
            print(value)
            if value:
                counter += 1
        feet_contact_penalty = 1 if counter != 2 else 0
        return 1

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

env = NewEnv(
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
        self.q_nominal = np.array([0.0, 0.9, -1.8,   #1.FL                
                                   0.0, 0.9, -1.8,   #2.FR
                                   0.0, 0.9, -1.8,   #3.RL
                                   0.0, 0.9, -1.8]   #4.RR
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