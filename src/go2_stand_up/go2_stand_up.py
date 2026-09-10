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
    def __init__(self, robot, state_obs_names = ..., scene = 'flat', sim_dt = 0.002, base_vel_command_type = 'forward', ref_base_lin_vel = 0.5, ref_base_ang_vel = 0, ground_friction_coeff = 1, legs_order = ('FL', 'FR', 'RL', 'RR'), sensors = None, sensors_kwargs = None, external_disturbances_kwargs = None,
                 leg_pair = (LegAssociation.FL.value,LegAssociation.RR.value)):
        super().__init__(robot, state_obs_names, scene, sim_dt, base_vel_command_type, ref_base_lin_vel, ref_base_ang_vel, ground_friction_coeff, legs_order, sensors, sensors_kwargs, external_disturbances_kwargs)
        self.leg_pair = leg_pair

    def distance_from_line_2D(self,p1:NDArray,p2:NDArray,p3:NDArray)->float:
        """
        Calculates the distance of the 2D point p3 from the 2D line defined by the points p1, p2
        """
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]

        numerator = abs((y2-y1)*p3[0] - (x2-x1)*p3[1] + x2*y1 - y2*x1)
        denumerator = ((y2-y1)**2 + (x2-x1)**2)**0.5

        return numerator/denumerator

    def _compute_reward(self):
        # stationary 
        lin_vel_err_B = self.base_lin_vel_err(frame="base")
        ang_vel_err_B = self.base_ang_vel_err(frame="base")
        sigma_lin_vel = 0.25
        sigma_ang_vel = 0.25
        tracking_lin_vel = np.exp(-np.sum(lin_vel_err_B[:2] ** 2) / (2 * sigma_lin_vel **2))
        tracking_yaw_rate = np.exp(-(ang_vel_err_B[2] ** 2) / (2 * sigma_ang_vel **2))

        # left-right rotation 
        base_lin_vel_B = self.base_lin_vel(frame="base")
        z_vel_penalty = base_lin_vel_B[2] ** 2

        # "kebab" rotation 
        base_ang_vel_B = self.base_ang_vel(frame="base")
        roll_pitch_ang_vel_penalty = np.sum(base_ang_vel_B[:2] ** 2)

        # high torque penalty
        tau = self.torque_ctrl_setpoint
        torque_penalty = np.sum(tau **2)

        # correct legs contact
        leg_contacts, contact_positions = env.feet_contact_state(frame="base",ground_reaction_forces=False)
        leg_contacts = leg_contacts.to_list()
        contact_count = sum(leg_contacts)
        feet_contact_penalty = 0
        com_offset = 1
        # print((contact_count !=2) ,(not leg_contacts[self.leg_pair[0]]),(not leg_contacts[self.leg_pair[1]]))
        if (contact_count!= 2) or (not leg_contacts[self.leg_pair[0]]) or (not leg_contacts[self.leg_pair[1]]):
            feet_contact_penalty = 1
        else:
            com_xy = env.com[0:2]
            contact_point2D_1 = contact_positions["FL"] # fix
            contact_point2D_2 = contact_positions["RR"]
            # try:
            contact_point2D_1 = contact_point2D_1[0].pos[0:2]
            # print(contact_point2D_1)
            contact_point2D_2 = contact_point2D_2[0].pos[0:2]
            com_offset = self.distance_from_line_2D(contact_point2D_1,contact_point2D_2,com_xy)
            # except:
            # com_offset = 1
            # print("not in contact")

        return (
            1 + # alive bonus MAYBE
            1 * tracking_lin_vel +
            1 * tracking_yaw_rate +
            -1 * feet_contact_penalty +
            -1 * com_offset +
            -1 * z_vel_penalty + # MAYBE
            -1 * roll_pitch_ang_vel_penalty + # MAYBE
            -1 * torque_penalty # MAYBE
            )
    

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