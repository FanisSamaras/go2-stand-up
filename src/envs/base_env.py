from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU
import mujoco
import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from enum import Enum
from internal_control.PID import PIDController

# episode_logs = []

device = "cuda" if torch.cuda.is_available() else "cpu"

_orignial_imu_init = IMU.__init__

def patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model,mj_data) 
    _orignial_imu_init(self, mj_model,mj_data,*args,**kwargs)
    self.step()

class PatchedIMU(IMU):
    def __init__(self, mj_model, mj_data, *args, **kwargs):
        patched_imu_init(self, mj_model, mj_data, *args, **kwargs)
# IMU.__init__ = patched_imu_init

class LegAssociation(Enum):
    FL = 0
    FR = 1
    RL = 2
    RR = 3

class NewEnv(QuadrupedEnv):
    def __init__(self, robot, state_obs_names = ..., scene = 'flat', sim_dt = 0.002, base_vel_command_type = 'forward', ref_base_lin_vel = 0.5, ref_base_ang_vel = 0, ground_friction_coeff = 1, legs_order = ('FL', 'FR', 'RL', 'RR'), sensors = None, sensors_kwargs = None, external_disturbances_kwargs = None,
                 desired_leg_pair = (LegAssociation.FL.value,LegAssociation.RR.value),undesired_leg_pair = (LegAssociation.FR.value,LegAssociation.RL.value)):
        super().__init__(robot, state_obs_names, scene, sim_dt, base_vel_command_type, ref_base_lin_vel, ref_base_ang_vel, ground_friction_coeff, legs_order, sensors, sensors_kwargs, external_disturbances_kwargs)
        self.desired_leg_pair = desired_leg_pair
        self.undesired_leg_pair = undesired_leg_pair
        self._last_action_for_reward = np.zeros(12)
        self._consecutive_legs_on_ground = 0
        self._time_since_violation = 0.0

    def set_leg_pair(self,desired_leg_pair:tuple,undesired_leg_pair:tuple):
        self.desired_leg_pair = desired_leg_pair
        self.undesired_leg_pair = undesired_leg_pair

    def distance_from_line_2D(self,p1:NDArray,p2:NDArray,p3:NDArray)->float:
        """
        Calculates the distance of the 2D point p3 from the 2D line defined by the points p1, p2
        """
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]

        numerator = abs((y2-y1)*p3[0] - (x2-x1)*p3[1] + x2*y1 - y2*x1)
        denominator = ((y2-y1)**2 + (x2-x1)**2)**0.5

        return numerator/max(denominator,1e-6)

    def reset(self, qpos = None, qvel = None, seed = None, random = True, options = None):
        self._consecutive_legs_on_ground = 0
        self._last_action_for_reward = np.zeros(12)
        self._time_since_violation = 0.0
        return super().reset(qpos, qvel, seed, random, options)

    def step(self, action):
        self._time_since_violation += self.simulation_time
        return super().step(action)

    def _compute_reward(self):
        '''
        The reward function for the learning process of the go2 quadruped to stand on the desired pair of legs

        ### Reward function terms: 
        \t#### Reward terms:
            1. lin_vel_reward (float): Reward for the go2 having close to 0 linear velocity, corresponding factor: 0.3
            2. ang_vel_reward (float): Reward for the go2 having close to 0 angular velocity, corresponding factor: 0.4
            3.

        \t#### Penalty terms:
        
        ### Returns:
            something
        '''

        # Value initialization
        alive_bonus = 1.0
        first_desired_foot_contact_reward = 0.0
        second_desired_foot_contact_reward = 0.0
        both_desired_foot_reward = 0.0

        #linear/Angular Velocity Reward
        lin_vel = self.base_lin_vel(frame="base")[:2]
        ang_vel = self.base_ang_vel(frame="base")[:2]
        sigma_lin_vel = 0.25
        sigma_ang_vel = 0.25
        lin_vel_reward = np.exp(-np.sum(lin_vel ** 2)/(2 * sigma_lin_vel ** 2))
        ang_vel_reward = np.exp(-np.sum(ang_vel ** 2)/(2 * sigma_ang_vel ** 2))

        #Height Penalty
        height = self.com[2]
        target_height = 0.31
        sigma_height = 0.08
        height_err = max(0.0, target_height - height)
        height_penalty = 1.0 - np.exp(-(height_err ** 2) / (2 * sigma_height ** 2))

        # Z_velocity Error 
        base_z_lin_vel = self.base_lin_vel(frame="base")[2]
        z_vel_penalty = base_z_lin_vel ** 2

        # "kebab" rotation 
        roll_pitch_ang_vel = self.base_ang_vel(frame="base")[:2]
        roll_pitch_sigma = 0.10
        roll_pitch_ang_vel_penalty =1 - np.exp(-np.sum(roll_pitch_ang_vel ** 2)/(2 * roll_pitch_sigma ** 2))

        # high torque penalty
        tau = self.torque_ctrl_setpoint
        torque_penalty = np.sum(tau **2)

        # correct legs contact
        leg_contacts, contact_positions = self.feet_contact_state(frame="base",ground_reaction_forces=False)
        leg_contacts = leg_contacts.to_list()
        contact_count = sum(leg_contacts)
        feet_contact_penalty = 0
        com_offset = 1

        #Desired/Undesired Leg contact Reward/Penalty
        if contact_count == 2:
            if leg_contacts[self.desired_leg_pair[0]]:
                first_desired_foot_contact_reward = 1.0
            if leg_contacts[self.desired_leg_pair[1]]:
                second_desired_foot_contact_reward = 1.0
            if leg_contacts[self.desired_leg_pair[0]] and leg_contacts[self.desired_leg_pair[1]]:
                both_desired_foot_reward = 1.0
        if contact_count!= 2:
            self._time_since_violation = 0.0
            if contact_count == 3: feet_contact_penalty = 1.25
            if contact_count == 4: feet_contact_penalty = 1.5
            if contact_count <= 1: feet_contact_penalty = 2.

        #time
        airtime_reward = min(self._time_since_violation, 2.)

        # Center Of Mass (COM) Reward
        sigma_com = 0.05
        feet_world = self.feet_pos(frame="world").to_list()
        foot_a_xy = feet_world[self.desired_leg_pair[0]][:2]
        foot_b_xy = feet_world[self.desired_leg_pair[1]][:2]
        com_xy = self.com[:2]
        com_offset = self.distance_from_line_2D(foot_a_xy, foot_b_xy, com_xy)
        center_of_mass_reward = np.exp(-(com_offset ** 2) / (2 * sigma_com ** 2))

        #Action Rate Penalty
        current_action = self.mjData.ctrl.copy()
        if not hasattr(self,"_last_action_for_reward"):
            self._last_action_for_reward = np.zeros_like(current_action)
        action_rate_penalty = np.sum((current_action - self._last_action_for_reward)**2)
        self._last_action_for_reward = current_action  

        #Feet Heigh Reward
        invalid_contact_penalty = max(0, self.mjData.ncon - contact_count)
        feet_position = self.feet_pos(frame="base").to_list()
        feet_one_height = feet_position[self.undesired_leg_pair[0]][2]
        feet_two_height = feet_position[self.undesired_leg_pair[1]][2]
        feet_one_err = min(0.0, feet_one_height)
        feet_two_err = min(0.0, feet_two_height)
        feet_sigma = 0.12
        feet_one_reward = np.exp(-(feet_one_err ** 2) / (2 * feet_sigma ** 2))
        feet_two_reward = np.exp(-(feet_two_err ** 2) / (2 * feet_sigma ** 2))
        feet_height_reward = (feet_one_reward + feet_two_reward) / 2
        # logging = {
        #     "episode":self.step_num,
        #     "alive_bonus": 0.2,
        #     "first_foot_contact_reward": 0.25 * first_foot_contact_reward,
        #     "second_foot_contact_reward": 0.25 * second_foot_contact_reward,
        #     "super_deluxe_reward_special": 1/2 * super_deluxe_reward_special,
        #     "super_duper_deluxe": .4 * super_duper_reward_special_pro_max,
        #     "lin_vel_reward": 0.3 * lin_vel_reward,
        #     "lin_ang_reward": 0.4 * ang_vel_reward,
        #     "center_of_mass_reward": 2.0 * center_of_mass_reward,
        #     "invalid_contact_penalty": -5.5 * invalid_contact_penalty,
        #     "feet_height_penalty": -1.0 * feet_height_reward,
        #     "height_penalty": -1.0 * height_penalty,
        #     "feet_contact_penalty": -2.5 * feet_contact_penalty,
        #     "z_vel_penalty": -0.1 * z_vel_penalty,
        #     "roll_pitch_and_vel_penalty": -0.1 * roll_pitch_ang_vel_penalty,
        #     "torque_penalty": -1e-4 * torque_penalty,
        #     "action_rate_penalty": -1e-4 * action_rate_penalty
        # }
        # episode_logs.append(logging)
        # df = pd.DataFrame(episode_logs)
        # df.to_csv("./reward_logs.csv",index=False)
        
        # print(feet_one_penalty/2,feet_two_penalty/2)
        # print(super_duper_reward_special_pro_max * 0.2)
        # print(center_of_mass_reward)
        # print(airtime_reward)
        return float(
            0.2 * alive_bonus + 
            0.25 * first_desired_foot_contact_reward +
            0.25 * second_desired_foot_contact_reward +
            0.5 * both_desired_foot_reward +
            0.3 * lin_vel_reward +
            0.4 * ang_vel_reward +
            2.0 * center_of_mass_reward + #### from 2.0
            1.5 * feet_height_reward + ### from 0.5
            1.5 * airtime_reward + # 0. to 2. (seconds)
            -5.0 * invalid_contact_penalty + #### from 5.0
            -1.0 * height_penalty +
            -4.0 * feet_contact_penalty + ### from 2.0
            -0.5 * z_vel_penalty + # MAYBE
            -0.25 * roll_pitch_ang_vel_penalty + # MAYBE
            -1e-4 * torque_penalty # MAYBE
            -3e-4 * action_rate_penalty
            )

class SB3QuadrupedWrapper(gym.Wrapper):
    def __init__(self, env, obs_keys, pid=None, action_scale=0.3, decimation=4,
                 max_episode_steps=2000, termination_penalty=0.0):
        super().__init__(env)
        self.obs_keys = list(obs_keys)
        self.pid = pid if pid is not None else PIDController()
        self.action_scale = action_scale
        self.decimation = decimation              
        self.max_episode_steps = max_episode_steps
        self.termination_penalty = termination_penalty
        self._elapsed_steps = 0

        first_obs, _ = self._raw_reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._flatten(first_obs).shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    def _flatten(self, obs) -> NDArray:
        if isinstance(obs, dict):
            return np.concatenate(
                [np.atleast_1d(obs[k]).astype(np.float32).ravel() for k in self.obs_keys])
        return np.asarray(obs, dtype=np.float32).ravel()

    def _raw_reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return out
        return out, {}

    def reset(self, *, seed=None, options=None):
        self._elapsed_steps = 0
        obs, info = self._raw_reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        q_des = self.pid.q_nominal + self.action_scale * action
        mj_data = self.env.unwrapped.mjData

        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self.decimation):
            q = mj_data.qpos[7:19]
            dq = mj_data.qvel[6:18]
            torque = self.pid.get_action(q, dq, q_des)
            obs, reward, terminated, truncated, info = self.env.step(torque)
            total_reward += float(reward)
            if terminated or truncated:
                break

        flat_obs = self._flatten(obs)
        if not np.all(np.isfinite(flat_obs)):
            flat_obs = np.nan_to_num(flat_obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated = True

        reward = total_reward / self.decimation
        if terminated:
            reward -= self.termination_penalty

        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps and not terminated:
            truncated = True

        return flat_obs, reward, bool(terminated), bool(truncated), info


if __name__ == "__main__":
    print("This file defines the environment and is meant to be imported")