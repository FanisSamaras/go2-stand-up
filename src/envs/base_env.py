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
        return super().reset(qpos, qvel, seed, random, options)

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

        lin_vel = self.base_lin_vel(frame="base")[:2]
        ang_vel = self.base_ang_vel(frame="base")[:2]
        sigma_lin_vel = 0.25
        sigma_ang_vel = 0.25
        lin_vel_reward = np.exp(-np.sum(lin_vel ** 2)/ (2 * sigma_lin_vel ** 2))
        ang_vel_reward = np.exp(-np.sum(ang_vel ** 2)/ (2 * sigma_ang_vel ** 2))


        height = self.com[2]
        target_height = 0.31
        sigma_height = 0.08
        height_err = max(0.0, target_height - height)
        height_penalty = 1.0 - np.exp(-(height_err ** 2) / (2 * sigma_height ** 2))



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
        leg_contacts, contact_positions = self.feet_contact_state(frame="base",ground_reaction_forces=False)
        leg_contacts = leg_contacts.to_list()
        contact_count = sum(leg_contacts)
        feet_contact_penalty = 0
        com_offset = 1
        first_foot_contact_reward = 0.0
        second_foot_contact_reward = 0.0
        super_deluxe_reward_special = 0.0
        # print((contact_count !=2) ,(not leg_contacts[self.leg_pair[0]]),(not leg_contacts[self.leg_pair[1]]))        
        if contact_count == 2:
            if leg_contacts[self.desired_leg_pair[0]]:
                first_foot_contact_reward = 1.0
            if leg_contacts[self.desired_leg_pair[1]]:
                second_foot_contact_reward = 1.0
            if first_foot_contact_reward and second_foot_contact_reward:
                super_deluxe_reward_special = 1.0
        if contact_count!= 2:
            if contact_count == 3: feet_contact_penalty = 0.5
            if contact_count == 4: feet_contact_penalty = 1.0
            if contact_count <= 1: feet_contact_penalty = 0.75

        if leg_contacts[self.undesired_leg_pair[0]]:
            third_foot_contact_penalty = 1.0
        if leg_contacts[self.undesired_leg_pair[1]]:
            fourth_foot_contact_penalty = 1.0
            
    
        sigma_com = 0.25
        feet_world = self.feet_pos(frame="world").to_list()
        foot_a_xy = feet_world[self.desired_leg_pair[0]][:2]
        foot_b_xy = feet_world[self.desired_leg_pair[1]][:2]
        com_xy = self.com[:2]

        com_offset = self.distance_from_line_2D(foot_a_xy, foot_b_xy, com_xy)
        center_of_mass_reward = np.exp(-(com_offset ** 2) / (2 * sigma_com ** 2))

        current_action = self.mjData.ctrl.copy()
        if not hasattr(self,"_last_action_for_reward"):
            self._last_action_for_reward = np.zeros_like(current_action)
        if not hasattr(self, "_consecutive_legs_on_ground"):
            self._consecutive_legs_on_ground = 0
        self._consecutive_legs_on_ground = self._consecutive_legs_on_ground + 1 if super_deluxe_reward_special else 0 
        super_duper_reward_special_pro_max = min(self._consecutive_legs_on_ground,100)
        action_rate_penalty = np.sum((current_action - self._last_action_for_reward))
        self._last_action_for_reward = current_action  
        invalid_contact_penalty = max(0, self.mjData.ncon - contact_count)
        feet_position = self.feet_pos(frame="base").to_list()
        feet_one_height = feet_position[LegAssociation.FR.value][2]
        feet_two_height = feet_position[LegAssociation.RL.value][2]
        feet_one_err = min(0.0, feet_one_height)
        feet_two_err = min(0.0, feet_two_height)
        feet_sigma = 0.09
        feet_one_penalty = np.exp(-(feet_one_err ** 2) / (2 * feet_sigma ** 2))
        feet_two_penalty = np.exp(-(feet_two_err ** 2) / (2 * feet_sigma ** 2))
        feet_height_reward = (feet_one_penalty + feet_two_penalty) / 2

        # print(
        #     f"first_foot_contact_reward: {0.25 * first_foot_contact_reward}|",
        #     f"second_foot_contact_reward: {0.25 * second_foot_contact_reward}|",
        #     f"super_deluxe_reward_special: {1 * super_deluxe_reward_special}|",
        #     f"super_duper_deluxe: {4 * super_duper_reward_special_pro_max}|",
        #     f"lin_vel_reward: {0.3 * lin_vel_reward}|",
        #     f"lin_ang_reward: {0.4 * ang_vel_reward}|",
        #     f"center_of_mass_reward: {2.0 * center_of_mass_reward}|",
        #     f"invalid_contact_penalty: {-5.0 * invalid_contact_penalty}|",
        #     f"feet_height_penalty: {-1.0 * feet_height_reward}|",
        #     f"height_penalty: {-1.0 * height_penalty}|",
        #     f"feet_contact_penalty: {-4.0 * feet_contact_penalty}|",
        #     f"z_vel_penalty: {-0.1 * z_vel_penalty}|",
        #     f"roll_pitch_and_vel_penalty: {-0.1 * roll_pitch_ang_vel_penalty}|",
        #     f"torque_penalty: {-1e-4 * torque_penalty}",
        #     f"action_rate_penalty: {-5e-3 * action_rate_penalty}|",sep="\n")
        # print(feet_one_penalty/2,feet_two_penalty/2)
        # print(super_duper_reward_special_pro_max * 0.2)
        # print(center_of_mass_reward)
        return float(
            0.2 + 
            0.25 * first_foot_contact_reward +
            0.25 * second_foot_contact_reward + 
            # 1.0 * super_deluxe_reward_special +
            0.2 * super_duper_reward_special_pro_max +
            0.3 * lin_vel_reward +
            0.4 * ang_vel_reward +
            2.0 * center_of_mass_reward +
            0.5 * feet_height_reward + 
            -5.0 * invalid_contact_penalty + 
            -1.0 * height_penalty +
            -2.0 * feet_contact_penalty +
            -0.1 * z_vel_penalty + # MAYBE
            -0.1 * roll_pitch_ang_vel_penalty + # MAYBE
            -1e-4 * torque_penalty # MAYBE
            -5e-3 * action_rate_penalty
            )
    

if __name__ == "__main__":
    print("This file defines the environment and is meant to be imported")