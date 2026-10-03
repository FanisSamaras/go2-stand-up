from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU
import mujoco
import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from enum import Enum
from internal_control.PID import PIDController
from pathlib import Path
from filelock import FileLock
import time, tempfile

device = "cuda" if torch.cuda.is_available() else "cpu"

_orignial_imu_init = IMU.__init__


def patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model, mj_data)
    _orignial_imu_init(self, mj_model, mj_data, *args, **kwargs)
    self.step()


class PatchedIMU(IMU):
    def __init__(self, mj_model, mj_data, *args, **kwargs):
        patched_imu_init(self, mj_model, mj_data, *args, **kwargs)


class LegAssociation(Enum):
    FL = 0
    FR = 1
    RL = 2
    RR = 3


_INIT_LOCK = str(Path(tempfile.gettempdir()) / "go2_env_init.lock")


class NewEnv(QuadrupedEnv):
    def __init__(
        self,
        robot,
        state_obs_names=...,
        scene="flat",
        sim_dt=0.002,
        base_vel_command_type="forward",
        ref_base_lin_vel=0.5,
        ref_base_ang_vel=0,
        ground_friction_coeff=1,
        legs_order=("FL", "FR", "RL", "RR"),
        sensors=None,
        sensors_kwargs=None,
        external_disturbances_kwargs=None,
        desired_leg_pair=(LegAssociation.FL.value, LegAssociation.RR.value),
        undesired_leg_pair=(LegAssociation.FR.value, LegAssociation.RL.value),
    ):
        for attempt in range(8):
            try:
                with FileLock(_INIT_LOCK, timeout=120):
                    super().__init__(
                        robot,
                        state_obs_names,
                        scene,
                        sim_dt,
                        base_vel_command_type,
                        ref_base_lin_vel,
                        ref_base_ang_vel,
                        ground_friction_coeff,
                        legs_order,
                        sensors,
                        sensors_kwargs,
                        external_disturbances_kwargs,
                    )
                break
            except ValueError:
                if attempt == 4:
                    raise
                time.sleep(1.0 + attempt)

        self.desired_leg_pair = desired_leg_pair
        self.undesired_leg_pair = undesired_leg_pair
        self._last_action_for_reward = np.zeros(12)
        self._step_count_episode = 0
        self._correct_stance = 0.0
        self._correct_stance_counter = 0
        self._incorrect_stance_count = 200

    def set_leg_pair(
        self, desired_leg_pair: tuple[int], undesired_leg_pair: tuple[int]
    ):
        self.desired_leg_pair = desired_leg_pair
        self.undesired_leg_pair = undesired_leg_pair

    def reset(self, qpos=None, qvel=None, seed=None, random=True, options=None):
        self._step_count_episode = 0
        self._correct_stance = 0.0
        self._correct_stance_counter = 0
        self._incorrect_stance_count = 200
        return super().reset(qpos, qvel, seed, random, options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._correct_stance:
            self._correct_stance_counter += 1
        if self._correct_stance_counter > 1000 and not self._correct_stance:
            self._incorrect_stance_count -= 1
        if self._incorrect_stance_count < 0:
            terminated = True

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """
        Reward for balancing the Go2 on a diagonal leg pair.
        """
        BASE_HEIGHT = 0.3
        TARGET_UNDESIRED_LEG_HEIGHT = 0.2
        FEET_OFFSET = 0.02
        SIGMA_HEIGHT = 0.05
        total_z_force = 0

        # 1. Base Motion Errors
        lin_vel = self.base_lin_vel(frame="world")
        ang_vel = self.base_ang_vel(frame="world")

        lin_vel_err = np.sum(lin_vel**2)
        roll, pitch, yaw = ang_vel
        roll_pitch_err = roll**2 + pitch**2
        yaw_rate_penalty = yaw**2  # Added yaw penalty

        # Center of Mass / Height reward
        height_err = self.com[2] - BASE_HEIGHT
        height_reward = np.exp(-(height_err**2) / (2 * SIGMA_HEIGHT**2))

        # 2. Leg Pairs & Contact State Checks
        d1, d2 = self.desired_leg_pair
        u1, u2 = self.undesired_leg_pair

        contacts_bool, _, ground_dict = self.feet_contact_state(
            frame="world", ground_reaction_forces=True
        )
        ground_forces_per_leg = ground_dict.to_list()
        leg_contacts = contacts_bool.to_list()
        d1_contact = leg_contacts[d1]
        d2_contact = leg_contacts[d2]
        u1_contact = leg_contacts[u1]
        u2_contact = leg_contacts[u2]

        desired_contacts = int(d1_contact) + int(d2_contact)
        undesired_contacts = int(u1_contact) + int(u2_contact)

        # 3. Lift Height Rewards (Threshold instead of sharp Gaussian penalty)
        feet_positions = self.feet_pos(frame="world").to_list()
        u1_height = feet_positions[u1][2] - FEET_OFFSET
        u2_height = feet_positions[u2][2] - FEET_OFFSET

        # Gives full reward (1.0) if foot is >= target height, smooth roll-off below
        u1_height_reward = (
            np.clip(u1_height / TARGET_UNDESIRED_LEG_HEIGHT, 0.0, 1.0)
            if undesired_contacts == 0
            else 0.0
        )
        u2_height_reward = (
            np.clip(u2_height / TARGET_UNDESIRED_LEG_HEIGHT, 0.0, 1.0)
            if undesired_contacts == 0
            else 0.0
        )

        # 4. Velocities of Grounded Feet (Prevent foot sliding)
        feet_velocities = self.feet_vel(frame="world").to_list()
        d1_vel_sq = np.sum(feet_velocities[d1] ** 2)
        d2_vel_sq = np.sum(feet_velocities[d2] ** 2)
        desired_feet_slip_penalty = d1_vel_sq + d2_vel_sq

        # 5. Control Penalties (Scalarized)
        # Assumes joint torques; pull from actuator_force if using position actuators
        tau = self.mjData.ctrl.copy()
        joint_vel = self.mjData.qvel[6:18]

        torque_penalty = np.sum(tau**2)
        work_penalty = np.sum(np.abs(joint_vel * tau))

        current_action = self.mjData.ctrl.copy()
        action_rate_penalty = np.sum(
            (self._last_action_for_reward - current_action) ** 2
        )
        self._last_action_for_reward = current_action

        # Knee contact penalty
        knee_contact_penalty = max(
            0, self.mjData.ncon - (desired_contacts + undesired_contacts)
        )

        # Ground forces big
        undesired_forces_1 = ground_forces_per_leg[u1]
        undesired_forces_2 = ground_forces_per_leg[u2]

        stomp_penalty = np.sum(undesired_forces_1**2) + np.sum(undesired_forces_2**2)

        # CoM
        # Extract XY coordinates
        x1, y1 = feet_positions[d1][0], feet_positions[d1][1]
        x2, y2 = feet_positions[d2][0], feet_positions[d2][1]
        x0, y0 = self.com[0], self.com[1]

        numerator = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        denominator = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        com_distance = numerator / (denominator + 1e-6)

        com_support_reward = np.exp(-(com_distance**2) / (2 * 0.05**2))

        correct_stance = (
            1.0
            if (
                desired_contacts == 2
                and undesired_contacts == 0
                and knee_contact_penalty == 0
                and u1_height > 0.12
                and u2_height > 0.12
            )
            else 0.0
        )
        com_support_reward *= correct_stance
        self._correct_stance = correct_stance

        # 6. Final Reward Weighting
        reward = (
            0.8 * height_reward
            + 2.0 * u1_height_reward
            + 2.0 * u2_height_reward
            + 1.0 * correct_stance
            + 1.0 * com_support_reward
            + -0.4 * lin_vel_err
            + -0.2 * roll_pitch_err
            + -0.4 * yaw_rate_penalty
            + -2.0 * undesired_contacts
            + -5.0 * knee_contact_penalty
            - 0.5 * desired_feet_slip_penalty
            + -1e-4 * torque_penalty
            + -4e-4 * stomp_penalty
            + -1e-4 * action_rate_penalty
            + -1e-4 * work_penalty
        )

        return float(reward)


class SB3QuadrupedWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        obs_keys,
        pid=None,
        action_scale=0.3,
        decimation=4,
        max_episode_steps=2000,
        termination_penalty=10.0,
    ):
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
            low=-np.inf,
            high=np.inf,
            shape=self._flatten(first_obs).shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

    def _flatten(self, obs) -> NDArray:
        if isinstance(obs, dict):
            return np.concatenate(
                [
                    np.atleast_1d(obs[k]).astype(np.float32).ravel()
                    for k in self.obs_keys
                ]
            )
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
        n_sub = 0
        terminated = truncated = False
        for _ in range(self.decimation):
            q = mj_data.qpos[7:19]
            dq = mj_data.qvel[6:18]
            torque = self.pid.get_action(q, dq, q_des)
            obs, reward, terminated, truncated, info = self.env.step(torque)
            total_reward += float(reward)
            n_sub += 1
            if terminated or truncated:
                break

        flat_obs = self._flatten(obs)
        if not np.all(np.isfinite(flat_obs)):
            flat_obs = np.nan_to_num(flat_obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated = True

        reward = total_reward / max(n_sub, 1)
        if terminated:
            reward -= self.termination_penalty

        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps and not terminated:
            truncated = True

        base = self.env.unwrapped
        info = {
            "reward_terms": dict(getattr(base, "reward_terms", {})),
            "termination_reason": getattr(base, "termination_reason", None),
        }
        return flat_obs, reward, bool(terminated), bool(truncated), info


if __name__ == "__main__":
    print("This file defines the environment and is meant to be imported")
