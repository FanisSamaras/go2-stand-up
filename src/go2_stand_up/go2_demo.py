from external_control.balance_controller import BalanceController, POLICIES_DIR
from internal_control.PID import PIDController
from go2_stand_up import LegPairs
from envs import create_env,FULL_STATE_OBS,IMITATION_OBS
from envs.base_env import SB3QuadrupedWrapper as SB3QuadrupedWrapperBase
from envs.imitation_env import SB3QuadrupedWrapper as SB3QuadrupedWrapperImitation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path
import numpy as np
import argparse

ACTION_SCALE_BASE = 0.3
ACTION_SCALE_IMITATION = 0.25
DECIMATION = 10
EPISODE_LENGTH = 2000
TERMINATION_PENALTY = 300
WARMUP_STEPS = 75

class Demonstrate:
    def __init__(self, BALANCE_POINTS):
        self.BALANCE_POINTS = BALANCE_POINTS
        
    def make_env_base(self):
        env = create_env(
            type="base",
            scene="flat",
            state_obs_names="full_state",
            balance_points=self.BALANCE_POINTS
        )
        env = SB3QuadrupedWrapperBase(
            env,
            obs_keys=FULL_STATE_OBS,
            pid=PIDController(balance_points=self.BALANCE_POINTS),
            action_scale=ACTION_SCALE_BASE,
            decimation=DECIMATION,
            max_episode_steps=EPISODE_LENGTH,
            termination_penalty=TERMINATION_PENALTY,
        )
        return Monitor(env)

    def make_env_imitation(self):
        env = create_env(
            type="imitation",
            scene="flat",
            balance_points=self.BALANCE_POINTS
        )
        env = SB3QuadrupedWrapperImitation(
            env,
            obs_keys=IMITATION_OBS,
            pid=PIDController(balance_points=self.BALANCE_POINTS),
            action_scale=ACTION_SCALE_IMITATION,
            decimation=DECIMATION,
            max_episode_steps=EPISODE_LENGTH,
            termination_penalty=TERMINATION_PENALTY,
        )
        return Monitor(env) 
        
    def _set_env_controller(self):
        if self.BALANCE_POINTS in [0,1]:
            self.env = DummyVecEnv([self.make_env_base])
            self.controller = BalanceController(env=self.env,balance_points=self.BALANCE_POINTS)
        elif self.BALANCE_POINTS in [2,3]:
            self.env = self.make_env_imitation()
            self.controller = BalanceController(env=self.env,balance_points=self.BALANCE_POINTS)
        else:
            raise
    def demonstration(self,steps:int = 2000):
        total_reward = 0
        self._set_env_controller()
        match self.BALANCE_POINTS:
            case 0:
                self.path = Path(POLICIES_DIR + "fl_rr_ppo_go2_vecnormalize_4400000_steps.pkl")
            case 1:
                self.path = Path(POLICIES_DIR + "fr_rl_ppo_go2_vecnormalize_4300000_steps.pkl")

        if self.BALANCE_POINTS in [0,1]:
            vec_env = VecNormalize.load(self.path,venv=self.env)
            obs = vec_env.reset()
            for _ in range(1,steps+1):
                action, _ = self.controller.action(observation=obs)
                obs, reward, done, info = vec_env.step(actions=action)
                total_reward += reward
                self.env.envs[0].render()
                if done[0]:
                    break
            print(f"Final Reward: {total_reward}")

        if self.BALANCE_POINTS in [2,3]:
            obs = self.env.reset()
            for _ in range(WARMUP_STEPS):
                obs, reward, terminated, truncated, info = self.env.step(np.zeros(12, dtype=np.float32))
                self.env.render()
                if terminated or truncated:
                    break

            for step in range(steps):
                action, _ = self.controller.action(observation=obs)
                obs, reward, terminated, truncated, _ = self.env.step(action=action)
                self.env.render() 
                if terminated or truncated:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Demonstration",
        description="It demonstrates",
        epilog="Cool demonstration!"
    )
    parser.add_argument("balance_points",type=str,help="Select the legs you want demonstrated: FL_RR, FR_RL, FL_FR, RL_RR")


    name = parser.parse_args().balance_points
    leg_pair = LegPairs[name].value
    
    if leg_pair not in [0,1,2,3]:
        raise ValueError(f"Balance points not implemented yet, please refer to the documentation")

    demon = Demonstrate(leg_pair)

    demon.demonstration()

