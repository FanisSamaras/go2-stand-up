import numpy as np
from numpy.typing import NDArray
from enum import Enum
from envs import NewEnv, ImitationEnv
from go2_stand_up.go2_stand_up import make_env as make_env_base
# from go2_stand_up import load_base, load_imitation
from go2_stand_up.go2_stand_up_imitation import make_env as make_env_imitation
from stable_baselines3 import PPO
from pathlib import Path

POLICIES_DIR = "./src/policies/final_policies/"

class PolicyMatch(Enum):
    FL_RR = "fl_rr_ppo_go2_4400000_steps"
    FR_RL = "fr_rl_ppo_go2_4400000_steps"
    FL_FR = "handstand_bc_ppo_150000_steps"
    RL_RR = "legstand_bc_ppo"

class LegPairs(Enum):
    FL_RR = 0
    FR_RL = 1
    FL_FR = 2
    RL_RR = 3


class BalanceController:
    """
    An external controller for the simulated go2-quadruped robot.\n
    It acts in a specified `env` using the policy selected by the the appropriate `balance_points`
    """

    def __init__(
        self,
        env,
        policies_directory:str = POLICIES_DIR,
        balance_points:int = 0,
        seed:int = 42,
    ):
        self.dir = policies_directory
        self.balance_points = balance_points
        self.seed = seed
        self.env = env

        self.path = Path(self.dir + PolicyMatch(LegPairs(self.balance_points).name))

    # def _initialise_env(self):
    #     if self.balance_points in [2,3]:
    #         return make_env_imitation(balance_points=self.balance_points)
    #     elif self.balance_points in [0,1]:
    #         return make_env_base(balance_points=self.balance_points)
    #     else:
    #         print("invalid selected balance points")
    #         raise ValueError

    def reset(self):
        self.env.reset(seed = self.seed)

    def action(self, observation):
        agent = PPO.load(path = self.path, env = self.env, device="cpu")
        action = agent.predict(observation,deterministic=True)
        return action

    def __str__(self):
        return f"This is a controller to balance a simulated go2-quadruped robot on the leg_pair {LegPairs(self.balance_points).name}"


if __name__ == "__main__":
    print(
        "this file sets the external balance controller class and is not meant to be executed"
    )
