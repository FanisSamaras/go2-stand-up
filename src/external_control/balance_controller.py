import numpy as np
from numpy.typing import NDArray
from enum import Enum
from envs import NewEnv

class LegAssociation(Enum):
    FL = 0
    FR = 1
    RL = 2
    RR = 3

class BalanceController():
    """An external controller for the simulated go2-quadruped robot.
     It acts in a specified `env` using the provided `policy`"""
    
    def __init__(self, env:NewEnv, trained_policy, target_velocity:NDArray, leg_pair:tuple = (LegAssociation.FL.value,LegAssociation.RR.value)):
        self.env = env
        self.trained_policy = trained_policy
        self.target_vel = target_velocity
        self.leg_pair = leg_pair

        self.env.set_leg_pair(leg_pair)

    def reset(self, seed:int = None):
        self.env.reset(seed)

    def act(self, observation:NDArray) -> tuple[NDArray,dict]:
        action = self.policy(observation)
        _, _, _, _, info = self.env.step(action)
        return action, info
    
    def __str__(self):
        return f"This is a controller to balance a simulated go2-quadruped robot on the leg_pair {self.leg_pair}"

if __name__ == "__main__":
    print("this file sets the external balance controller class and is not meant to be executed")