import numpy as np
from numpy.typing import NDArray

class PIDController:
    '''
    Simple PID controller based around a single q_nominal position trying to stabilize around this position
    \n Leg order FL, FR, RL, RR
    '''
    def __init__(self) -> None:
        self.q_nominal = np.array([0.0, 0.9, -1.8,   #1.FL                
                                    -0.18, 0.73, -1.85,   #2.FR
                                    +0.18, 0.73, -1.85,   #3.RL
                                    0.0, 0.9, -1.8]   #4.RR
                                   ,dtype=np.float32)
        self.kp = np.array([20,35,45] * 4, dtype=np.float32) * 5
        self.kd = self.kp/18

    def get_action(self,q,dq) -> NDArray:
        action = self.kp * (self.q_nominal - q) - self.kd * dq
        torque_limit = np.array([23.7, 23.7, 45.3] * 4, dtype=np.float32)
        return np.clip(action,-torque_limit,torque_limit)

if __name__ == "main" :
    print("This file sets the internal PID controller and is not meant to be executed ")
