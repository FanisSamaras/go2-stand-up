import numpy as np
from numpy.typing import NDArray

class PIDController:
    '''
    Simple PD controller around a q_nominal joint position.
    The RL policy shifts the target (q_desired) around q_nominal; this class turns the target into torques.
    \n Leg order FL, FR, RL, RR
    '''
    def __init__(self) -> None:
        self.q_nominal = np.array([0.0, 0.9, -1.8,   #1.FL                
                                  -0.1, 1.0, -2.0,  #2.FR
                                   0.1, 1.0, -2.0,  #3.RL
                                   0.0, 0.9, -1.8]   #4.RR
                                ,dtype=np.float32)
        self.kp = np.array([20,35,45] * 4, dtype=np.float32) * 4
        self.kd = self.kp/18
        self.torque_limit = np.array([23.7, 23.7, 45.3] * 4, dtype=np.float32)

    def get_action(self, q, dq, q_desired=None) -> NDArray:
        """
        Returns the (clipped) joint torques. If q_desired is None the controller holds q_nominal.
        """
        if q_desired is None:
            q_desired = self.q_nominal
        action = self.kp * (q_desired - q) - self.kd * dq
        return np.clip(action, -self.torque_limit, self.torque_limit)

if __name__ == "__main__":
    print("This file sets the internal PID controller and is not meant to be executed ")
