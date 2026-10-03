import numpy as np
from numpy.typing import NDArray




class PIDController:
    """
    Simple PD controller around a q_nominal joint position.
    The RL policy shifts the target (q_desired) around q_nominal; this class turns the target into torques.
    \n Leg order FL, FR, RL, RR
    \n Leg Pairs: 
        - FL_RR -> 0
        - FR_RL -> 1
        - FL_FR -> 2
        - RL_RR -> 3
    """

    def __init__(self, balance_points: int = 0) -> None:
        
        q_nominal_default = np.array([
                    0.0, 0.9, -1.8,
                    0.0, 0.9, -1.8,
                    0.0, 0.9, -1.8,
                    0.0, 0.9, -1.8
                    ],dtype=np.float32)

        q_nominal_FR_RL = np.array([
                    0.0, 1.5, -2.2,
                    0.0, 0.9, -1.8, 
                    0.0, 0.9, -1.8,
                    0.0, 1.5, -2.2
                    ],dtype=np.float32)

        q_nominal_FL_RR = np.array([
                    0.0, 0.9, -1.8, 
                    0.0, 1.5, -2.2,
                    0.0, 1.5, -2.2,
                    0.0, 0.9, -1.8,
                    ],dtype=np.float32)

        q_stand_2F_2R = np.array([
                    0.1,0.8,-1.5, 
                    -0.1,0.8,-1.5, 
                    0.1,1.0,-1.5,  
                    -0.1,1.0,-1.5, 
                    ],dtype=np.float32)

        match balance_points:
            case 0:
                self.q_nominal = q_nominal_FL_RR
            case 1:
                self.q_nominal = q_nominal_FR_RL
            case 2:
                self.q_nominal = q_stand_2F_2R
            case 3:
                self.q_nominal = q_stand_2F_2R
            case _:
                print("No valid leg pair selected defaulting to default q_nominal")
                self.q_nominal = q_nominal_default


        
        self.kp = np.array([40.0] * 12, dtype=np.float32)
        self.kd = np.array([1.0] * 12, dtype=np.float32)
        self.torque_limit = np.array([33.5] * 12, dtype=np.float32)

    def get_action(self, q, dq, q_desired=None) -> NDArray:
        """
        Returns the (clipped) joint torques. If q_desired is None the controller holds q_nominal.
        """
        if q_desired is None:
            q_desired = self.q_nominal
        action = self.kp * (q_desired - q) - self.kd * dq
        return np.clip(action, -self.torque_limit, self.torque_limit)


if __name__ == "__main__":
    print("This file sets the internal PID controller and is not meant to be executed")
