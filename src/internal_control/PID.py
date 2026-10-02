import numpy as np
from numpy.typing import NDArray

class PIDController:
    '''
    Simple PD controller around a q_nominal joint position.
    The RL policy shifts the target (q_desired) around q_nominal; this class turns the target into torques.
    \n Leg order FL, FR, RL, RR
    '''
    def __init__(self) -> None:
        # self.q_nominal = np.array(
        #             [0.0, 0.9, -1.8,
        #             0.0, 0.9, -1.8,
        #             0.0, 0.9, -1.8,
        #             0.0, 0.9, -1.8],
        #             dtype=np.float32,
        #         )
        self.q_nominal = np.array(
                    [0.0, 0.9, -1.8,
                    0.0, 1.5, -2.4,
                    0.0, 1.5, -2.4,
                    0.0, 0.9, -1.8],
                    dtype=np.float32,
                )
        self.q_stand = np.array([
                    0.1, 0.8, -1.5,     # FL
                    -0.1, 0.8, -1.5,     # FR
                    0.1, 1.0, -1.5,     # RL
                    -0.1, 1.0, -1.5,     # RR
                ], dtype=np.float32)
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
