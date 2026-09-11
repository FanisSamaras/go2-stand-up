from .base_env import PatchedIMU, NewEnv

FULL_STATE_OBS = (
    "base_pos",
    "base_ori_quat_wxyz",
    "base_lin_vel",
    "base_ang_vel",
    "qpos_js",
    "qvel_js"
)

PROPRIOCEPTIVE_OBS = (
    "gravity_vector:base",
    "imu_acc",
    "imu_gyro",
    "qpos_js",
    "qvel_js"
)

def create_env(scene:str = "flat", state_obs_names:str = "full_state")->NewEnv:
    """
    Creates the environment
    - scene : *`"flat"`* | `"perlin"`
    - state_obs_names : *`"full_state"`* | `"proprioceptive"`
    Returns
    - **NewEnv** object (QuadrupedEnv with our reward function)
    """
    imu_kwargs = {
        "accel_name": "imu_acc",
        "gyro_name" : "imu_gyro",
        "imu_site_name" : "imu"
    }
    OBSERVATION_VARIANT = state_obs_names

    if OBSERVATION_VARIANT == "full_state":
        state_obs_name = FULL_STATE_OBS
    elif OBSERVATION_VARIANT == "proprioceptive":
        state_obs_name = PROPRIOCEPTIVE_OBS
    else:
        raise ValueError(
            f"Unknown observation variant: {OBSERVATION_VARIANT}"
        )

    return NewEnv(
        robot="go2",
        scene=scene,
        state_obs_names=state_obs_name,
        sensors=(PatchedIMU,),
        sensors_kwargs=(imu_kwargs,),
        legs_order=("FL","FR","RL","RR")
    )
