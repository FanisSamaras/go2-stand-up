from .base_env import PatchedIMU, NewEnv
from .imitation_env import NewEnv as ImitationEnv

FULL_STATE_OBS = (
    "base_pos",
    "base_ori_quat_wxyz",
    "base_lin_vel",
    "base_ang_vel",
    "qpos_js",
    "qvel_js"
)

IMITATION_OBS = (
    "constants",
    "base_ang_vel:base",
    "gravity_vector:base",
    "velocity_cmd",
    "qpos_js",
    "qvel_js",
    "last_action"
)

PROPRIOCEPTIVE_OBS = (
    "gravity_vector:base",
    "imu_acc",
    "imu_gyro",
    "qpos_js",
    "qvel_js"
)

EXTERNAL_DISTURBANCES ={
    "type":"reset",
    "x":(-40.,40.),
    "y":(-40.,40.),
}

def create_env(type:str = "base", scene:str = "flat", state_obs_names:str = "full_state", disturbances:dict = None)->NewEnv:
    """
    Creates the environment
    - type : *`"base"`* | `"imitation"`
    - scene : *`"flat"`* | `"perlin"`
    - state_obs_names : *`"full_state"`* | `"proprioceptive"`

    Returns
    - **NewEnv** object (QuadrupedEnv with our reward function)
    - **ImitationEnv**
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

    if type == "base":
        return NewEnv(
            robot="go2",
            scene=scene,
            state_obs_names=state_obs_name,
            base_vel_command_type="human",
            sensors=(PatchedIMU,),
            sensors_kwargs=(imu_kwargs,),
            legs_order=("FL","FR","RL","RR"),
            external_disturbances_kwargs = disturbances
        )
    elif type == "imitation":
        return ImitationEnv(
            robot="go2",
            scene=scene,
            state_obs_names=state_obs_name,
            base_vel_command_type="human",
            sensors=(PatchedIMU,),
            sensors_kwargs=(imu_kwargs,),
            legs_order=("FL","FR","RL","RR"),
            external_disturbances_kwargs = disturbances
        )
    else:
         print("invalid env type")
    
