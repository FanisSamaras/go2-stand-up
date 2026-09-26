from envs import create_env, FULL_STATE_OBS
from envs.base_env import SB3QuadrupedWrapper  
from internal_control.PID import PIDController
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


N_ENVS = 4
MAX_STEPS = 3_000_000
EPISODE_LENGTH = 3000
SAVE_INTERVAL = 100_000
ACTION_SCALE = 0.3
DECIMATION = 4
TERMINATION_PENALTY = 500.

CHECKPOINT_DIR = "./src/policies/checkpoint/"
TENSORBOARD_DIR = "./src/go2_stand_up/tensor"


def make_env():
    env = create_env()
    env = SB3QuadrupedWrapper(
        env,
        obs_keys=FULL_STATE_OBS,
        pid=PIDController(),
        action_scale=ACTION_SCALE,
        decimation=DECIMATION,
        max_episode_steps=EPISODE_LENGTH,
        termination_penalty=TERMINATION_PENALTY,
    )
    return Monitor(env)  # logs ep_rew_mean / ep_len_mean


def train():
    base_vec = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecNormalize(base_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    agent = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=1,
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_INTERVAL // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_go2",
        save_vecnormalize=True,
        verbose=2,
    )

    agent.learn(total_timesteps=MAX_STEPS, callback=checkpoint_callback)
    agent.save(CHECKPOINT_DIR + "final")
    vec_env.save(CHECKPOINT_DIR + "final_vecnormalize.pkl")


def load_test(num_of_steps:int, steps: int = 2000):
    base_vec = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(f"{CHECKPOINT_DIR}ppo_go2_vecnormalize_{num_of_steps}_steps.pkl", base_vec)
    vec_env.training = False
    vec_env.norm_reward = False

    agent = PPO.load(f"{CHECKPOINT_DIR}ppo_go2_{num_of_steps}_steps.zip", env=vec_env, device="cpu")
    obs = vec_env.reset()
    for _ in range(steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        base_vec.envs[0].render()
        if done[0]:
            break

def resume_training(extra_timesteps: int = MAX_STEPS):
    base_vec = DummyVecEnv([make_env for _ in range(N_ENVS)])

    vec_env = VecNormalize.load(
        f"{CHECKPOINT_DIR}final_vecnormalize.pkl",
        base_vec,
    )

    vec_env.training = True
    vec_env.norm_reward = True

    agent = PPO.load(
        f"{CHECKPOINT_DIR}final.zip",
        env=vec_env,
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_INTERVAL // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_go2_resumed",
        save_vecnormalize=True,
        verbose=2,
    )

    agent.learn(
        total_timesteps=extra_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    agent.save(f"{CHECKPOINT_DIR}final_resumed")
    vec_env.save(f"{CHECKPOINT_DIR}final_resumed_vecnormalize.pkl")

if __name__ == "__main__":
    pass
