from envs import create_env, IMITATION_OBS
from envs.imitation_env import SB3QuadrupedWrapper
from internal_control.PID import PIDController
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

import numpy as np
import onnx
import onnxruntime as ort

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from imitation.algorithms import bc
from imitation.data import types as imitation_types

N_ENVS = 4
N_EXPERT_WORKERS = 4
MAX_STEPS = 3_000_000
EPISODE_LENGTH = 3000
SAVE_INTERVAL = 50_000
ACTION_SCALE = 0.25
DECIMATION = 10
TERMINATION_PENALTY = 500.0

BASE_SEED = 42

CHECKPOINT_DIR = "./src/policies/checkpoint/"
TENSORBOARD_DIR = "./src/go2_stand_up/tensor"
EXPERT_DIR = "./src/policies/pretrained/trot_policy.onnx"

BC_DATASET = "./src/policies/checkpoint/expert_demonstrations.npz"
BC_POLICY = "./src/policies/checkpoint/bc_policy"
BC_PPO = "./src/policies/checkpoint/bc_ppo"


def _collect_expert_worker(
    worker_id: int,
    n_steps: int,
    warmup_steps: int,
    noise_std: float,
    base_seed: int,
):
    worker_seed = base_seed + worker_id

    rng = np.random.default_rng(worker_seed)

    env = make_env()

    session = ort.InferenceSession(
        EXPERT_DIR,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    obs_buffer = []
    action_buffer = []
    next_obs_buffer = []
    done_buffer = []

    zero_action = np.zeros(12, dtype=np.float32)

    # ---------------------------------------------------------
    # Reset + settle into the nominal q_stand pose
    # ---------------------------------------------------------

    def reset_and_settle(seed=None):
        obs, info = env.reset(seed=seed)

        settled_steps = 0

        while settled_steps < warmup_steps:
            obs, reward, terminated, truncated, info = env.step(zero_action)

            if terminated or truncated:
                obs, info = env.reset()
                settled_steps = 0
            else:
                settled_steps += 1

        return obs

    obs = reset_and_settle(worker_seed)

    episode_index = 0

    # ---------------------------------------------------------
    # Collect this worker's shard
    # ---------------------------------------------------------

    for step in range(n_steps):
        obs_array = np.asarray(obs, dtype=np.float32)

        if obs_array.shape != (48,):
            raise RuntimeError(
                f"Worker {worker_id}: expected obs (48,), got {obs_array.shape}"
            )

        # -----------------------------------------------------
        # Expert label for state s_t
        # -----------------------------------------------------

        expert_action = session.run([output_name], {input_name: obs_array[None, :]})[0][
            0
        ]

        expert_action = np.asarray(expert_action, dtype=np.float32)

        if expert_action.shape != (12,):
            raise RuntimeError(
                f"Worker {worker_id}: expected action (12,), got {expert_action.shape}"
            )

        # Supervised pair:
        #
        #     observation_t -> expert_action_t

        obs_buffer.append(obs_array.copy())

        action_buffer.append(expert_action.copy())

        # -----------------------------------------------------
        # Execute action
        # -----------------------------------------------------

        executed_action = expert_action.copy()

        # Small noise gives different workers different
        # trajectories while the expert still supplies labels
        # for the perturbed states.
        if noise_std > 0.0:
            executed_action += rng.normal(
                0.0,
                noise_std,
                size=12,
            ).astype(np.float32)

        executed_action = np.clip(
            executed_action,
            env.action_space.low,
            env.action_space.high,
        )

        next_obs, reward, terminated, truncated, info = env.step(executed_action)

        done = bool(terminated or truncated)

        next_obs_array = np.asarray(next_obs, dtype=np.float32)

        if next_obs_array.shape != (48,):
            raise RuntimeError(
                f"Worker {worker_id}: "
                f"expected next_obs (48,), "
                f"got {next_obs_array.shape}"
            )

        next_obs_buffer.append(next_obs_array.copy())

        done_buffer.append(done)

        # -----------------------------------------------------
        # Continue / reset
        # -----------------------------------------------------

        if done:
            episode_index += 1

            obs = reset_and_settle(worker_seed + 1000 + episode_index)
        else:
            obs = next_obs

    env.close()

    observations = np.asarray(obs_buffer, dtype=np.float32)

    actions = np.asarray(action_buffer, dtype=np.float32)

    next_observations = np.asarray(next_obs_buffer, dtype=np.float32)

    dones = np.asarray(done_buffer, dtype=bool)

    print(f"Worker {worker_id}: {len(observations)} demonstrations")

    return (
        observations,
        actions,
        next_observations,
        dones,
    )


def make_parallel_vec_env(
    n_envs: int = N_ENVS,
    seed: int = BASE_SEED,
):

    vec_env = SubprocVecEnv(
        [make_env for _ in range(n_envs)],
        start_method="spawn",
    )

    vec_env.seed(seed)

    return vec_env


def collect_expert_demonstrations(
    n_steps: int = 100_000,
    warmup_steps: int = 75,
    noise_std: float = 0.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    env = make_env()

    session = ort.InferenceSession(
        EXPERT_DIR,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    obs_buffer = []
    action_buffer = []
    info_buffer = []

    def reset_and_settle():
        obs, info = env.reset()

        # Same idea as the working expert deployment:
        # settle into q_stand before enabling the expert.
        for _ in range(warmup_steps):
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(12, dtype=np.float32)
            )

            if terminated or truncated:
                obs, info = env.reset()

        return obs

    obs = reset_and_settle()

    for step in range(n_steps):
        obs_array = np.asarray(obs, dtype=np.float32)

        assert obs_array.shape == (48,), obs_array.shape

        # -----------------------------------------------------
        # Expert action for CURRENT observation
        # -----------------------------------------------------

        expert_action = session.run([output_name], {input_name: obs_array[None, :]})[0][
            0
        ]

        expert_action = np.asarray(expert_action, dtype=np.float32)

        assert expert_action.shape == (12,)

        # -----------------------------------------------------
        # Save supervised pair:
        #
        #       obs_t -> expert_action_t
        # -----------------------------------------------------

        obs_buffer.append(obs_array.copy())

        action_buffer.append(expert_action.copy())

        info_buffer.append({})

        # -----------------------------------------------------
        # Normally execute the expert action itself.
        #
        # Later we can add a SMALL amount of noise to visit
        # nearby states and improve BC robustness.
        # -----------------------------------------------------

        executed_action = expert_action.copy()

        if noise_std > 0.0:
            executed_action += rng.normal(
                loc=0.0,
                scale=noise_std,
                size=12,
            ).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(executed_action)

        if (step + 1) % 5000 == 0:
            print(f"Collected {step + 1}/{n_steps} transitions")

        if terminated or truncated:
            obs = reset_and_settle()

    env.close()

    observations = np.asarray(obs_buffer, dtype=np.float32)

    actions = np.asarray(action_buffer, dtype=np.float32)

    infos = np.asarray(info_buffer, dtype=object)

    print("\nEXPERT DATASET")
    print("=" * 70)
    print("Observations:", observations.shape)
    print("Actions:     ", actions.shape)
    print("Obs min/max: ", observations.min(), observations.max())
    print("Act min/max: ", actions.min(), actions.max())
    print("=" * 70)

    np.savez_compressed(
        BC_DATASET,
        obs=observations,
        acts=actions,
    )

    demonstrations = imitation_types.TransitionsMinimal(
        obs=observations,
        acts=actions,
        infos=infos,
    )

    return demonstrations


def collect_expert_demonstrations_parallel(
    total_steps: int = 100_000,
    n_workers: int = N_EXPERT_WORKERS,
    warmup_steps: int = 75,
    noise_std: float = 0.02,
    base_seed: int = BASE_SEED,
):

    n_workers = min(n_workers, total_steps)

    steps_per_worker = total_steps // n_workers

    remainder = total_steps % n_workers

    jobs = [steps_per_worker + (1 if i < remainder else 0) for i in range(n_workers)]

    print("\n" + "=" * 70)
    print("PARALLEL EXPERT COLLECTION")
    print("=" * 70)
    print("Workers:          ", n_workers)
    print("Total transitions:", total_steps)
    print("Worker loads:     ", jobs)
    print("=" * 70)

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
    ) as executor:
        futures = [
            executor.submit(
                _collect_expert_worker,
                worker_id,
                jobs[worker_id],
                warmup_steps,
                noise_std,
                base_seed,
            )
            for worker_id in range(n_workers)
        ]

        shards = [future.result() for future in futures]

    observations = np.concatenate([shard[0] for shard in shards], axis=0)

    actions = np.concatenate([shard[1] for shard in shards], axis=0)

    next_observations = np.concatenate([shard[2] for shard in shards], axis=0)

    dones = np.concatenate([shard[3] for shard in shards], axis=0)

    print("\nFINAL EXPERT DATASET")
    print("=" * 70)
    print("Observations:", observations.shape)
    print("Actions:     ", actions.shape)
    print("Action range:", actions.min(), actions.max())
    print("=" * 70)

    np.savez_compressed(
        BC_DATASET,
        obs=observations,
        acts=actions,
        next_obs=next_observations,
        dones=dones,
    )

    infos = np.asarray([{} for _ in range(len(observations))], dtype=object)

    demonstrations = imitation_types.Transitions(
        obs=observations,
        acts=actions,
        infos=infos,
        next_obs=next_observations,
        dones=dones,
    )

    return demonstrations


def load_expert_demonstrations():
    data = np.load(BC_DATASET, allow_pickle=True)

    observations = data["obs"].astype(np.float32)

    actions = data["acts"].astype(np.float32)

    next_observations = data["next_obs"].astype(np.float32)

    dones = data["dones"].astype(bool)

    infos = np.asarray([{} for _ in range(len(observations))], dtype=object)

    return imitation_types.Transitions(
        obs=observations,
        acts=actions,
        infos=infos,
        next_obs=next_observations,
        dones=dones,
    )


def evaluate_bc_action_error(
    policy,
    demonstrations,
    n_samples: int = 5000,
):
    n = min(n_samples, len(demonstrations.obs))

    idx = np.random.default_rng(123).choice(
        len(demonstrations.obs),
        size=n,
        replace=False,
    )

    obs = demonstrations.obs[idx]
    expert_actions = demonstrations.acts[idx]

    predicted_actions, _ = policy.predict(
        obs,
        deterministic=True,
    )

    error = predicted_actions - expert_actions

    mse = np.mean(error**2)
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))

    print("\nBC ACTION ERROR")
    print("=" * 70)
    print(f"MSE:       {mse:.6f}")
    print(f"MAE:       {mae:.6f}")
    print(f"Max error: {max_error:.6f}")
    print("=" * 70)


def inspect_onnx_policy(model_path=EXPERT_DIR):
    """
    Check that the ONNX model is valid and print its input/output signatures.
    """
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    print("\nONNX POLICY")
    print("=" * 70)

    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  name={inp.name}, shape={inp.shape}, type={inp.type}")

    print("\nOutputs:")
    for out in session.get_outputs():
        print(f"  name={out.name}, shape={out.shape}, type={out.type}")

    print("=" * 70)

    return session


def load_render_onnx(steps: int = 5000):
    # ---------------------------------------------------------
    # Create the SAME environment/wrapper used by your policy
    # ---------------------------------------------------------
    env = make_env()

    # ---------------------------------------------------------
    # Load ONNX policy
    # ---------------------------------------------------------
    session = inspect_onnx_policy(EXPERT_DIR)

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(
            f"Expected one ONNX input, but model has {len(inputs)} inputs: "
            f"{[x.name for x in inputs]}"
        )

    input_name = inputs[0].name
    output_name = outputs[0].name

    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------
    reset_result = env.reset()

    # Gymnasium:
    # obs, info = env.reset()
    #
    # Older Gym:
    # obs = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}

    print("\nENVIRONMENT")
    print("=" * 70)
    print("Observation shape:", np.asarray(obs).shape)
    print("Action space:", env.action_space)
    print("Action shape:", env.action_space.shape)
    print("=" * 70)

    expected_obs_shape = inputs[0].shape

    print("\nONNX expected input:", expected_obs_shape)
    print("Actual env observation:", np.asarray(obs).shape)

    # ---------------------------------------------------------
    # Run policy
    # ---------------------------------------------------------
    for _ in range(100):
        action = np.zeros(12, dtype=np.float32)
        result = env.step(action)
        env.render()

    for step in range(steps):
        # ONNX neural networks normally expect:
        #
        #     [batch_size, observation_dimension]
        #
        # whereas env.reset()/step() usually returns:
        #
        #     [observation_dimension]
        obs_array = np.asarray(obs, dtype=np.float32)

        if obs_array.ndim == 1:
            obs_input = obs_array[None, :]
        else:
            obs_input = obs_array

        # -----------------------------------------------------
        # ONNX inference
        # -----------------------------------------------------
        action = session.run([output_name], {input_name: obs_input})[0]

        action = np.asarray(action, dtype=np.float32)

        # Remove batch dimension
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        # -----------------------------------------------------
        # Sanity check
        # -----------------------------------------------------
        if action.shape != env.action_space.shape:
            raise RuntimeError(
                "\nONNX action dimension does not match environment.\n"
                f"ONNX action shape: {action.shape}\n"
                f"Environment action shape: {env.action_space.shape}"
            )

        # Optional safety clipping
        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high,
        )

        # -----------------------------------------------------
        # Environment step
        # -----------------------------------------------------
        result = env.step(action)

        # Gymnasium API
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        # Older Gym API
        else:
            obs, reward, done, info = result

        # -----------------------------------------------------
        # Render Go2
        # -----------------------------------------------------
        env.render()

        if step % 100 == 0:
            print(
                f"step={step:5d} "
                f"reward={float(reward):8.3f} "
                f"|action|max={np.abs(action).max():.3f}"
            )

        if done:
            print(f"Episode finished at step {step}")

            reset_result = env.reset()

            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}

    env.close()


def make_env():
    env = create_env(type="imitation", scene="flat")
    env = SB3QuadrupedWrapper(
        env,
        obs_keys=IMITATION_OBS,
        pid=PIDController(),
        action_scale=ACTION_SCALE,
        decimation=DECIMATION,
        max_episode_steps=EPISODE_LENGTH,
        termination_penalty=TERMINATION_PENALTY,
    )
    return Monitor(env)  # logs ep_rew_mean / ep_len_mean


def train_bc(
    demonstrations=None,
    n_epochs: int = 20,
):
    if demonstrations is None:
        demonstrations = load_expert_demonstrations()

    base_vec = DummyVecEnv([make_env for _ in range(N_ENVS)])

    vec_env = base_vec

    agent = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=1,
        device="cpu",
    )

    rng = np.random.default_rng(0)

    bc_trainer = bc.BC(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        policy=agent.policy,
        demonstrations=demonstrations,
        rng=rng,
        batch_size=512,
        ent_weight=1e-4,
        l2_weight=1e-6,
        device="cpu",
    )

    print("\n" + "=" * 70)
    print("BEHAVIORAL CLONING")
    print("=" * 70)

    bc_trainer.train(
        n_epochs=n_epochs,
        progress_bar=True,
    )

    bc_trainer.save_policy(BC_POLICY)

    agent.save(BC_PPO)

    print("\nBC training complete.")
    print("Saved:")
    print(" ", BC_POLICY)
    print(" ", BC_PPO)

    return agent, vec_env


def train_bc_parallel(
    demonstrations=None,
    n_epochs: int = 20,
):
    if demonstrations is None:
        demonstrations = load_expert_demonstrations()

    vec_env = make_parallel_vec_env(n_envs=N_ENVS)

    agent = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        # This is PER environment.
        # With 4 envs:
        #
        # 2048 * 4 = 8192 samples / PPO rollout
        n_steps=2048,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=1,
        device="cpu",
    )

    bc_trainer = bc.BC(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        policy=agent.policy,
        demonstrations=demonstrations,
        rng=np.random.default_rng(BASE_SEED),
        # Large logical batch
        batch_size=2048,
        # Process it as smaller minibatches.
        minibatch_size=512,
        ent_weight=1e-4,
        l2_weight=1e-6,
        device="cpu",
    )

    print("\n" + "=" * 70)
    print("BEHAVIORAL CLONING")
    print("=" * 70)

    epoch_counter = {"value": 0}

    def save_bc_checkpoint():
        epoch_counter["value"] += 1

        agent.save(f"{CHECKPOINT_DIR}bc_epoch_{epoch_counter['value']}")

    bc_trainer.train(
        n_epochs=n_epochs,
        progress_bar=True,
        on_epoch_end=save_bc_checkpoint,
    )

    agent.save(BC_PPO)

    return agent, vec_env


def train():
    base_vec = DummyVecEnv([make_env for _ in range(N_ENVS)])
    vec_env = VecNormalize(
        base_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99
    )

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


def load_test(num_of_steps: int, steps: int = 2000):
    base_vec = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(
        f"{CHECKPOINT_DIR}ppo_go2_vecnormalize_{num_of_steps}_steps.pkl", base_vec
    )
    vec_env.training = False
    vec_env.norm_reward = False

    agent = PPO.load(
        f"{CHECKPOINT_DIR}ppo_go2_{num_of_steps}_steps.zip", env=vec_env, device="cpu"
    )
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


def render_student(
    policy,
    steps: int = 5000,
    warmup_steps: int = 75,
):
    env = make_env()

    obs, info = env.reset()

    # Settle first.
    for _ in range(warmup_steps):
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(12, dtype=np.float32)
        )

        env.render()

        if terminated or truncated:
            obs, info = env.reset()

    for step in range(steps):
        action, _ = policy.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if step % 100 == 0:
            print(
                f"step={step:5d} "
                f"reward={float(reward):8.3f} "
                f"|action|max={np.abs(action).max():.3f}"
            )

        if terminated or truncated:
            print(f"Student terminated at step {step}")
            break

    env.close()


def fine_tune_bc_with_ppo(
    agent,
    vec_env,
    total_timesteps: int = MAX_STEPS,
):

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_INTERVAL // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="bc_ppo",
        verbose=2,
    )

    print("\n" + "=" * 70)
    print("PARALLEL PPO FINE-TUNING")
    print("=" * 70)
    print("Environments:", N_ENVS)
    print("=" * 70)

    agent.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=True,
    )

    agent.save(CHECKPOINT_DIR + "final_bc_ppo")

    vec_env.close()

    return agent


if __name__ == "__main__":
    vec_env = make_parallel_vec_env()
    load_render_onnx()
    # agent = PPO.load(f"{CHECKPOINT_DIR}bc_ppo_150000_steps",env = vec_env, device="cpu") #_200000_steps
    # agent.learning_rate = 2e-5
    # agent.target_kl = 0.01
    # agent.ent_coef = 0.0002
    # render_student(policy=agent)

    # mp.freeze_support()

    # demonstrations = collect_expert_demonstrations_parallel(
    #     total_steps=3_000_000,
    #     n_workers=N_EXPERT_WORKERS,
    #     warmup_steps=75,
    #     noise_std=0.02,
    # )

    # demonstrations = load_expert_demonstrations()

    # agent, vec_env = train_bc_parallel(
    #     demonstrations=demonstrations,
    #     n_epochs=20,
    # )

    # evaluate_bc_action_error(
    #     agent.policy,
    #     demonstrations,
    # )

    # agent = fine_tune_bc_with_ppo(
    #     agent,
    #     vec_env,
    #     total_timesteps=600_000,
    # )
