import os

# Limit numerical-library threads BEFORE importing numpy/torch, my laptop can't handle much
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from imitation.algorithms import bc, dagger
from imitation.data import serialize as imitation_serialize
from imitation.data import types as imitation_types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger as imit_logger

from envs import create_env, IMITATION_OBS
from envs.imitation_env import SB3QuadrupedWrapper
from internal_control.PID import PIDController

th.set_num_threads(1)

try:
    th.set_num_interop_threads(1)
except RuntimeError:
    pass

BASE_SEED = 42

N_ENVS = 4
N_EXPERT_WORKERS = 4

OBS_DIM = 48
ACTION_DIM = 12

MAX_STEPS = 3_000_000
EPISODE_LENGTH = 3000
SAVE_INTERVAL = 50_000
WARMUP_STEPS = 75

ACTION_SCALE = 0.25
DECIMATION = 10
TERMINATION_PENALTY = 500.0

EXPERT_DIR = "./src/policies/pretrained/backlegs_policy.onnx"
CHECKPOINT_DIR = Path("./src/policies/checkpoint_dagger")
DAGGER_SCRATCH_DIR = CHECKPOINT_DIR / "dagger_state"
EXPERT_DATASET = CHECKPOINT_DIR / "expert_demonstrations.npz"
EXPERT_TRAJECTORIES = CHECKPOINT_DIR / "expert_trajectories.npz"
BC_INITIALIZED = CHECKPOINT_DIR / "bc_initialized"
FINAL_DAGGER = CHECKPOINT_DIR / "final_dagger"

EXPERT_TOTAL_STEPS = 300_000
EXPERT_NOISE_STD = 0.02

INITIAL_BC_EPOCHS = 10
BC_BATCH_SIZE = 256

DAGGER_ROUNDS = 10
DAGGER_ROUND_STEPS = 50_000
DAGGER_BC_EPOCHS = 4
DAGGER_BETA_RAMP_ROUNDS = 10

DAGGER_WINDOW_ROUNDS = 2

REUSE_SAVED_EXPERT_DATA = True
REUSE_BC_INITIALIZATION = True

MODE = "visualize" # train | tune | visualize

VISUALIZE_POLICY = CHECKPOINT_DIR / "dagger_ppo_150000_steps.zip"

VISUALIZE_STEPS = 3000


def free_memory():
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()


def memory_report(label=""):
    """
    Print RAM usage of this process and its children if psutil
    happens to be installed.

    Falls back gracefully otherwise.
    """

    try:

        import psutil

        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        total_mb = total / (1024**2)
        print(f"[RAM] {label}: " f"{total_mb:.1f} MB " f"(parent + children)")

        return

    except Exception:
        pass

    try:
        status = Path("/proc/self/status").read_text()
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                print(f"[RAM] {label}: " f"{kb / 1024:.1f} MB " f"(parent only)")

                return

    except Exception:
        pass


def make_env():

    env = create_env(
        type="imitation",
        scene="flat",
    )

    env = SB3QuadrupedWrapper(
        env,
        obs_keys=IMITATION_OBS,
        pid=PIDController(),
        action_scale=ACTION_SCALE,
        decimation=DECIMATION,
        max_episode_steps=EPISODE_LENGTH,
        termination_penalty=TERMINATION_PENALTY,
    )

    return Monitor(env)


def make_dagger_env():
    return RolloutInfoWrapper(make_env())


def make_parallel_dagger_env(
    n_envs=N_ENVS,
    seed=BASE_SEED,
):

    vec_env = SubprocVecEnv(
        [make_dagger_env for _ in range(n_envs)],
        start_method="spawn",
    )

    vec_env.seed(seed)

    return vec_env


class ONNXExpertPolicy(BasePolicy):

    def __init__(
        self,
        path,
        observation_space,
        action_space,
    ):

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            squash_output=False,
        )

        self.path = str(path)

        self._create_session()

    def _create_session(self):

        options = ort.SessionOptions()

        # Avoid every ONNX worker spawning many CPU threads.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        self.output_name = self.session.get_outputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape

    def _run_onnx(
        self,
        obs,
    ):

        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        if obs.ndim == 1:

            obs = obs[None, :]

        if obs.ndim != 2 or obs.shape[1] != OBS_DIM:

            raise RuntimeError(
                f"Expected observation " f"(*, {OBS_DIM}), " f"got {obs.shape}"
            )

        batch_dim = self.input_shape[0]

        fixed_batch_one = (
            isinstance(
                batch_dim,
                int,
            )
            and batch_dim == 1
        )

        if fixed_batch_one and obs.shape[0] > 1:

            actions = []

            for ob in obs:

                action = self.session.run(
                    [self.output_name],
                    {self.input_name: ob[None, :]},
                )[0]

                action = np.asarray(
                    action,
                    dtype=np.float32,
                )

                if action.ndim == 2:
                    action = action[0]

                actions.append(action)

            actions = np.asarray(
                actions,
                dtype=np.float32,
            )

        else:

            actions = self.session.run(
                [self.output_name],
                {self.input_name: obs},
            )[0]

            actions = np.asarray(
                actions,
                dtype=np.float32,
            )

        if actions.ndim == 1:

            actions = actions[None, :]

        expected = (
            obs.shape[0],
            ACTION_DIM,
        )

        if actions.shape != expected:

            raise RuntimeError(
                f"ONNX expert produced " f"{actions.shape}; " f"expected {expected}"
            )

        actions = np.clip(
            actions,
            self.action_space.low,
            self.action_space.high,
        )

        return actions.astype(np.float32)

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
    ):

        del deterministic

        obs = observation.detach().cpu().numpy().astype(np.float32)

        actions = self._run_onnx(obs)

        return th.as_tensor(
            actions,
            dtype=th.float32,
            device=observation.device,
        )

    def __getstate__(self):

        state = self.__dict__.copy()

        for key in (
            "session",
            "input_name",
            "output_name",
            "input_shape",
        ):

            state.pop(
                key,
                None,
            )

        return state

    def __setstate__(
        self,
        state,
    ):

        self.__dict__.update(state)

        self._create_session()


def create_student(
    env,
):

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        # Smaller than before.
        n_steps=1024,
        batch_size=256,
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
                pi=[
                    256,
                    256,
                ],
                vf=[
                    256,
                    256,
                ],
            )
        ),
        verbose=1,
        device="cpu",
    )


def reset_and_settle(
    env,
    seed=None,
):

    zero_action = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    obs, info = env.reset(seed=seed)

    settled = 0

    while settled < WARMUP_STEPS:

        (
            obs,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(zero_action)

        if terminated or truncated:

            obs, info = env.reset()

            settled = 0

        else:

            settled += 1

    return obs


def collect_expert_worker(
    args,
):

    worker_id, n_steps = args

    worker_seed = BASE_SEED + worker_id

    rng = np.random.default_rng(worker_seed)

    env = make_env()

    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        EXPERT_DIR,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    output_name = session.get_outputs()[0].name

    obs_buffer = []
    action_buffer = []
    next_obs_buffer = []
    done_buffer = []

    obs = reset_and_settle(
        env,
        worker_seed,
    )

    episode = 0

    for _ in range(n_steps):

        obs_array = np.asarray(
            obs,
            dtype=np.float32,
        )

        expert_action = session.run(
            [output_name],
            {input_name: obs_array[None, :]},
        )[0]

        expert_action = np.asarray(
            expert_action,
            dtype=np.float32,
        )

        if expert_action.ndim == 2:

            expert_action = expert_action[0]

        obs_buffer.append(obs_array.copy())

        action_buffer.append(expert_action.copy())

        executed_action = expert_action.copy()

        if EXPERT_NOISE_STD > 0:

            executed_action += rng.normal(
                0.0,
                EXPERT_NOISE_STD,
                size=ACTION_DIM,
            ).astype(np.float32)

        executed_action = np.clip(
            executed_action,
            env.action_space.low,
            env.action_space.high,
        )

        (
            next_obs,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(executed_action)

        done = bool(terminated or truncated)

        next_obs_buffer.append(
            np.asarray(
                next_obs,
                dtype=np.float32,
            ).copy()
        )

        done_buffer.append(done)

        if done:

            episode += 1

            obs = reset_and_settle(
                env,
                worker_seed + 1000 + episode,
            )

        else:

            obs = next_obs

    env.close()

    return (
        np.asarray(
            obs_buffer,
            dtype=np.float32,
        ),
        np.asarray(
            action_buffer,
            dtype=np.float32,
        ),
        np.asarray(
            next_obs_buffer,
            dtype=np.float32,
        ),
        np.asarray(
            done_buffer,
            dtype=bool,
        ),
    )


def build_transitions(
    obs,
    acts,
    next_obs,
    dones,
):

    infos = np.asarray(
        [{} for _ in range(len(obs))],
        dtype=object,
    )

    return imitation_types.Transitions(
        obs=obs,
        acts=acts,
        infos=infos,
        next_obs=next_obs,
        dones=dones,
    )


def collect_expert_data():

    n_workers = min(
        N_EXPERT_WORKERS,
        EXPERT_TOTAL_STEPS,
    )

    per_worker = EXPERT_TOTAL_STEPS // n_workers

    remainder = EXPERT_TOTAL_STEPS % n_workers

    jobs = [
        (
            i,
            per_worker + (1 if i < remainder else 0),
        )
        for i in range(n_workers)
    ]

    print("\n" + "=" * 70)

    print("PARALLEL EXPERT COLLECTION")

    print(
        "Workers:",
        n_workers,
    )

    print(
        "Transitions:",
        EXPERT_TOTAL_STEPS,
    )

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
    ) as executor:

        shards = list(
            executor.map(
                collect_expert_worker,
                jobs,
            )
        )

    obs = np.concatenate(
        [x[0] for x in shards],
        axis=0,
    )

    acts = np.concatenate(
        [x[1] for x in shards],
        axis=0,
    )

    next_obs = np.concatenate(
        [x[2] for x in shards],
        axis=0,
    )

    dones = np.concatenate(
        [x[3] for x in shards],
        axis=0,
    )

    np.savez_compressed(
        EXPERT_DATASET,
        obs=obs,
        acts=acts,
        next_obs=next_obs,
        dones=dones,
    )

    demonstrations = build_transitions(
        obs,
        acts,
        next_obs,
        dones,
    )

    print(
        "\nEXPERT DATA SAVED:",
        EXPERT_DATASET,
    )

    return demonstrations


def load_expert_data():

    data = np.load(
        EXPERT_DATASET,
        allow_pickle=True,
    )

    obs = data["obs"].astype(np.float32)

    acts = data["acts"].astype(np.float32)

    next_obs = data["next_obs"].astype(np.float32)

    dones = data["dones"].astype(bool)

    print("LOADED EXPERT SAVEPOINT")

    print(
        "Samples:",
        len(obs),
    )

    return build_transitions(
        obs,
        acts,
        next_obs,
        dones,
    )


def collect_or_load_expert_data():

    if REUSE_SAVED_EXPERT_DATA and EXPERT_DATASET.exists():

        return load_expert_data()

    return collect_expert_data()


def initialize_bc(
    student,
    demonstrations,
):

    bc_trainer = bc.BC(
        observation_space=student.observation_space,
        action_space=student.action_space,
        policy=student.policy,
        demonstrations=demonstrations,
        batch_size=BC_BATCH_SIZE,
        ent_weight=1e-4,
        l2_weight=1e-6,
        rng=np.random.default_rng(BASE_SEED),
        device="cpu",
    )

    print("INITIAL BEHAVIORAL CLONING")

    bc_trainer.train(
        n_epochs=INITIAL_BC_EPOCHS,
        progress_bar=True,
        log_rollouts_n_episodes=0,
    )

    student.save(str(BC_INITIALIZED))

    print(
        "Saved:",
        str(BC_INITIALIZED) + ".zip",
    )

    del bc_trainer
    free_memory()

    return student


def latest_sb3_dagger_checkpoint():

    paths = list(CHECKPOINT_DIR.glob("dagger_round_*.zip"))

    if not paths:

        return None

    def number(path):

        try:

            return int(
                path.stem.rsplit(
                    "_",
                    1,
                )[-1]
            )

        except ValueError:

            return -1

    return max(
        paths,
        key=number,
    )


def current_round_demo_dir(
    round_num,
):

    return DAGGER_SCRATCH_DIR / "demos" / f"round-{round_num:03d}"


def recent_dagger_window(
    trainer,
):
    """
    Bound DAgger RAM usage.

    Upstream DAgger keeps self._all_demos from all historical
    rounds.

    We intentionally discard the in-memory historical list and
    tell DAgger to reload only the latest DAGGER_WINDOW_ROUNDS
    from disk.

    Example:

        current round = 6
        window = 2

    loads only:
        round 5
        round 6

    rather than:
        round 0 ... round 6
    """

    current_round = trainer.round_num
    trainer._all_demos = []

    if hasattr(
        trainer.bc_trainer,
        "_demo_data_loader",
    ):

        trainer.bc_trainer._demo_data_loader = None

    trainer._last_loaded_round = max(
        -1,
        current_round - DAGGER_WINDOW_ROUNDS,
    )

    free_memory()


def compact_trainer_before_save(
    trainer,
):
    """
    Keep policy + optimizer + DAgger state, but avoid serializing
    a giant list of all historical trajectories.
    """

    trainer._all_demos = []

    if hasattr(
        trainer.bc_trainer,
        "_demo_data_loader",
    ):

        trainer.bc_trainer._demo_data_loader = None

    trainer._last_loaded_round = max(
        -1,
        trainer.round_num - DAGGER_WINDOW_ROUNDS,
    )

    free_memory()


def reconstruct_dagger_trainer(
    scratch_dir,
    venv,
):

    checkpoint = Path(scratch_dir) / "checkpoint-latest.pt"

    print(
        "Loading trusted DAgger checkpoint:",
        checkpoint,
    )

    custom_logger = imit_logger.configure()

    trainer = th.load(
        checkpoint,
        map_location=th.device("cpu"),
        # This is our own trusted local checkpoint.
        weights_only=False,
    )

    trainer.venv = venv

    trainer.logger = custom_logger

    if hasattr(
        trainer.bc_trainer,
        "_bc_logger",
    ):

        trainer.bc_trainer._bc_logger._logger = custom_logger

    trainer.bc_trainer.batch_size = BC_BATCH_SIZE

    trainer.bc_trainer.minibatch_size = BC_BATCH_SIZE

    recent_dagger_window(trainer)

    return trainer


def sync_trainer_to_student(
    trainer,
    student,
):

    student.policy.load_state_dict(trainer.bc_trainer.policy.state_dict())

    return student


def sync_student_to_trainer(
    student,
    trainer,
):

    trainer.bc_trainer.policy.load_state_dict(student.policy.state_dict())

    return trainer


def demo_dir_stats(
    demo_dir,
):

    demo_dir = Path(demo_dir)

    if not demo_dir.exists():

        return (
            0,
            0,
        )

    n_trajectories = 0
    n_transitions = 0

    for path in sorted(demo_dir.glob("*.npz")):

        trajectories = imitation_serialize.load(path)

        for traj in trajectories:

            n_trajectories += 1

            n_transitions += len(traj.acts)

        del trajectories

        gc.collect()

    return (
        n_trajectories,
        n_transitions,
    )


def recover_current_round(
    trainer,
    student,
    env,
):

    round_num = trainer.round_num

    demo_dir = current_round_demo_dir(round_num)

    (
        n_trajectories,
        n_transitions,
    ) = demo_dir_stats(demo_dir)

    if n_trajectories == 0:

        return (
            trainer,
            student,
        )

    print("FOUND SAVED CURRENT-ROUND DEMOS")

    print(
        "Round:",
        round_num,
    )

    print(
        "Trajectories:",
        n_trajectories,
    )

    print(
        "Transitions:",
        n_transitions,
    )

    newer_ppo = CHECKPOINT_DIR / ("dagger_round_" f"{round_num + 1:03d}.zip")

    complete = newer_ppo.exists() or (
        n_trajectories >= N_ENVS and n_transitions >= DAGGER_ROUND_STEPS
    )

    if not complete:

        print("Current round is incomplete.")

        print(
            "Deleting only:",
            demo_dir,
        )

        shutil.rmtree(demo_dir)

        free_memory()

        print(f"Round {round_num} " "will be recollected.")

        return (
            trainer,
            student,
        )

    print("Saved round is complete.")

    print("Reusing demos; no rollout recollection.")

    recent_dagger_window(trainer)

    memory_report("before recovery BC")

    trainer.extend_and_update(
        bc_train_kwargs={
            "n_epochs": DAGGER_BC_EPOCHS,
            "log_rollouts_n_episodes": 0,
            "progress_bar": True,
        }
    )

    free_memory()

    if newer_ppo.exists():

        print(
            "Restoring existing PPO:",
            newer_ppo,
        )

        student = PPO.load(
            str(newer_ppo),
            env=env,
            device="cpu",
        )

        sync_student_to_trainer(
            student,
            trainer,
        )

    else:

        sync_trainer_to_student(
            trainer,
            student,
        )

    compact_trainer_before_save(trainer)

    (
        checkpoint_path,
        policy_path,
    ) = trainer.save_trainer()

    free_memory()

    sb3_path = CHECKPOINT_DIR / ("dagger_round_" f"{trainer.round_num:03d}")

    student.save(str(sb3_path))

    free_memory()

    print("\nRECOVERY COMPLETE")

    print(
        "Round:",
        trainer.round_num,
    )

    print(
        "DAgger:",
        checkpoint_path,
    )

    print(
        "PPO:",
        str(sb3_path) + ".zip",
    )

    return (
        trainer,
        student,
    )


def initialise_dagger(
    env,
):

    bc_zip = Path(str(BC_INITIALIZED) + ".zip")

    if REUSE_BC_INITIALIZATION and bc_zip.exists():

        print(
            "Loading existing BC policy:",
            bc_zip,
        )

        student = PPO.load(
            str(BC_INITIALIZED),
            env=env,
            device="cpu",
        )

    else:

        print("No BC checkpoint found.")

        print("Loading expert dataset for initial BC...")

        demonstrations = collect_or_load_expert_data()

        student = create_student(env)

        student = initialize_bc(
            student,
            demonstrations,
        )

        del demonstrations

        free_memory()

    if DAGGER_SCRATCH_DIR.exists():

        print(
            "Removing stale DAgger scratch:",
            DAGGER_SCRATCH_DIR,
        )

        shutil.rmtree(DAGGER_SCRATCH_DIR)

    DAGGER_SCRATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expert = ONNXExpertPolicy(
        EXPERT_DIR,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=student.policy,
        demonstrations=None,
        batch_size=BC_BATCH_SIZE,
        ent_weight=1e-4,
        l2_weight=1e-6,
        rng=np.random.default_rng(BASE_SEED + 1),
        device="cpu",
    )

    beta_schedule = dagger.LinearBetaSchedule(rampdown_rounds=DAGGER_BETA_RAMP_ROUNDS)

    trainer = dagger.SimpleDAggerTrainer(
        venv=env,
        scratch_dir=DAGGER_SCRATCH_DIR,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        beta_schedule=beta_schedule,
        rng=np.random.default_rng(BASE_SEED + 2),
    )

    compact_trainer_before_save(trainer)

    trainer.save_trainer()

    free_memory()

    return (
        trainer,
        student,
    )


def train_dagger():

    env = make_parallel_dagger_env()

    print("RAM-OPTIMIZED PARALLEL DAGGER")

    print(
        "Parallel environments:",
        N_ENVS,
    )

    print(
        "Target rounds:",
        DAGGER_ROUNDS,
    )

    print(
        "New steps / round:",
        DAGGER_ROUND_STEPS,
    )

    print(
        "BC epochs / round:",
        DAGGER_BC_EPOCHS,
    )

    print(
        "BC batch:",
        BC_BATCH_SIZE,
    )

    print(
        "DAgger memory window:",
        DAGGER_WINDOW_ROUNDS,
        "rounds",
    )

    native_checkpoint = DAGGER_SCRATCH_DIR / "checkpoint-latest.pt"

    if native_checkpoint.exists():

        print("\nExisting native DAgger checkpoint found.")

        print("Skipping 300k expert dataset entirely.")

        trainer = reconstruct_dagger_trainer(
            DAGGER_SCRATCH_DIR,
            env,
        )

        print(
            "Resumed DAgger round:",
            trainer.round_num,
        )

        latest_ppo = latest_sb3_dagger_checkpoint()

        if latest_ppo is not None:

            print(
                "Loading PPO:",
                latest_ppo,
            )

            student = PPO.load(
                str(latest_ppo),
                env=env,
                device="cpu",
            )

        else:

            bc_zip = Path(str(BC_INITIALIZED) + ".zip")

            if not bc_zip.exists():

                raise FileNotFoundError(
                    "No DAgger PPO checkpoint " "and no BC checkpoint."
                )

            student = PPO.load(
                str(BC_INITIALIZED),
                env=env,
                device="cpu",
            )

        sync_trainer_to_student(
            trainer,
            student,
        )

        (
            trainer,
            student,
        ) = recover_current_round(
            trainer,
            student,
            env,
        )

    else:

        trainer, student = initialise_dagger(env)

    free_memory()

    memory_report("after startup")

    while trainer.round_num < DAGGER_ROUNDS:

        current_round = trainer.round_num

        beta = trainer.beta_schedule(current_round)

        print(f"DAGGER ROUND " f"{current_round} " f"(target {DAGGER_ROUNDS})")

        print(
            "beta:",
            beta,
        )

        print(
            "new transitions:",
            DAGGER_ROUND_STEPS,
        )

        print(
            "training window:",
            DAGGER_WINDOW_ROUNDS,
            "rounds",
        )

        recent_dagger_window(trainer)

        memory_report("before rollout")

        trainer.train(
            total_timesteps=DAGGER_ROUND_STEPS,
            rollout_round_min_episodes=N_ENVS,
            rollout_round_min_timesteps=DAGGER_ROUND_STEPS,
            bc_train_kwargs={
                "n_epochs": DAGGER_BC_EPOCHS,
                # Avoid extra evaluation episodes during BC.
                "log_rollouts_n_episodes": 0,
                "progress_bar": True,
            },
        )

        # ----------------------------------------------------
        # BC-updated DAgger policy -> complete PPO object.
        # ----------------------------------------------------

        sync_trainer_to_student(
            trainer,
            student,
        )

        free_memory()

        # ----------------------------------------------------
        # Remove demonstrations BEFORE serializing trainer.
        #
        # This makes checkpoint-latest.pt much smaller.
        # ----------------------------------------------------

        compact_trainer_before_save(trainer)

        memory_report("before DAgger save")

        # ----------------------------------------------------
        # SAVE NATIVE TRAINER FIRST.
        #
        # If laptop dies after this point, native round state
        # is already safe.
        # ----------------------------------------------------

        (
            checkpoint_path,
            policy_path,
        ) = trainer.save_trainer()

        free_memory()

        # ----------------------------------------------------
        # SAVE FULL PPO SECOND.
        # ----------------------------------------------------

        sb3_path = CHECKPOINT_DIR / ("dagger_round_" f"{trainer.round_num:03d}")

        student.save(str(sb3_path))

        free_memory()

        memory_report("after checkpoint")

        print("\n" + "-" * 70)

        print("DAGGER SAVEPOINT CREATED")

        print("-" * 70)

        print(
            "Completed round:",
            trainer.round_num,
        )

        print(
            "DAgger state:",
            checkpoint_path,
        )

        print(
            "DAgger policy:",
            policy_path,
        )

        print(
            "PPO:",
            str(sb3_path) + ".zip",
        )

        print("-" * 70)

    # ========================================================
    # FINAL SAVE
    # ========================================================

    sync_trainer_to_student(
        trainer,
        student,
    )

    compact_trainer_before_save(trainer)

    trainer.save_trainer()

    free_memory()

    student.save(str(FINAL_DAGGER))

    free_memory()

    print("DAGGER COMPLETE")

    print(
        "Final policy:",
        str(FINAL_DAGGER) + ".zip",
    )

    return (
        student,
        env,
        trainer,
    )


def visualize_policy(
    policy_path,
    steps=VISUALIZE_STEPS,
):

    policy_path = Path(policy_path)

    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    print("VISUALIZING POLICY")

    print(
        "Policy:",
        policy_path,
    )

    model = PPO.load(
        str(policy_path),
        device="cpu",
    )

    env = make_env()
    obs = reset_and_settle(env)

    for step in range(steps):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        env.render()

        if step % 100 == 0:

            print(
                f"step={step:5d} "
                f"reward="
                f"{float(reward):9.3f} "
                f"|action|max="
                f"{np.abs(action).max():.3f}"
            )

        if terminated or truncated:

            print(f"Episode terminated " f"at step {step}")

            obs = reset_and_settle(env)

    env.close()


def fine_tune_bc_with_ppo(
    agent,
    vec_env,
    total_timesteps: int = MAX_STEPS,
):

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_INTERVAL // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="dagger_ppo",
        verbose=2,
    )

    print("PARALLEL PPO FINE-TUNING")
    print("Environments:", N_ENVS)

    agent.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=True,
    )

    agent.save(CHECKPOINT_DIR + "final_dagger_ppo")
    vec_env.close()

    return agent


if __name__ == "__main__":
    mp.freeze_support()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MODE == "visualize":
        visualize_policy(
            VISUALIZE_POLICY,
            VISUALIZE_STEPS,
        )
    if MODE == "tune":
        vec_env = make_parallel_dagger_env()
        student = PPO.load(FINAL_DAGGER, vec_env)
        student.learning_rate = 2e-5
        student.target_kl = 0.01
        student.ent_coef = 0.0002
        fine_tune_bc_with_ppo(student, vec_env, 500_000)

    else:
        student, vec_env, dagger_trainer = train_dagger()
        vec_env.close()
        free_memory()
