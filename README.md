# Unitree Go2 - Two-Leg Balancing

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Simulation-blue)
![PPO](https://img.shields.io/badge/RL-PPO-7952B3)

**Reinforcement learning and imitation learning for two-leg balancing of a Unitree Go2 quadruped in MuJoCo.**

Developed by **Fanis Samaras** and **Manos Drakos**$^2$ as part of the Intelligent Control course.

## Overview

This project explores learning-based control strategies for balancing a simulated Unitree Go2 quadruped on only two of its four legs.

The objective is to maintain a stationary, dynamically balanced posture across four distinct support configurations, while minimizing undesired body motion, unnecessary contacts, and excessive control effort.

We investigate two complementary approaches:

- **Residual Reinforcement Learning:** PPO-based policies that learn corrections to nominal joint configurations for diagonal balancing.
- **Behavioral Cloning:** Imitation of pretrained expert policies for front- and rear-leg balancing, with optional PPO fine-tuning.

The project uses the [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped) environment, built on MuJoCo, and Stable-Baselines3 for reinforcement learning.

### Design Decisions: From a Unified Policy to Configuration-Specific Controllers

Our initial objective was to develop a single goal-conditioned reinforcement learning policy capable of balancing the Unitree Go2 across all four two-leg support configurations.

However, learning a unified control strategy proved challenging due to the substantially different balancing requirements of each configuration. In particular, the policy had to learn distinct joint configurations and stabilization behaviors while simultaneously optimizing across multiple balancing objectives.

To make the problem more tractable, we adopted a modular approach consisting of four independently trained policies, one for each support configuration.

For the diagonal configurations (FL–RR and FR–RL), we used residual reinforcement learning with PPO. For the front and rear-leg configurations (FL–FR and RL–RR), we used behavioral cloning from pretrained expert policies, with the option of subsequent PPO fine-tuning.

This approach allowed us to focus on the individual control challenges of each configuration rather than requiring a single policy to solve all four tasks simultaneously.

**A unified goal-conditioned policy remains a natural direction for future work.**

## Balancing Configurations

The robot is trained to balance using the following support-leg combinations:

| Configuration | Supporting Legs | Learning Approach |
|---|---|---|
| FL–RR | Front Left + Rear Right | Residual RL (PPO) |
| FR–RL | Front Right + Rear Left | Residual RL (PPO) |
| FL–FR | Front Left + Front Right | Behavioral Cloning |
| RL–RR | Rear Left + Rear Right | Behavioral Cloning |

Each configuration presents different stability and control challenges due to the geometry of the support line and the resulting constraints on the robot's center of mass.

![](assets/demo.gif)

## Methodology

### 1. Diagonal Balancing — Residual Reinforcement Learning

For the two diagonal configurations, we use Proximal Policy Optimization (PPO), implemented through Stable-Baselines3.

Rather than learning joint targets entirely from scratch, the policy learns residual corrections around a predefined nominal joint configuration.

The desired joint positions are calculated as:

$$
q_{\mathrm{des}} = q_{\mathrm{nominal}} + \alpha \pi_\theta(s)
$$

where:

- $q_{\mathrm{nominal}}$ represents the nominal joint configuration.
- $\pi_\theta(s)$ is the action produced by the learned policy.
- $\alpha$ is the action scaling factor.
- $q_{\mathrm{des}}$ represents the resulting desired joint positions.

To facilitate exploration, we modify the nominal pose so that the two non-supporting legs are already slightly elevated.

This provides the policy with an initial configuration closer to the target balancing posture, reducing the difficulty of discovering useful behaviors through random exploration.

### 2. Front and Rear Balancing — Behavioral Cloning

For the front-leg and rear-leg configurations, we employ behavioral cloning to leverage existing expert policies.

We use pretrained ONNX policies from the [legged_rl_deploy](https://github.com/Renkunzhao/legged_rl_deploy/tree/master/policies/go2/My_unitree_go2_gym) project.

The expert policies are executed in simulation to collect observation-action pairs:

$$
\mathcal{D} = \{(s_i,a_i)\}_{i=1}^{N}
$$

These demonstrations are then used to train a student policy through supervised learning using the `imitation` library, which integrates with Stable-Baselines3.

The behavioral cloning objective is to reproduce the expert's actions from the corresponding observations.

Following imitation learning, the resulting policy can be further optimized using PPO.

This allows us to modify the learned behavior through task-specific rewards, such as discouraging unwanted gait motion while maintaining balance.

### 3. Joint-Level PD Control

The learned policies generate desired joint positions rather than directly applying actuator torques.

These position targets are converted into joint torques using a proportional-derivative (PD) controller:

$$
\tau = K_p(q_{\mathrm{des}}-q) - K_d\dot q
$$

where:

- $\tau$ is the commanded joint torque.
- $K_p$ and $K_d$ are proportional and derivative gains.
- $q$ and $\dot q$ are the measured joint positions and velocities.
- $q_{\mathrm{des}}$ are the desired joint positions.

This architecture separates high-level learned behavior from low-level joint tracking.

The RL policy learns which joint configurations support balance, while the PD controller provides corrective torques to track those configurations.

## Reward Function Design

All four balancing tasks share a common reward design philosophy: encourage stable two-leg support while penalizing unwanted motion, contact violations, and excessive control effort.

### Positive Rewards

$$
r_{\mathrm{positive}} =
r_{\mathrm{height}}+
r_{\mathrm{leg\ height}}+
r_{\mathrm{stance}}+
r_{\mathrm{CoM}}
$$

These terms encourage:

- Maintaining an appropriate body height.
- Raising the two non-supporting legs.
- Preserving the intended support configuration.
- Keeping the center of mass aligned with the support geometry.

### Motion and Contact Penalties

$$
c_{\mathrm{motion}} =
c_{\mathrm{linear/angular\ velocity}}+
c_{\mathrm{unwanted\ contact}}+
c_{\mathrm{knee\ contact}}+
c_{\mathrm{support\ leg\ velocity}}
$$

These terms discourage unnecessary movement, unintended foot or knee contacts, and instability.

### Control Regularization

$$
c_{\mathrm{control}} =
c_{\mathrm{torque}}+
c_{\mathrm{stomp}}+
c_{\mathrm{action\ rate}}+
c_{\mathrm{work}}
$$

These penalties discourage excessive actuator effort, abrupt contact behavior, rapid action changes, and unnecessary mechanical work.

### Final Reward

$$
r = r_{\mathrm{positive}}-c_{\mathrm{motion}}-c_{\mathrm{control}}
$$

The relative weighting of individual terms can be adjusted according to the balancing configuration.

### Center of Mass and Support Geometry

A central challenge of two-leg balancing is maintaining the robot's center of mass relative to the support line connecting the two supporting feet.

The CoM reward encourages the horizontal projection of the robot's center of mass to remain close to this line.

This is a useful geometric objective for maintaining balance, although dynamic stability also depends on velocities, angular momentum, and contact forces.

## Installation

### Requirements

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) (recommended)

### Clone the Repository

```bash
git clone https://github.com/FanisSamaras/go2-stand-up.git
cd go2-stand-up
```

### Install with uv (Recommended)

```bash
uv sync
```

This creates a project environment and installs the dependencies specified in `pyproject.toml` and `uv.lock`.

### Alternative: pip


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, the activation command differs.

## Running the Demo

The repository includes four pretrained policies and a demonstration script.

Choose one of the supported configurations:

```bash
uv run python src/go2_stand_up/go2_demo.py FL_RR
```

Available options:

- `FL_RR` — Diagonal balancing
- `FR_RL` — Diagonal balancing
- `FL_FR` — Front-leg balancing
- `RL_RR` — Rear-leg balancing

Run the command from the repository root because the policy paths are currently relative.

## Results

For the full video results of each configuration go to [assets](assets)

## References and Acknowledgments

This project builds upon the following open-source tools and research implementations:

- [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped) — Quadruped simulation environment.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation.
- [imitation](https://github.com/HumanCompatibleAI/imitation) — Behavioral cloning framework.
- [legged_rl_deploy](https://github.com/Renkunzhao/legged_rl_deploy) — Pretrained expert policies.

**Authors:** Fanis Samaras and Manos Drakos $^2$.