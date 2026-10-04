[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)

## Goal-Conditioned Two-Leg Balancing with RL

- Fanis Samaras
- Manos Drakos $^2$

This project is an attempt to train a `Go2-Quadruped` to maintain balance in one of the 4 desired positions using RL. The env used for this project was the [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped/tree/master), based on the gym environment

The 4 desired positions all revolve around the idea of balancing the quadruped only using 2 legs so the desired configurations are:
1. {FL,RR}
2. {FR,RL}
3. {FL,FR}
4. {RL,RR}

### Diagonal balance points
----

For the diagonal balance points since they were the "easier" configuration of the 2, we used Residual Learning combined with SB3's implementation of the PPO algorithm. 
$$action = q_{nominal} + rl_{action} * action\ scale$$

For the training we used a slightly different $q_{nominal}$ since it would be hard to discover the balancing positions all on it's own, we gave it a starting position with it's `undesired_legs` already slightly lifted from the ground 

### Handstand and (leg)stand balance points
----

Since there are already plenty of developed projects tackling these 2 configurations, we decided to use behavioral cloning. We used some onnx [pretrained policies](https://github.com/Renkunzhao/legged_rl_deploy/tree/master/policies/go2/My_unitree_go2_gym) in combination with the imitation library, since it is compatible with SB3. The program then loads and samples **observation - action** pairs of an "expert" policy, and trains our PPO policy via supervised learning using the imitation bc algorithm. 

Note that we can further train the policy *explicitly* with PPO, to limit any undesired expert behaviors. In our demo, we attempted to limit the imitated gait

### Reward function Design
----

For all 4 configurations the key concepts of what should be rewarded/punished remain pretty similar. Since we wanted for the robot to balance around a stationary position the task gets simplified.

Rewards:

$$r_{rewards} = r_{\text{height}} + r_{\text{leg1/2height}} + r_{\text{correct stance}} + r_{\text{CoM}}$$


Penalties:

$$r_{penalties} = c_{\text{lin/ang vel}} + c_{\text{u1/u2 contact}} + c_{\text{knee contact}} + c_{\text{d1/d2 vel}}$$


Stabilizing terms:

$$r_{\text{stabilizing terms}} = c_{torque} + c_{stomp} + c_{\text{action rate}} + c_{work}$$

Final Reward:

$$r_d = r_{rewards} - r_{penalties} - r_{\text{stabilizing terms}}$$

Where $d_1,d_2$ and $u_1,u_2$ are the desired and undesired legs respectively.

It is also really crucial for the stabilization of the quadruped that the Center of Mass (CoM) is over the support line drawn between it's 2 balance points (desired feet) hence the CoM reward term

## Requirements

- python-3.11 (up to 3.13 for python 3.14 and above there seems to be issues with some dependencies)
- uv (not necessary but provides the most seamless installation process)

## Installation

Clone the repository to your device

```bash
cd <your/preferred/workspace/dir>
git clone https://github.com/FanisSamaras/go2-stand-up
```

We recommend creating a virtual env (venv) 

With uv:
```bash
uv sync 
```
And it will automatically spin up the new env with the correct python version and dependencies

with pip:
```bash
python3 -m venv <desired_venv_name>
# activate the virtual environment
pip install -r requirements.txt
```


The project comes with 4 pre-trained policies each corresponding to one of the 4 aforementioned leg configurations. It also includes a simple demo file along with a specialized controller that works for each of the desired configurations

To run the demo 

```bash
cd ./your_workspace/
python3 ./src/go2_stand_up/go2_demo.py <arg> 
# args = {FL_RR, FR_RL, FL_FR, RL_RR}
```

The paths to load the polices are all relative so it's important to be in the correct directory.
