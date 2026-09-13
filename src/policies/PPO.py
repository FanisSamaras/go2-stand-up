import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    '''Actor Critic Algorithm for choosing the next action'''
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_dim=512,
            init_log_std=-0.5,
            min_log_var=-5.0,
            max_log_var=1.0):
        super().__init__()

        self.min_log_var = min_log_var 
        self.max_log_var = max_log_var 

        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )

        self.mu = nn.Linear(hidden_dim,action_dim)

        self.log_var = nn.Linear(hidden_dim,action_dim)

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )

        # Useful for residual learning:
        # initial policy outputs near-zero residual actions.

        nn.init.uniform_(self.log_var.weight,-1e-4,1e-4)
        nn.init.constant_(self.log_var.bias, init_log_std)

    def forward(self, obs):
        actor_features = self.actor_body(obs)

        mu = torch.tanh(self.mu(actor_features))

        log_var = self.log_var(actor_features)
        log_var = torch.clamp(
            log_var,
            self.min_log_var,
            self.max_log_var
        )

        var = torch.exp(log_var)

        value = self.critic(obs).squeeze(-1)

        return mu,var,value
    
    def act(self, obs):
        mu, var, value = self.forward(obs)
        dist = torch.distributions.Normal(mu, var)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        mu, var, value = self.forward(obs)

        dist = torch.distributions.Normal(mu,var)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

class RolloutBuffer:
    def __init__(self, obs_dim, action_dim, rollout_steps, device):
        self.obs = np.zeros((rollout_steps,obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps,action_dim), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps,),dtype=np.float32)
        self.rewards = np.zeros((rollout_steps,),dtype=np.float32)
        self.dones = np.zeros((rollout_steps,),dtype=np.float32)
        self.values = np.zeros((rollout_steps,),dtype=np.float32)

        self.ptr = 0 
        self.rollout_steps = rollout_steps
        self.device = device

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def is_full(self):
        return self.ptr == self.rollout_steps
    
    def clear(self):
        self.ptr = 0

    def compute_returns_and_advantages(
            self,
            last_value,
            gamma=0.99,
            gae_lambda=0.95
    ):
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.rewards)

        gae = 0.0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps -1:
                next_value = last_value
                next_non_terminal = 1.0  - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
        
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )

            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

        return advantages, returns
    
    def get_tensors(self,advantages, returns):
        obs = torch.tensor(self.obs, dtype=torch.float32,device=self.device)
        actions = torch.tensor(self.actions,dtype=torch.float32,device=self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages,dtype=torch.float32,device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32,device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        return obs, actions, old_log_probs, advantages, returns
    
class PPO:
    def __init__(
            self,
            obs_dim,
            action_dim,
            device="cuda",
            lr=3e-4,
            gamma=0.98,
            gae_lambda=0.95,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=5,
            minibatch_size=256):
        
        self.device = torch.device(device)

        self.policy = ActorCritic(obs_dim,action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=lr,
            eps=1e-5
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.act(obs_t)

        # print(f"action:{action}\nlog_prob{log_prob}\nvalue{value}")

        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy()
        )
    def value(self, obs):
        obs_t = torch.tensor(obs,dtype=torch.float32,device=self.device)

        with torch.no_grad():
            _,_, value = self.policy(obs_t)

        return value.cpu().numpy()
    
    def update(self, buffer, last_value):
        advantages, returns = buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        obs,actions, old_log_probs, advantages, returns = buffer.get_tensors(
            advantages=advantages,
            returns=returns
        )

        n = obs.shape[0]
        indices = np.arange(n)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0,n,self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    mb_obs,
                    mb_actions
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps
                ) * mb_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.smooth_l1_loss(values,mb_returns)

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                num_updates += 1

        return {
            "policy_loss" : total_policy_loss / num_updates,
            "value_loss" : total_value_loss / num_updates,
            "entropy" : total_entropy / num_updates,
            "approx_kl" : total_approx_kl / num_updates
        }
    
    def save(self, path):
        torch.save(
            {
                "policy" : self.policy.state_dict(),
                "optimizer" : self.optimizer.state_dict()
            },
            path
        )

    def load(self, path): 
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])