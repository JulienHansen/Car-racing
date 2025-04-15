"""
Self-contained DDPG implementation for the CarRacing-v3 environment.
This implementation uses CNNs to process image observations and outputs a 3-dimensional action:
    - Steering in [-1, 1]
    - Gas and Brake in [0, 1]
"""

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------
# Preprocessing Function
# ----------------------

def preprocess(obs: np.ndarray) -> torch.Tensor:
    """Convert (H,W,C) image to (C,H,W) tensor in [0,1]"""
    return torch.from_numpy(np.transpose(obs, (2, 0, 1))).float().div(255.0)

# ----------------------
# Replay Buffer
# ----------------------

class ReplayBuffer:
    """Cyclic buffer of bounded capacity."""
    def __init__(self, capacity: int = 2**16, batch_size: int = 32):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.memory)

    def is_ready(self) -> bool:
        return len(self) >= self.batch_size

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self):
        choices = random.sample(self.memory, self.batch_size)
        x, a, r, x_next, done = zip(*choices)
        x = torch.stack(x)
        x_next = torch.stack(x_next)
        a = torch.tensor(np.array(a)).float()  # ✅ Fast conversion
        r = torch.tensor(r).float().unsqueeze(1)
        done = torch.tensor(done).float().unsqueeze(1)
        return x, a, r, x_next, done

# ----------------------
# Ornstein-Uhlenbeck Noise
# ----------------------

class OrnsteinUhlenbeck:
    """Ornstein-Uhlenbeck noisy action sampler."""
    def __init__(self, action_space: gym.spaces.Box, theta: float = 0.15, mu: float = 0.0, sigma: float = 0.2):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def draw(self):
        return np.random.normal(loc=0.0, scale=1.0, size=self.low.shape)

    def reset(self):
        self.noise = self.draw()

    def action(self, u: np.array):
        self.noise += self.theta * (self.mu - self.noise)
        self.noise += self.sigma * self.draw()
        return np.clip(u + self.noise, self.low, self.high)

# ----------------------
# CNN-based Actor and Critic
# ----------------------

class CNNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = 64 * 8 * 8
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.fc_steer = nn.Linear(512, 1)
        self.fc_acc_brake = nn.Linear(512, 2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        steer = self.tanh(self.fc_steer(features))
        acc_brake = self.sigmoid(self.fc_acc_brake(features))
        action = torch.cat([steer, acc_brake], dim=1)
        return action

class CNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = 64 * 8 * 8
        self.fc_state = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        state_features = self.fc_state(features)
        state_action = torch.cat([state_features, a], dim=1)
        q_value = self.fc(state_action)
        return q_value

# ----------------------
# DDPG Agent
# ----------------------

class DDPG:
    def __init__(self, env, gamma: float = 0.99, tau: float = 1e-2, capacity: int = 2**16,
                 batch_size: int = 32, lr_actor: float = 1e-4, lr_critic: float = 1e-3):
        self.gamma = gamma
        self.tau = tau

        self.actor = CNNActor()
        self.critic = CNNCritic()

        self.actor_tar = CNNActor()
        self.critic_tar = CNNCritic()
        self.actor_tar.load_state_dict(self.actor.state_dict())
        self.critic_tar.load_state_dict(self.critic.state_dict())

        self.memory = ReplayBuffer(capacity, batch_size)

        self.criterion = nn.MSELoss()
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def action(self, obs: np.array) -> np.array:
        obs_tensor = preprocess(obs).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(obs_tensor)
        return a.cpu().numpy()[0]

    def optimize(self):
        if not self.memory.is_ready():
            return

        x, a, r, x_next, done = self.memory.sample()

        # Critic
        q = self.critic(x, a)
        with torch.no_grad():
            a_next = self.actor_tar(x_next)
            q_next = self.critic_tar(x_next, a_next)
            target = r + (1 - done) * self.gamma * q_next
        critic_loss = self.criterion(q, target)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # Actor
        actor_loss = -self.critic(x, self.actor(x)).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # Soft updates
        for param_tar, param in zip(self.actor_tar.parameters(), self.actor.parameters()):
            param_tar.data.copy_(param.data * self.tau + param_tar.data * (1.0 - self.tau))
        for param_tar, param in zip(self.critic_tar.parameters(), self.critic.parameters()):
            param_tar.data.copy_(param.data * self.tau + param_tar.data * (1.0 - self.tau))

# ----------------------
# Main Training Loop
# ----------------------

def main(render: bool = False, n_episodes: int = 500):
    gym.logger.min_level = 40
    env = gym.make('CarRacing-v3', continuous=True) 
    max_steps = 1000
    rewards = []

    agent = DDPG(env, gamma=0.99)
    noise = OrnsteinUhlenbeck(env.action_space)

    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        obs, info = env.reset() 
        noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            if render:
                env.render()
            a = agent.action(obs)
            a = noise.action(a)

            obs_next, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            obs_proc = preprocess(obs)
            obs_next_proc = preprocess(obs_next)

            agent.memory.push((obs_proc, a, reward, obs_next_proc, float(done))) 
            if agent.memory.is_ready():
                agent.optimize()

            obs = obs_next
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)
        tqdm.write(f"Episode {episode+1}/{n_episodes}, Reward: {episode_reward:.2f}")

    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('DDPG on CarRacing-v3')
    plt.savefig('ddpg_carracing_rewards.pdf')
    plt.show()
    env.close()

# ----------------------
# Run Training
# ----------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-render', action='store_true', default=False, help="Render the environment.")
    parser.add_argument('-episodes', type=int, default=500, help="Number of episodes to train.")
    args = parser.parse_args()
    main(render=args.render, n_episodes=args.episodes)
