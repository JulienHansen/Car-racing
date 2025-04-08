#!/usr/bin/env python3
"""
Self-contained DDPG implementation for the CarRacing-v0 environment.
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
        a = torch.tensor(a).float()
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
        # Convolutional layers to extract features from the image input (3 x 96 x 96)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # -> (32, 22, 22)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 8, 8)
            nn.ReLU()
        )
        conv_out_size = 64 * 8 * 8
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        # Steering output: [-1,1]
        self.fc_steer = nn.Linear(512, 1)
        # Gas and Brake outputs: [0,1]
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
        # Convolutional layers for state feature extraction
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
        # Fully connected layers to combine state features with action input (3-dim)
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
        # Preprocess observation: convert (96,96,3) to (3,96,96) and scale to [0,1]
        obs_proc = np.transpose(obs, (2, 0, 1)) / 255.0
        obs_tensor = torch.from_numpy(obs_proc).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(obs_tensor)
        self.actor.train()
        return a.cpu().numpy()[0]

    def optimize(self):
        if not self.memory.is_ready():
            return

        x, a, r, x_next, done = self.memory.sample()

        # Update Critic
        self.critic.train()
        q = self.critic(x, a)
        with torch.no_grad():
            a_next = self.actor_tar(x_next)
            q_next = self.critic_tar(x_next, a_next)
            target = r + (1 - done) * self.gamma * q_next
        critic_loss = self.criterion(q, target)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # Update Actor
        actor_loss = -self.critic(x, self.actor(x)).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # Soft update target networks
        for param_tar, param in zip(self.actor_tar.parameters(), self.actor.parameters()):
            param_tar.data.copy_(param.data * self.tau + param_tar.data * (1.0 - self.tau))
        for param_tar, param in zip(self.critic_tar.parameters(), self.critic.parameters()):
            param_tar.data.copy_(param.data * self.tau + param_tar.data * (1.0 - self.tau))

# ----------------------
# Main Training Loop
# ----------------------

def main(render: bool = False, n_episodes: int = 500):
    gym.logger.min_level = 40
    # Standard environment name is CarRacing-v2 in Gymnasium
    env = gym.make('CarRacing-v3', continuous=True) # Ensure continuous actions
    max_steps = 1000
    rewards = []

    agent = DDPG(env, gamma=0.99)
    noise = OrnsteinUhlenbeck(env.action_space)

    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        # --- CORRECTED RESET ---
        obs, info = env.reset() # Unpack observation and info
        noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            if render:
                env.render()

            # --- Pass ONLY obs (the array) to action ---
            a = agent.action(obs)
            a = noise.action(a) # Apply noise AFTER getting the deterministic action

            # --- CORRECTED STEP ---
            obs_next, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated # Combine termination conditions

            # Preprocess observations for storage (convert to torch tensor with shape (3,96,96))
            # Ensure obs and obs_next are the numpy arrays before processing
            obs_proc = torch.from_numpy(np.transpose(obs, (2, 0, 1)) / 255.0).float()
            obs_next_proc = torch.from_numpy(np.transpose(obs_next, (2, 0, 1)) / 255.0).float()

            agent.memory.push((obs_proc, a, reward, obs_next_proc, float(done))) # Store done as float
            if agent.memory.is_ready():
                agent.optimize()

            obs = obs_next # Update obs for the next loop iteration
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)
        # Use TQDM's set_postfix for cleaner progress updates
        tqdm.write(f"Episode {episode+1}/{n_episodes}, Reward: {episode_reward:.2f}")

    # Plot and save the training rewards
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('DDPG on CarRacing-v2')
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
