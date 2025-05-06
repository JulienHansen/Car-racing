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
# Prioritized Replay Buffer
# ----------------------
class PrioritizedReplayBuffer:
    """Proportional PER buffer with importance sampling weight support."""
    def __init__(self, capacity: int = 2**18, batch_size: int = 32,
                 alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 1_000_000,
                 eps: float = 1e-6):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        self.pos = 0
        self.full = False
        self.memory = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1

    def __len__(self):
        return self.capacity if self.full else self.pos

    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))

    def push(self, transition):
        # store transition
        self.memory[self.pos] = transition
        # assign max priority to new transition
        max_prio = self.priorities.max() if (self.pos > 0 or self.full) else 1.0
        self.priorities[self.pos] = max_prio
        # increment
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or (self.pos == 0)

    def sample(self):
        N = len(self)
        if N < self.batch_size:
            raise ValueError("Not enough samples to draw from buffer")

        prios = self.priorities[:N] ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(N, self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        beta = self.beta_by_frame()
        self.frame += 1

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        x, a, r, x_next, done = zip(*samples)
        x       = torch.stack(x)
        x_next  = torch.stack(x_next)
        a       = torch.tensor(np.array(a)).float()
        r       = torch.tensor(r).float().unsqueeze(1)
        done    = torch.tensor(done).float().unsqueeze(1)
        weights = torch.tensor(weights).float().unsqueeze(1)

        return x, a, r, x_next, done, indices, weights

    def update_priorities(self, indices, td_errors):
        # td_errors is shape (batch_size, 1) → .flatten() gives a 1D array of scalars
        for idx, err in zip(indices, td_errors.detach().cpu().numpy().flatten()):
            self.priorities[idx] = abs(err) + self.eps


# ----------------------
# Ornstein-Uhlenbeck Noise
# ----------------------
class OrnsteinUhlenbeck:
    def __init__(self, action_space: gym.spaces.Box, theta: float = 0.15,
                 mu: float = 0.0, sigma: float = 0.2):
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
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        conv_out_size = 64 * 8 * 8
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU())
        self.fc_steer = nn.Linear(512, 1)
        self.fc_acc_brake = nn.Linear(512, 2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv(x).view(x.size(0), -1)
        f = self.fc(f)
        steer = self.tanh(self.fc_steer(f))
        acc_brake = self.sigmoid(self.fc_acc_brake(f))
        return torch.cat([steer, acc_brake], dim=1)

class CNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        conv_out_size = 64 * 8 * 8
        self.fc_state = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(512 + 3, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        f = self.conv(x).view(x.size(0), -1)
        sf = self.fc_state(f)
        return self.fc(torch.cat([sf, a], dim=1))

# ----------------------
# DDPG Agent
# ----------------------
class DDPG:
    def __init__(self, env, gamma: float = 0.99, tau: float = 1e-2,
                 capacity: int = 2**18, batch_size: int = 32):
        self.gamma = gamma
        self.tau = tau

        self.actor = CNNActor()
        self.critic = CNNCritic()
        self.actor_tar = CNNActor()
        self.critic_tar = CNNCritic()
        self.actor_tar.load_state_dict(self.actor.state_dict())
        self.critic_tar.load_state_dict(self.critic.state_dict())

        self.memory = PrioritizedReplayBuffer(capacity, batch_size)
        self.criterion = nn.MSELoss()
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=1e-3)

    def action(self, obs: np.ndarray) -> np.ndarray:
        t = preprocess(obs).unsqueeze(0)
        with torch.no_grad(): a = self.actor(t)
        return a.cpu().numpy()[0]

    def optimize(self):
        if len(self.memory) < self.memory.batch_size:
            return

        x, a, r, x_next, done, idxs, weights = self.memory.sample()

        # Critic
        q = self.critic(x, a)
        with torch.no_grad():
            a_next = self.actor_tar(x_next)
            q_next = self.critic_tar(x_next, a_next)
            target = r + (1 - done) * self.gamma * q_next

        td_err = q - target
        critic_loss = (weights * td_err.pow(2)).mean()
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        self.memory.update_priorities(idxs, td_err)

        # Actor
        actor_loss = -self.critic(x, self.actor(x)).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # Soft updates
        for pt, p in zip(self.actor_tar.parameters(), self.actor.parameters()):
            pt.data.copy_(p.data * self.tau + pt.data * (1 - self.tau))
        for pt, p in zip(self.critic_tar.parameters(), self.critic.parameters()):
            pt.data.copy_(p.data * self.tau + pt.data * (1 - self.tau))

# ----------------------
# Main Training Loop
# ----------------------
def main(render: bool = False, n_episodes: int = 5000):
    gym.logger.min_level = 40
    env = gym.make('CarRacing-v3', continuous=True)
    agent = DDPG(env)
    noise = OrnsteinUhlenbeck(env.action_space)
    rewards = []

    for ep in tqdm(range(n_episodes), desc="Training Episodes"):
        obs, _ = env.reset()
        noise.reset()
        ep_reward = 0

        for step in range(10000):
            if render: env.render()
            a = agent.action(obs)
            a = noise.action(a)
            obs_next, rew, term, trunc, _ = env.step(a)
            done = term or trunc

            agent.memory.push((
                preprocess(obs), a, rew,
                preprocess(obs_next), float(done)
            ))
            agent.optimize()

            obs = obs_next
            ep_reward += rew
            if done: break

        rewards.append(ep_reward)
        tqdm.write(f"Episode {ep+1}/{n_episodes}, Reward: {ep_reward:.2f}")

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DDPG w/ PER on CarRacing-v3')
    plt.savefig('ddpg_per_rewards.pdf')
    plt.show()
    env.close()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-render', action='store_true')
    p.add_argument('-episodes', type=int, default=5000)
    args = p.parse_args()
    main(render=args.render, n_episodes=args.episodes)
