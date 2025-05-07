import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse 
import yaml    
import os     
import time 

# --- YAML handling logic (EXACTLY AS PROVIDED BY USER) ---
def parse_cli_args():
    parser = argparse.ArgumentParser(description="DDPG agent for CarRacing with YAML config") # Description updated
    parser.add_argument("--config", type=str, default="cfg_ddpg.yaml", help="Path to YAML config file") # Default updated
    return parser.parse_args()

def load_yaml_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) # safe_load is generally preferred
        cfg.setdefault('save_model', False)
        print(f"Configuration loaded from {config_path}")
        return cfg
    except FileNotFoundError:
        print(f"FATAL: Configuration file '{config_path}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML file '{config_path}': {e}")
        exit(1)

# ----------------------
# Preprocessing Function
# ----------------------
def preprocess(obs: np.ndarray, normalize_factor: float) -> torch.Tensor:
    return torch.from_numpy(np.transpose(obs, (2, 0, 1))).float().div(normalize_factor)

# ----------------------
# Prioritized Replay Buffer
# ----------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, batch_size: int,
                 alpha: float, beta_start: float, beta_frames: int,
                 eps: float):
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
        self.frame = 1 # For beta annealing

    def __len__(self):
        return self.capacity if self.full else self.pos

    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))

    def push(self, transition):
        self.memory[self.pos] = transition
        max_prio = self.priorities.max() if (self.pos > 0 or self.full) else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or (self.pos == 0)

    def sample(self):
        N = len(self)
        if N < self.batch_size:
            return None # Indicate not enough samples

        prios = self.priorities[:N] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(N, self.batch_size, p=probs, replace=True)
        samples = [self.memory[idx] for idx in indices]

        beta = self.beta_by_frame()
        self.frame += 1 # Increment frame for beta annealing
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        x, a, r, x_next, done = zip(*samples)
        x_tensor = torch.stack(x)
        x_next_tensor = torch.stack(x_next)
        a_tensor = torch.tensor(np.array(a), dtype=torch.float32)
        r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        done_tensor = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        return x_tensor, a_tensor, r_tensor, x_next_tensor, done_tensor, indices, weights_tensor

    def update_priorities(self, indices, td_errors: torch.Tensor):
        # Ensure td_errors is a 1D numpy array for zipping
        td_errors_np = td_errors.squeeze().cpu().numpy()
        for idx, err in zip(indices, td_errors_np):
            self.priorities[idx] = abs(err) + self.eps

# ----------------------
# Ornstein-Uhlenbeck Noise
# ----------------------
class OrnsteinUhlenbeck:
    def __init__(self, action_space: gym.spaces.Box, mu: float, theta: float, sigma: float):
        self.action_dim = action_space.shape[0]
        self.mu = mu * np.ones(self.action_dim)
        self.theta = theta
        self.sigma = sigma
        self.low = action_space.low
        self.high = action_space.high
        self.noise = np.copy(self.mu) # Initialize to mu

    def reset(self):
        self.noise = np.copy(self.mu)

    def draw(self) -> np.ndarray:
        return np.random.standard_normal(self.action_dim)

    def get_action(self, base_action: np.ndarray) -> np.ndarray: # Renamed for clarity
        self.noise += self.theta * (self.mu - self.noise) + self.sigma * self.draw()
        return np.clip(base_action + self.noise, self.low, self.high)

# ----------------------
# CNN-based Actor and Critic (Fixed Architecture)
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
        f = self.conv(x).reshape(x.size(0), -1) # Flatten after conv
        f = self.fc(f)
        steer = self.tanh(self.fc_steer(f))
        acc_brake = self.sigmoid(self.fc_acc_brake(f))
        return torch.cat([steer, acc_brake], dim=1)

class CNNCritic(nn.Module): # Architecture as per original DDPG script
    def __init__(self, action_dim: int = 3): # Pass action_dim for flexibility
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        conv_out_size = 64 * 8 * 8
        self.fc_state = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU())
        self.fc_action_value = nn.Sequential(nn.Linear(512 + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        f = self.conv(x).reshape(x.size(0), -1) # Flatten after conv
        state_features = self.fc_state(f)
        return self.fc_action_value(torch.cat([state_features, a], dim=1))

# ----------------------
# DDPG Agent
# ----------------------
class DDPG:
    def __init__(self, action_space_shape: tuple, ddpg_cfg: dict, buffer_cfg: dict, device: torch.device):
        self.gamma = ddpg_cfg['gamma']
        self.tau = ddpg_cfg['tau']
        self.device = device
        action_dim = action_space_shape[0]

        self.actor = CNNActor().to(self.device)
        self.critic = CNNCritic(action_dim=action_dim).to(self.device) # Pass action_dim
        self.actor_target = CNNActor().to(self.device)
        self.critic_target = CNNCritic(action_dim=action_dim).to(self.device) # Pass action_dim

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.actor_target.parameters(): param.requires_grad = False
        for param in self.critic_target.parameters(): param.requires_grad = False

        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_cfg['capacity'], batch_size=buffer_cfg['batch_size'],
            alpha=buffer_cfg['alpha'], beta_start=buffer_cfg['beta_start'],
            beta_frames=buffer_cfg['beta_frames'], eps=buffer_cfg['eps']
        )
        self.criterion = nn.MSELoss() # Not weighted for PER, weights applied manually
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ddpg_cfg['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=ddpg_cfg['critic_lr'])
        self.batch_size = buffer_cfg['batch_size']

    def select_action(self, obs_tensor: torch.Tensor) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_tensor.unsqueeze(0).to(self.device))
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def store_transition(self, obs_t, action_np, reward_val, next_obs_t, done_val):
        self.memory.push((obs_t, action_np, reward_val, next_obs_t, float(done_val)))

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        
        sample_result = self.memory.sample()
        if sample_result is None: return
        
        states, actions, rewards, next_states, dones, indices, per_weights = sample_result
        states, actions, rewards, next_states, dones, per_weights = (
            t.to(self.device) for t in [states, actions, rewards, next_states, dones, per_weights]
        )

        # Critic update
        self.critic.train()
        current_q_values = self.critic(states, actions)
        with torch.no_grad():
            next_target_actions = self.actor_target(next_states)
            next_target_q_values = self.critic_target(next_states, next_target_actions)
            q_targets = rewards + (1.0 - dones) * self.gamma * next_target_q_values
        
        td_errors = q_targets - current_q_values
        critic_loss = (per_weights * td_errors.pow(2)).mean() # Weighted MSE for PER
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.memory.update_priorities(indices, td_errors.detach()) # Detach errors for PER update

        # Actor update
        self.actor.train()
        for p in self.critic.parameters(): p.requires_grad = False # Freeze critic
        actor_policy_actions = self.actor(states)
        q_values_for_actor = self.critic(states, actor_policy_actions)
        actor_loss = -q_values_for_actor.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.critic.parameters(): p.requires_grad = True # Unfreeze critic

        # Soft update target networks
        for target, local in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * local.data + (1.0 - self.tau) * target.data)
        for target, local in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * local.data + (1.0 - self.tau) * target.data)

# ----------------------
# Main Training Function
# ----------------------
def run_ddpg_training(cfg: dict):
    # Setup seeds and device
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        if cfg.get('torch_deterministic', True): # Default to True if not in config
             torch.backends.cudnn.deterministic = True
             torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    render_mode = "human" if cfg['render_during_training'] else None
    env = gym.make(cfg['env_id'], continuous=True, render_mode=render_mode)

    agent = DDPG(env.action_space.shape, cfg['ddpg_params'], cfg['replay_buffer_params'], device)
    noise = OrnsteinUhlenbeck(
        env.action_space,
        mu=cfg['ou_noise_params']['mu'],
        theta=cfg['ou_noise_params']['theta'],
        sigma=cfg['ou_noise_params']['sigma']
    )
    normalize_factor = cfg['preprocessing_params']['normalize_factor']
    episode_rewards_history = []
    run_signature = f"{cfg['env_id']}_DDPG_{cfg['seed']}_{int(time.time())}"

    for ep_idx in tqdm(range(cfg['n_episodes']), desc="Training DDPG"):
        obs_np, _ = env.reset(seed=cfg['seed'] + ep_idx) # Vary seed for stochastic resets
        obs_tensor = preprocess(obs_np, normalize_factor)
        noise.reset()
        total_episode_reward = 0.0

        for _step in range(cfg['max_steps_per_episode']):
            if cfg['render_during_training'] and ep_idx % 10 == 0 : # Example: render every 10 episode
                env.render()

            action_from_policy_np = agent.select_action(obs_tensor)
            action_with_noise_np = noise.get_action(action_from_policy_np)

            next_obs_np, reward, terminated, truncated, _ = env.step(action_with_noise_np)
            done = terminated or truncated
            next_obs_tensor = preprocess(next_obs_np, normalize_factor)

            agent.store_transition(obs_tensor, action_with_noise_np, reward, next_obs_tensor, done)
            agent.optimize()

            obs_tensor = next_obs_tensor
            total_episode_reward += reward
            if done: break
        
        episode_rewards_history.append(total_episode_reward)
        tqdm.write(f"Ep: {ep_idx+1}/{cfg['n_episodes']}, Reward: {total_episode_reward:.2f}")

    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards_history)
    plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.title(cfg['plotting_params']['reward_plot_title'])
    plt.savefig(cfg['plotting_params']['reward_plot_filename'])
    print(f"Reward plot saved: {cfg['plotting_params']['reward_plot_filename']}")
    # plt.show() # Can be blocking

    # Model Saving
    if cfg['save_model']:
        model_dir = "saved_models_ddpg" 
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{run_signature}_final.pt")
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'config_params': cfg # Save the config used for this run
        }, model_path)
        print(f"Trained model saved to: {model_path}")
    env.close()

if __name__ == '__main__':
    cli_args = parse_cli_args()
    config = load_yaml_config(cli_args.config) # Uses your exact load_yaml_config
    print("--- DDPG Training Configuration ---")
    import pprint
    pprint.pprint(config)
    print("---------------------------------")
    run_ddpg_training(config)