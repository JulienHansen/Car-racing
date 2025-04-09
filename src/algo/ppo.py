import gymnasium as gym
import numpy as np
import cv2  # OpenCV for image preprocessing
import torch
from torch import nn, optim
from torch.distributions import Beta
from torch.utils.data import DataLoader, Dataset
from collections import deque
from time import sleep
from os.path import join


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Memory Dataset ----------------- #
class Memory(Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

# ------------- Convolutional Policy Network ------------- #
class ConvPolicyNetwork(nn.Module):
    def __init__(self, input_channels: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        # Convolutional feature extractor.
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 96, 96)  
            conv_out = self.conv(dummy)
            conv_size = conv_out.view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_size, hidden_dim),
            nn.ReLU(),
        )
        # Value head and policy heads (Beta parameters).
        self.value_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Softplus()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        value = self.value_head(x)
        alpha = self.alpha_head(x) + 1e-5  # Avoid zeros.
        beta = self.beta_head(x) + 1e-5
        return value, alpha, beta

# -------------- PPO Agent with Frame Stacking and Skipping -------------- #
class PPO:
    def __init__(
        self,
        env: gym.Env,
        net: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        horizon: int = 1024,
        epochs_per_step: int = 10,
        num_steps: int = 1000,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.001,
        save_dir: str = "ckpt",
        save_interval: int = 100,
        frame_skip: int = 1,       # How many frames to skip.
        stack_size: int = 4        # Number of frames to stack.
    ) -> None:
        self.env = env
        self.net = net.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.horizon = horizon
        self.epochs_per_step = epochs_per_step
        self.num_steps = num_steps
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.frame_skip = frame_skip
        self.stack_size = stack_size

        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)

        # Initialize state stack with preprocessed frames.
        obs, _ = env.reset()
        init_frame = self._preprocess(obs)
        self.state_stack = deque([init_frame.copy() for _ in range(stack_size)], maxlen=stack_size)
        self.state = self._get_state()

        self.alpha = 1.0  # For learning rate scheduling.

    def train(self):
        for step in range(self.num_steps):
            self._set_step_params(step)
            # Collect trajectory for the horizon.
            with torch.no_grad():
                memory = self.collect_trajectory(self.horizon)

            total_reward = memory.data[3].sum().item()  # rewards stored as 4th element.
            print(f"Step {step}: Total Reward: {total_reward:.2f}")

            memory_loader = DataLoader(memory, batch_size=self.batch_size, shuffle=True)

            avg_loss = 0.0
            for epoch in range(self.epochs_per_step):
                for (
                    states,
                    actions,
                    old_log_probs,
                    rewards,
                    advantages,
                    old_values
                ) in memory_loader:
                    loss, ploss, vloss, eloss = self.train_batch(
                        states, actions, old_log_probs, rewards, advantages, old_values
                    )
                    avg_loss += loss

            avg_loss /= (self.epochs_per_step * len(memory_loader))
            print(f"Step {step}: Loss: {avg_loss:.6f}")

            if step % self.save_interval == 0:
                self.save(join(self.save_dir, f"net_{step}.pth"))

        self.save(join(self.save_dir, "net_final.pth"))

    def train_batch(
        self,
        states: torch.Tensor,
        old_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
    ):
        self.optim.zero_grad()
        values, alpha, beta = self.net(states)
        values = values.squeeze(1)

        policy = Beta(alpha, beta)
        entropy = policy.entropy().mean()
        log_probs = policy.log_prob(old_actions).sum(dim=1)

        ratio = (log_probs - old_log_probs).exp()  # ratio between new and old policy.
        policy_loss_raw = ratio * advantages
        policy_loss_clip = ratio.clamp(1 - self.clip, 1 + self.clip) * advantages
        policy_loss = -torch.min(policy_loss_raw, policy_loss_clip).mean()

        with torch.no_grad():
            value_target = advantages + old_values

        value_loss = nn.MSELoss()(values, value_target)
        entropy_loss = -entropy

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        loss.backward()
        self.optim.step()

        return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()

    def collect_trajectory(self, num_steps: int) -> Memory:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        for _ in range(num_steps):
            # Evaluate the current policy.
            value, alpha, beta = self.net(self.state)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
            policy = Beta(alpha, beta)
            action = policy.sample()
            log_prob = policy.log_prob(action).sum()

            # Frame skipping: execute the same action for a number of frames, summing rewards.
            cumulative_reward = 0.0
            done = False
            for _ in range(self.frame_skip):
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                cumulative_reward += reward
                if done:
                    break
                # Update frame stack with the new observation.
                frame = self._preprocess(obs)
                self.state_stack.append(frame)
            if done:
                obs, _ = self.env.reset()
                frame = self._preprocess(obs)
                self.state_stack = deque([frame.copy() for _ in range(self.stack_size)], maxlen=self.stack_size)
            else:
                # If not done, update using the last observation.
                frame = self._preprocess(obs)
                self.state_stack.append(frame)
            next_state = self._get_state()

            # Store the transition.
            states.append(self.state)
            actions.append(action)
            rewards.append(cumulative_reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            self.state = next_state

            self.env.render()

        # Final state value (for GAE).
        final_value, _, _ = self.net(self.state)
        final_value = final_value.squeeze(0)

        advantages = self._compute_gae(rewards, values, dones, final_value)

        # Convert collected lists to tensors.
        states = torch.cat(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        values = torch.cat(values)

        return Memory(states, actions, log_probs, rewards, advantages, values)


    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = [0] * len(rewards)
        last_advantage = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (1 - dones[i]) * self.gamma * last_value - values[i]
            advantages[i] = delta + (1 - dones[i]) * self.gamma * self.gae_lambda * last_advantage
            last_value = values[i]
            last_advantage = advantages[i]
        return advantages

    def save(self, filepath: str):
        import os  # if not already imported
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.net.state_dict(), filepath)
        print(f"Model saved to {filepath}")


    def predict(self, state: np.ndarray):
        state = self._to_tensor(state)
        value, alpha, beta = self.net(state)
        return value, alpha, beta

    def _preprocess(self, obs):
        """
        Convert observation to grayscale, resize to 96x96, and normalize to [0,1].
        """
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (96, 96))
        obs = np.array(obs, dtype=np.float32) / 255.0
        # Add channel dimension.
        return obs[np.newaxis, ...]

    def _get_state(self):
        """
        Stack the frames in the state stack into a tensor for the network.
        Shape: [1, stack_size, H, W]
        """
        state = np.concatenate(list(self.state_stack), axis=0)
        return self._to_tensor(state)

    def _to_tensor(self, x):
        # Convert numpy array to tensor and add batch dimension.
        return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    def _set_step_params(self, step):
        # Linearly decay learning rate.
        self.alpha = 1.0 - step / self.num_steps
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr * self.alpha
        print(f"Learning Rate: {self.optim.param_groups[0]['lr']:.8f}")

# ------------------ Main Section ------------------ #
if __name__ == '__main__':
    # Create the CarRacing environment.
    env = gym.make("CarRacing-v3", continuous=True)
    # Wrap the environment to disable built-in rendering delays if needed.
    env.reset()

    # CarRacing outputs RGB images. After preprocessing to grayscale,
    # our stacked input has 4 channels.
    input_channels = 4
    num_actions = env.action_space.shape[0]  # Continuous action space (steering, accel, brake).

    # Instantiate the convolutional policy network.
    net = ConvPolicyNetwork(input_channels=input_channels, num_actions=num_actions, hidden_dim=128)

    # Create the PPO agent.
    ppo_agent = PPO(
        env,
        net,
        num_steps=1000,
        horizon=512,
        frame_skip=4,
        stack_size=4
    )

    try:
        ppo_agent.train()
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()

