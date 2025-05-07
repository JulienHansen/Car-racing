import os
import random
import time
import argparse
import yaml
import math
from tqdm import tqdm
from collections import deque 

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.beta import Beta
from gymnasium import ObservationWrapper, Wrapper


class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Check the original observation space (should be float after TransformObservation)
        obs_shape = self.observation_space.shape
        assert len(obs_shape) == 3 and obs_shape[-1] == 3, \
            f"ManualGrayScaleObservation expects RGB image (H, W, 3), got {obs_shape}"
        assert self.observation_space.dtype == np.float32, \
             f"ManualGrayScaleObservation expects float32 dtype, got {self.observation_space.dtype}"

        # Define the new observation space (grayscale)
        new_shape = obs_shape[:-1] + (1,) # Change last dim from 3 to 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )
        print(f"ManualGrayScaleObservation initialized. New observation space: {self.observation_space.shape}")

    def observation(self, obs):
        """Converts RGB observation to grayscale."""
        # Apply luminosity weights: 0.299*R + 0.587*G + 0.114*B
        # obs shape: (H, W, 3)
        r, g, b = obs[..., 0], obs[..., 1], obs[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        # Add channel dimension: (H, W) -> (H, W, 1)
        # Ensure correct dtype
        return np.expand_dims(gray, axis=-1).astype(np.float32)

# --- Custom Manual Frame Stacking Wrapper (remains the same) ---
class FrameStack(ObservationWrapper):
    """
    Manually stacks observations along the last dimension (channels).
    Assumes observations are numpy arrays with shape (H, W, C).
    """
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Check the base observation space shape (should be grayscale H, W, 1)
        assert len(self.observation_space.shape) == 3, \
            f"ManualFrameStack expects 3D observation space (H, W, C), got {self.observation_space.shape}"
        assert self.observation_space.shape[-1] == 1, \
            f"ManualFrameStack expects single channel input (e.g., grayscale), got {self.observation_space.shape[-1]} channels"

        # Modify the observation space shape
        low = self.observation_space.low.repeat(num_stack, axis=-1)
        high = self.observation_space.high.repeat(num_stack, axis=-1)
        new_shape = self.observation_space.shape[:-1] + (self.observation_space.shape[-1] * num_stack,)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=new_shape, dtype=self.observation_space.dtype
        )
        print(f"ManualFrameStack initialized. New observation space: {self.observation_space.shape}")


    def _stack_frames(self):
        assert len(self.frames) == self.num_stack, "Frame buffer not full!"
        return np.concatenate(list(self.frames), axis=-1)

    def observation(self, observation):
        return self._stack_frames()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation) # Add the single grayscale frame
        return self._stack_frames(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Overrides reset to clear and initialize the frame buffer."""
        observation, info = self.env.reset(**kwargs) # Gets single grayscale frame
        for _ in range(self.num_stack):
            self.frames.append(observation)
        return self._stack_frames(), info


def parse_args():
    parser = argparse.ArgumentParser(description="PPO agent for CarRacing-v3 with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="cfg_ppo.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded configuration from {config_path}")
        config.setdefault('save_model', False)
        config.setdefault('capture_video', False)
        config.setdefault('video_frequency', 50)
        config.setdefault('frame_stack', 1)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)

# --- Updated make_env to fix TransformObservation ---
def make_env(env_id, idx, gamma, capture_video, run_name, video_trigger, frame_stack):
    """Creates a function to generate an environment instance with wrappers."""
    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            os.makedirs(f"videos/{run_name}", exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=video_trigger,
                disable_logger=True
            )

        # Apply wrappers
        env = gym.wrappers.ClipAction(env)

        # --- Define the observation space AFTER normalization ---
        # The original space is Box(0, 255, (H, W, 3), uint8)
        # After dividing by 255.0, the space becomes Box(0.0, 1.0, (H, W, 3), float32)
        original_obs_space = env.observation_space
        normalized_obs_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=original_obs_space.shape,
            dtype=np.float32
        )
        # --- End define observation space ---

        # Normalize RGB pixel values first, providing the correct output space
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: obs / 255.0,
            observation_space = normalized_obs_space # Provide the new space
        )

        # Apply manual grayscale conversion
        # This MUST come before ManualFrameStack
        env = GrayScaleObservation(env) # This wrapper defines its own output space

        # Use the custom ManualFrameStack wrapper if frame_stack > 1
        if frame_stack > 1:
            env = FrameStack(env, num_stack=frame_stack) # This wrapper defines its own output space
        # If frame_stack is 1, the observation remains (H, W, 1) from grayscale

        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """PPO Agent with CNN feature extractor for image observations (handles frame stacking)."""
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        num_input_channels = obs_shape[-1]  # Will be frame_stack or 1
        print(f"Agent initialized with input shape: {obs_shape} -> CNN channels: {num_input_channels}")

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(num_input_channels, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *envs.single_observation_space.shape).permute(0, 3, 1, 2)
            flattened_size = self.cnn(dummy).shape[1]
            print(f"CNN flattened output size: {flattened_size}")

        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(flattened_size, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        # ===== Beta policy heads =====
        action_dim = int(np.prod(envs.single_action_space.shape))
        # α‐head
        self.actor_alpha = nn.Sequential(
            layer_init(nn.Linear(flattened_size, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
            nn.Softplus(),  # ensure α > 0
        )
        # β‐head
        self.actor_beta = nn.Sequential(
            layer_init(nn.Linear(flattened_size, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
            nn.Softplus(),  # ensure β > 0
        )

    def get_value(self, x):
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C*stack) -> (N, C*stack, H, W)
        features = self.cnn(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)

        # get α, β
        alpha = self.actor_alpha(features)
        beta  = self.actor_beta(features)
        dist  = Beta(alpha, beta)

        # sample u ∈ [0,1]
        u = dist.rsample() if self.training else dist.sample()

        # map to action space:
        # steering (u[:,0]) → [-1,1], others stay in [0,1]
        steering = u[:, 0] * 2.0 - 1.0
        others   = u[:, 1:]
        action   = torch.cat([steering.unsqueeze(-1), others], dim=-1)

        # log-prob + Jacobian for steering scale=2
        log_prob_u = dist.log_prob(u).sum(-1)
        log_prob   = log_prob_u - math.log(2.0)
        entropy    = dist.entropy().sum(-1)

        value = self.critic(features)
        return action, log_prob, entropy, value



if __name__ == "__main__":
    cli_args = parse_args()
    config = load_config(cli_args.config)

    # Calculate derived parameters
    num_envs = config['num_envs']
    num_steps = config['num_steps']
    num_minibatches = config['num_minibatches']
    total_timesteps = config['total_timesteps']
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size

    # Create run name and directories
    run_name = f"{config['env_id']}__{config['seed']}__{int(time.time())}"
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running experiment: {run_name}")
    print(f"Saving models to: {run_dir}")
    if config['capture_video']:
        print(f"Saving videos to: videos/{run_name}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu'}")
    print(f"Manual Frame stacking: {config['frame_stack']}")
    print(f"Using Manual Grayscale Conversion") # Indicate manual grayscale
    print(f"Hyperparameters: {config}")

    # Seeding
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config['torch_deterministic']

    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

    # Video recording trigger
    video_frequency = config['video_frequency']
    def video_trigger(episode_id):
        return episode_id == 0 or (episode_id + 1) % video_frequency == 0

    # Environment setup using the updated make_env
    envs = gym.vector.SyncVectorEnv(
        [make_env(config['env_id'], i, config['gamma'], config['capture_video'], run_name, video_trigger, config['frame_stack']) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action space is supported"
    print("Wrapped environment observation space:", envs.single_observation_space.shape)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'], eps=1e-5)

    # Storage setup - uses the correct shape from the wrapped envs
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Training loop
    global_step = 0
    start_time = time.time()
    # Initial reset returns the first stacked observation
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    print(f"Starting training for {total_timesteps} timesteps ({num_iterations} iterations)...")

    for iteration in tqdm(range(1, num_iterations + 1), desc="Training Progress"):
        # Learning rate annealing
        if config['anneal_lr']:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * config['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout phase
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs # Store the stacked observation
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # env.step now returns the *next* stacked observation via the wrapper
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos.get("final_info", []):
                    if info and "episode" in info:
                        tqdm.write(f"  global_step={global_step}, episodic_return={info['episode']['r']:.2f}")

        # Bootstrap value and compute advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1) # Use next stacked obs
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            gamma = config['gamma']
            gae_lambda = config['gae_lambda']
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch for training
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) # Shape includes stack dim
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update phase (remains largely the same)
        b_inds = np.arange(batch_size)
        update_epochs = config['update_epochs']
        clip_coef = config['clip_coef']
        norm_adv = config['norm_adv']
        clip_vloss = config['clip_vloss']
        ent_coef = config['ent_coef']
        vf_coef = config['vf_coef']
        max_grad_norm = config['max_grad_norm']
        target_kl = config['target_kl']
        pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Agent processes the stacked observations
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kls.append(approx_kl.item())

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_losses.append(pg_loss.item())

                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                v_losses.append(v_loss.item())

                entropy_loss = entropy.mean()
                entropy_losses.append(entropy_loss.item())

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and np.mean(approx_kls[-len(b_inds)//minibatch_size:]) > target_kl:
                tqdm.write(f"  Early stopping at epoch {epoch+1} due to reaching max KL: {np.mean(approx_kls[-len(b_inds)//minibatch_size:]):.4f}")
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # Model saving logic
    if config['save_model']:
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(run_dir, f"{run_name}.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")

    envs.close()
    print("\nTraining finished.")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")