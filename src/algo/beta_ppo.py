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
from gymnasium import ObservationWrapper

# --- Environment Wrappers ---

class GrayScaleObservation(ObservationWrapper):
    """Converts RGB image observations to grayscale."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        if not (len(obs_shape) == 3 and obs_shape[-1] == 3 and self.observation_space.dtype == np.float32):
            raise ValueError(
                f"GrayScaleObservation expects RGB float32 image (H, W, 3), got {obs_shape} with dtype {self.observation_space.dtype}"
            )
        new_shape = obs_shape[:-1] + (1,)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Luminosity technique
        r, g, b = obs[..., 0], obs[..., 1], obs[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.expand_dims(gray, axis=-1).astype(np.float32)

class FrameStack(ObservationWrapper):
    """Manually stacks x consecutive frames ."""
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        obs_shape = self.observation_space.shape
        low = np.repeat(self.observation_space.low, num_stack, axis=-1)
        high = np.repeat(self.observation_space.high, num_stack, axis=-1)
        new_shape = obs_shape[:-1] + (obs_shape[-1] * num_stack,)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=new_shape, dtype=self.observation_space.dtype)

    def _get_stacked_frames(self) -> np.ndarray:
        assert len(self.frames) == self.num_stack, "Frame buffer not full for stacking."
        return np.concatenate(list(self.frames), axis=-1)

    def observation(self, observation: np.ndarray) -> np.ndarray: # observation arg not used, uses internal buffer
        return self._get_stacked_frames()

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(next_obs) # Appends the new (single, unstacked) frame
        return self._get_stacked_frames(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        initial_obs, info = self.env.reset(**kwargs) # Gets a single (unstacked) frame
        for _ in range(self.num_stack):
            self.frames.append(initial_obs)
        return self._get_stacked_frames(), info

# YAML handling logic
def parse_cli_args():
    parser = argparse.ArgumentParser(description="PPO agent for CarRacing with YAML config")
    parser.add_argument("--config", type=str, default="cfg_ppo.yaml", help="Path to YAML config file")
    return parser.parse_args()

def load_yaml_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) # safe_load is generally preferred
        cfg.setdefault('save_model', False)
        cfg.setdefault('capture_video', False)
        cfg.setdefault('video_frequency', 50)
        cfg.setdefault('frame_stack', 1) # Default to no frame stacking if not specified
        print(f"Configuration loaded from {config_path}")
        return cfg
    except FileNotFoundError:
        print(f"FATAL: Configuration file '{config_path}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML file '{config_path}': {e}")
        exit(1)

def create_env_factory(env_id, env_idx, cfg, run_name, video_rec_trigger, frame_stack_count):
    """Returns a function that creates and wraps an environment instance."""
    def _init_env():
        # Only record video for the first environment if enabled
        is_video_env = cfg['capture_video'] and env_idx == 0
        render_mode = "rgb_array" if is_video_env else None
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if is_video_env:
            video_path = f"videos/{run_name}"
            os.makedirs(video_path, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=video_rec_trigger, disable_logger=True)

        env = gym.wrappers.ClipAction(env)
        # Normalize observations to [0, 1]
        normalized_obs_space = gym.spaces.Box(0.0, 1.0, env.observation_space.shape, np.float32)
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / 255.0, observation_space=normalized_obs_space)
        env = GrayScaleObservation(env) # Grayscale conversion
        if frame_stack_count > 1: # Apply frame stacking if needed
            env = FrameStack(env, num_stack=frame_stack_count)
        env = gym.wrappers.NormalizeReward(env, gamma=cfg['gamma'])
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10)) # Clip rewards
        return env
    return _init_env

def init_layer_weights(layer, std_dev=np.sqrt(2), bias_fill=0.0):
    """Initializes network layer weights orthogonally and biases to a constant."""
    torch.nn.init.orthogonal_(layer.weight, std_dev)
    torch.nn.init.constant_(layer.bias, bias_fill)
    return layer

# --- PPO Agent ---

class PPOAgent(nn.Module):
    def __init__(self, vectorized_envs):
        super().__init__()
        obs_space = vectorized_envs.single_observation_space
        action_space = vectorized_envs.single_action_space
        num_channels = obs_space.shape[-1] # Accounts for frame stacking

        self.feature_extractor = nn.Sequential(
            init_layer_weights(nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)), nn.ReLU(),
            init_layer_weights(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            init_layer_weights(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate flattened size dynamically
        with torch.no_grad():
            # (N, H, W, C) -> (N, C, H, W) for PyTorch CNN
            dummy_obs = torch.zeros(1, *obs_space.shape).permute(0, 3, 1, 2)
            cnn_output_size = self.feature_extractor(dummy_obs).shape[1]

        self.critic_head = nn.Sequential(
            init_layer_weights(nn.Linear(cnn_output_size, 512)), nn.ReLU(),
            init_layer_weights(nn.Linear(512, 1), std_dev=1.0),
        )
        action_dim = int(np.prod(action_space.shape))
        self.actor_alpha_head = nn.Sequential( # For Beta distribution's alpha parameter
            init_layer_weights(nn.Linear(cnn_output_size, 512)), nn.ReLU(),
            init_layer_weights(nn.Linear(512, action_dim), std_dev=0.01), nn.Softplus(),
        )
        self.actor_beta_head = nn.Sequential( # For Beta distribution's beta parameter
            init_layer_weights(nn.Linear(cnn_output_size, 512)), nn.ReLU(),
            init_layer_weights(nn.Linear(512, action_dim), std_dev=0.01), nn.Softplus(),
        )

    def _get_features(self, obs_batch: torch.Tensor) -> torch.Tensor:
        # (N, H, W, C) -> (N, C, H, W)
        return self.feature_extractor(obs_batch.permute(0, 3, 1, 2))

    def get_value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._get_features(obs_batch))

    def get_action_dist_params(self, obs_batch: torch.Tensor):
        features = self._get_features(obs_batch)
        alpha = self.actor_alpha_head(features)
        beta = self.actor_beta_head(features)
        return alpha, beta

    def get_action_and_value(self, obs_batch: torch.Tensor, chosen_action: torch.Tensor = None):
        alpha, beta = self.get_action_dist_params(obs_batch)
        dist = Beta(alpha, beta) # Beta distribution for actions in [0,1]

        if chosen_action is None:
            # Sample u from Beta distribution (in [0,1])
            u_sampled = dist.rsample() if self.training else dist.sample()
        else:
            # If action is provided, we need to map it back to u in [0,1] to get log_prob
            # Assuming chosen_action is in the true action space (e.g., steering [-1,1])
            u_chosen_steering = (chosen_action[:, 0] + 1.0) / 2.0 # Map steering from [-1,1] to [0,1]
            u_chosen_others = chosen_action[:, 1:]
            u_sampled = torch.cat([u_chosen_steering.unsqueeze(-1), u_chosen_others], dim=-1)

        action_steering = u_sampled[:, 0] * 2.0 - 1.0 # needs to be shift to be in [-1 , 1]
        action_others = u_sampled[:, 1:]
        final_action = torch.cat([action_steering.unsqueeze(-1), action_others], dim=-1)

        log_prob_u = dist.log_prob(u_sampled).sum(axis=-1)
        log_prob_action = log_prob_u - math.log(2.0)
        value = self.critic_head(self._get_features(obs_batch)) 
        return final_action, log_prob_action, dist.entropy().sum(axis=-1), value


def evaluate(agent, env_eval, device):
    agent.eval()

    obs, _ = env_eval.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    dones = np.zeros(env_eval.num_envs, dtype=bool)
    
    episode_rewards = np.zeros(env_eval.num_envs)

    while not np.any(dones):
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs)
        next_obs, rewards, terminations, truncations, infos = env_eval.step(actions.cpu().numpy())
        dones = np.logical_or(terminations, truncations)
        
        episode_rewards[~dones] += rewards[~dones]

    return episode_rewards.mean(), episode_rewards.std()


# --- Main Training Script ---

if __name__ == "__main__":
    args = parse_cli_args()
    cfg = load_yaml_config(args.config)

    # Training setup
    num_envs = cfg['num_envs']
    num_envs_eval = cfg['eval_envs']
    steps_per_rollout = cfg['num_steps'] # Renamed for clarity
    total_timesteps = cfg['total_timesteps']
    minibatches_per_epoch = cfg['num_minibatches']
    anneal_lr = cfg['anneal_lr']

    batch_size = num_envs * steps_per_rollout
    minibatch_size = batch_size // minibatches_per_epoch
    num_training_iterations = total_timesteps // batch_size

    eval_timesteps = []
    rewards_mean = []
    rewards_std = []
    max_reward = None

    # Experiment naming and paths
    timestamp = int(time.time())
    run_name = f"{cfg['env_id']}_PPO__{cfg['seed']}__{timestamp}"
    model_save_dir = f"runs/{run_name}"
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"--- Experiment: {run_name} ---")
    print(f"Device: {'cuda' if torch.cuda.is_available() and cfg['cuda'] else 'cpu'}")
    print(f"Frame stacking: {cfg['frame_stack']}")
    # print(f"Full config: {cfg}") # Can be very verbose

    # Reproducibility
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = cfg['torch_deterministic']
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['cuda'] else "cpu")

    # Video recording setup
    video_rec_trigger = lambda ep_idx: ep_idx == 0 or (ep_idx + 1) % cfg['video_frequency'] == 0

    # Create vectorized environments
    env_factories = [
        create_env_factory(cfg['env_id'], i, cfg, run_name, video_rec_trigger, cfg['frame_stack'])
        for i in range(num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(env_factories)

    envs_eval = gym.vector.SyncVectorEnv([
            create_env_factory(cfg['env_id'], i, cfg, run_name + "_eval", None , cfg['frame_stack'])
            for i in range(num_envs_eval)
        ])

    assert isinstance(envs.single_action_space, gym.spaces.Box), "Continuous actions required."
    print(f"Observation space: {envs.single_observation_space.shape}, Action space: {envs.single_action_space.shape}")


    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg['learning_rate'], eps=1e-5)

    # Data storage for rollouts
    obs_buffer = torch.zeros((steps_per_rollout, num_envs) + envs.single_observation_space.shape).to(device)
    action_buffer = torch.zeros((steps_per_rollout, num_envs) + envs.single_action_space.shape).to(device)
    logprob_buffer = torch.zeros((steps_per_rollout, num_envs)).to(device)
    reward_buffer = torch.zeros((steps_per_rollout, num_envs)).to(device)
    done_buffer = torch.zeros((steps_per_rollout, num_envs)).to(device)
    value_buffer = torch.zeros((steps_per_rollout, num_envs)).to(device)

    # Training loop initialization
    current_global_step = 0
    training_start_time = time.time()
    next_obs_batch, _ = envs.reset(seed=cfg['seed']) # _ for info
    next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(device)
    next_done_batch = torch.zeros(num_envs).to(device)

    print(f"Starting training for {total_timesteps} timesteps ({num_training_iterations} iterations).")

    for iteration in tqdm(range(1, num_training_iterations + 1), desc="PPO Iterations"):
        if anneal_lr:
            current_lr_frac = 1.0 - (iteration - 1.0) / num_training_iterations
            new_lr = current_lr_frac * cfg['learning_rate']
            optimizer.param_groups[0]["lr"] = new_lr

        # --- Collect experiences (Rollout Phase) ---
        agent.eval() # Set agent to evaluation mode for rollout
        for step in range(steps_per_rollout):
            current_global_step += num_envs
            obs_buffer[step] = next_obs_batch
            done_buffer[step] = next_done_batch

            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(next_obs_batch)
                value_buffer[step] = values.flatten()
            action_buffer[step] = actions
            logprob_buffer[step] = log_probs

            next_obs_batch_np, rewards_np, terminated_np, truncated_np, infos = envs.step(actions.cpu().numpy())
            next_done_batch_np = np.logical_or(terminated_np, truncated_np)

            reward_buffer[step] = torch.tensor(rewards_np).to(device).view(-1)
            next_obs_batch = torch.tensor(next_obs_batch_np, dtype=torch.float32).to(device)
            next_done_batch = torch.tensor(next_done_batch_np, dtype=torch.float32).to(device)

            for item in infos.get("final_info", []): # "final_info" for vectorized envs
                if item and "episode" in item:
                    tqdm.write(f"  step={current_global_step}, ep_return={item['episode']['r']:.2f}, ep_length={item['episode']['l']}")


        # --- Calculate Advantages and Returns (GAE) ---
        agent.eval() # Ensure agent is in eval mode for value prediction
        with torch.no_grad():
            next_values = agent.get_value(next_obs_batch).reshape(1, -1)
            advantages = torch.zeros_like(reward_buffer).to(device)
            last_gae_lambda = 0
            for t in reversed(range(steps_per_rollout)):
                if t == steps_per_rollout - 1:
                    next_not_done = 1.0 - next_done_batch
                    next_step_values = next_values
                else:
                    next_not_done = 1.0 - done_buffer[t + 1]
                    next_step_values = value_buffer[t + 1]
                delta = reward_buffer[t] + cfg['gamma'] * next_step_values * next_not_done - value_buffer[t]
                advantages[t] = last_gae_lambda = delta + cfg['gamma'] * cfg['gae_lambda'] * next_not_done * last_gae_lambda
            returns = advantages + value_buffer

        # Flatten rollout data for batch processing
        b_obs = obs_buffer.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprob_buffer.reshape(-1)
        b_actions = action_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buffer.reshape(-1) # For Value loss clipping

        # --- PPO Update Phase ---
        agent.train() # Set agent to training mode
        clip_fracs_epoch = [] # For tracking clipping, if desired
        for epoch in range(cfg['update_epochs']):
            batch_indices = np.arange(batch_size)
            np.random.shuffle(batch_indices)
            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = start_idx + minibatch_size
                mb_indices = batch_indices[start_idx:end_idx]

                _, new_logprobs, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_indices], b_actions[mb_indices]
                )
                log_ratio = new_logprobs - b_logprobs[mb_indices]
                ratio = torch.exp(log_ratio)

                # For KL divergence tracking (optional)
                with torch.no_grad():
                    # approx_kl = ((ratio - 1) - log_ratio).mean() # Not used by default in this version
                    clipped = ratio.gt(1 + cfg['clip_coef']) | ratio.lt(1 - cfg['clip_coef'])
                    clip_fracs_epoch.append(clipped.float().mean().item())


                mb_adv = b_advantages[mb_indices]
                if cfg['norm_adv']: # Advantage normalization
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy (Actor) Loss - Clipped Surrogate Objective
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg['clip_coef'], 1 + cfg['clip_coef'])
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value (Critic) Loss
                new_values = new_values.view(-1)
                if cfg['clip_vloss']:
                    v_loss_unclipped = (new_values - b_returns[mb_indices]) ** 2
                    v_clipped = b_values[mb_indices] + torch.clamp(
                        new_values - b_values[mb_indices], -cfg['clip_coef'], cfg['clip_coef']
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_values - b_returns[mb_indices]) ** 2).mean()

                entropy_loss = entropy.mean()
                total_loss = policy_loss - cfg['ent_coef'] * entropy_loss + cfg['vf_coef'] * value_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg['max_grad_norm'])
                optimizer.step()

        if iteration % cfg.get("eval_frequency", 5000) == 0:
            mean_reward, std_reward = evaluate(agent, envs_eval, device)

            if max_reward is None or mean_reward > max_reward:
                tqdm.write(f"New best model saved : {mean_reward}")
                model_path = os.path.join(model_save_dir, f"model_best.pt")
                torch.save(agent.state_dict(), model_path)
                max_reward = mean_reward

            eval_timesteps.append(iteration)
            rewards_mean.append(mean_reward)
            rewards_std.append(std_reward)

            tqdm.write(f"[Eval @ iter {iteration}] Avg return: {mean_reward:.2f} +-{std_reward:.2f}")

    # --- Save Model and Cleanup ---
    if cfg['save_model']:
        model_path = os.path.join(model_save_dir, f"model_final.pt")
        torch.save(agent.state_dict(), model_path)

        eval_path = os.path.join(model_save_dir, f"model_eval")

        print(rewards_mean, rewards_std)
        np.savez(eval_path, timesteps=np.array(eval_timesteps), 
                 mean_rewards=np.array(rewards_mean), 
                 std_rewards=np.array(rewards_std),
                )
        print(f"\nModel saved to {model_path}")

    envs.close()
    total_training_duration = (time.time() - training_start_time) / 60
    print("\n--- Training Complete ---")
    print(f"Total duration: {total_training_duration:.2f} minutes")