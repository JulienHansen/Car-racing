import os
import random
import time
import argparse
import yaml
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

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

def create_env_factory(env_id, env_idx, cfg, run_name, video_rec_trigger):
    """Returns a function that creates and appropriately wraps an environment instance."""
    def _init_env():
        is_video_env = cfg['capture_video'] and env_idx == 0
        render_mode = "rgb_array" if is_video_env else None
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if is_video_env:
            video_path = f"videos/{run_name}"
            os.makedirs(video_path, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=video_rec_trigger, disable_logger=True)

        env = gym.wrappers.ClipAction(env)
        # Normalize observations to [0, 1] and define the new observation space
        normalized_obs_space = gym.spaces.Box(0.0, 1.0, env.observation_space.shape, np.float32)
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / 255.0, observation_space=normalized_obs_space)
        env = gym.wrappers.NormalizeReward(env, gamma=cfg['gamma'])
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10)) # Clip rewards
        return env
    return _init_env

def init_layer_weights(layer, std_dev=np.sqrt(2), bias_const=0.0):
    """Initializes layer weights orthogonally and biases to a constant."""
    torch.nn.init.orthogonal_(layer.weight, std_dev)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    """PPO Agent: CNN feature extractor + Actor (Normal dist) & Critic heads."""
    def __init__(self, vectorized_envs):
        super().__init__()
        obs_space = vectorized_envs.single_observation_space
        action_space = vectorized_envs.single_action_space
        # (H, W, C) -> C for Conv2d input_channels
        num_channels = obs_space.shape[2]

        self.feature_extractor = nn.Sequential(
            init_layer_weights(nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)), nn.ReLU(),
            init_layer_weights(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            init_layer_weights(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        # Determine CNN output size for MLP heads
        with torch.no_grad():
            # (N,H,W,C) -> (N,C,H,W) for PyTorch
            dummy_obs = torch.zeros(1, *obs_space.shape).permute(0, 3, 1, 2)
            cnn_output_size = self.feature_extractor(dummy_obs).shape[1]

        self.critic_head = nn.Sequential(
            init_layer_weights(nn.Linear(cnn_output_size, 512)), nn.ReLU(),
            init_layer_weights(nn.Linear(512, 1), std_dev=1.0),
        )
        self.actor_mean_head = nn.Sequential(
            init_layer_weights(nn.Linear(cnn_output_size, 512)), nn.ReLU(),
            init_layer_weights(nn.Linear(512, np.prod(action_space.shape)), std_dev=0.01),
        )
        self.actor_log_std_param = nn.Parameter(torch.zeros(1, np.prod(action_space.shape)))

    def _get_features(self, obs_batch: torch.Tensor) -> torch.Tensor:
        # Permute (N, H, W, C) to (N, C, H, W) for Conv2D
        return self.feature_extractor(obs_batch.permute(0, 3, 1, 2))

    def get_value(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self._get_features(obs_batch))

    def get_action_and_value(self, obs_batch: torch.Tensor, chosen_action: torch.Tensor = None):
        features = self._get_features(obs_batch)
        action_mean = self.actor_mean_head(features)
        action_log_std = self.actor_log_std_param.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        dist = Normal(action_mean, action_std)

        if chosen_action is None:
            chosen_action = dist.sample()

        log_prob = dist.log_prob(chosen_action).sum(axis=1) # Sum over action dimensions
        entropy = dist.entropy().sum(axis=1)      # Sum over action dimensions
        value = self.critic_head(features)         # Or self.get_value(obs_batch) if features are not passed
        return chosen_action, log_prob, entropy, value
    

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


if __name__ == "__main__":
    args = parse_cli_args()
    cfg = load_yaml_config(args.config)

    # Derived training parameters
    num_envs = cfg['num_envs']
    num_envs_eval = cfg['eval_envs']
    steps_per_rollout = cfg['num_steps']
    total_timesteps = cfg['total_timesteps']
    minibatches_per_epoch = cfg['num_minibatches']

    batch_size = num_envs * steps_per_rollout
    minibatch_size = batch_size // minibatches_per_epoch
    num_train_iterations = total_timesteps // batch_size

    eval_timesteps = []
    rewards_mean = []
    rewards_std = []
    max_reward = None

    # Setup run
    run_name = f"{cfg['env_id']}_PPO-Normal_{cfg['seed']}_{int(time.time())}"
    model_save_dir = f"runs/{run_name}"
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"--- Experiment: {run_name} ---")
    print(f"Device: {'cuda' if torch.cuda.is_available() and cfg['cuda'] else 'cpu'}")
    # print(f"Full config: {cfg}") # Uncomment for verbose config details

    # Reproducibility
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = cfg['torch_deterministic']
    device = torch.device("cuda" if torch.cuda.is_available() and cfg['cuda'] else "cpu")

    # Video recording trigger (records first ep and then every 'video_frequency' episodes)
    video_rec_trigger = lambda ep_idx: ep_idx == 0 or (ep_idx + 1) % cfg['video_frequency'] == 0

    env_factories = [
        create_env_factory(cfg['env_id'], i, cfg, run_name, None)
        for i in range(num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(env_factories)

    envs_eval = gym.vector.SyncVectorEnv([
            create_env_factory(cfg['env_id'], i, cfg, run_name + "_eval", None)
            for i in range(num_envs_eval)
        ])

    assert isinstance(envs.single_action_space, gym.spaces.Box), "Continuous actions expected."

    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg['learning_rate'], eps=1e-5)

    # Rollout data storage
    obs_buf = torch.zeros((steps_per_rollout, num_envs) + envs.single_observation_space.shape, device=device)
    action_buf = torch.zeros((steps_per_rollout, num_envs) + envs.single_action_space.shape, device=device)
    logprob_buf = torch.zeros((steps_per_rollout, num_envs), device=device)
    reward_buf = torch.zeros((steps_per_rollout, num_envs), device=device)
    done_buf = torch.zeros((steps_per_rollout, num_envs), device=device)
    value_buf = torch.zeros((steps_per_rollout, num_envs), device=device)

    # Training loop init
    global_step_count = 0
    training_start_time = time.time()
    next_obs_tensor, _ = envs.reset(seed=cfg['seed'])
    next_obs_tensor = torch.tensor(next_obs_tensor, dtype=torch.float32).to(device)
    next_done_tensor = torch.zeros(num_envs, device=device)

    print(f"Training for {total_timesteps} timesteps ({num_train_iterations} iterations)...")

    for iteration in tqdm(range(1, num_train_iterations + 1), desc="PPO Iteration"):
        if cfg['anneal_lr']:
            lr_frac = 1.0 - (iteration - 1.0) / num_train_iterations
            optimizer.param_groups[0]["lr"] = lr_frac * cfg['learning_rate']

        # --- Collect Rollout ---
        agent.eval()
        for step in range(steps_per_rollout):
            global_step_count += num_envs
            obs_buf[step] = next_obs_tensor
            done_buf[step] = next_done_tensor

            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(next_obs_tensor)
                value_buf[step] = values.flatten()
            action_buf[step] = actions
            logprob_buf[step] = log_probs

            next_obs_np, rewards_np, terminated_np, truncated_np, infos = envs.step(actions.cpu().numpy())
            next_done_np = np.logical_or(terminated_np, truncated_np)

            reward_buf[step] = torch.tensor(rewards_np, device=device).view(-1)
            next_obs_tensor = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done_tensor = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            for item in infos.get("final_info", []):
                if item and "episode" in item: # Check if episode stats are available
                    tqdm.write(f"  step={global_step_count}, ep_return={item['episode']['r']:.2f}")

        # --- Compute Advantages (GAE) & Returns ---
        agent.eval()
        with torch.no_grad():
            next_values_pred = agent.get_value(next_obs_tensor).reshape(1, -1)
            advantages = torch.zeros_like(reward_buf, device=device)
            last_gae_lambda = 0
            for t in reversed(range(steps_per_rollout)):
                next_not_done = 1.0 - (next_done_tensor if t == steps_per_rollout - 1 else done_buf[t + 1])
                current_values = value_buf[t]
                next_step_values = next_values_pred if t == steps_per_rollout - 1 else value_buf[t+1]

                delta = reward_buf[t] + cfg['gamma'] * next_step_values * next_not_done - current_values
                advantages[t] = last_gae_lambda = delta + cfg['gamma'] * cfg['gae_lambda'] * next_not_done * last_gae_lambda
            returns = advantages + value_buf

        # Flatten batch data
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = (
            x.reshape((-1,) + x.shape[2:]) if x.ndim > 2 else x.reshape(-1)
            for x in (obs_buf, logprob_buf, action_buf, advantages, returns, value_buf)
        )

        # --- PPO Update Epochs ---
        agent.train()
        batch_indices = np.arange(batch_size)
        for epoch in range(cfg['update_epochs']):
            np.random.shuffle(batch_indices)
            for start_idx in range(0, batch_size, minibatch_size):
                mb_indices = batch_indices[start_idx : start_idx + minibatch_size]

                _, new_logprobs, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_indices], b_actions[mb_indices]
                )
                log_ratio = new_logprobs - b_logprobs[mb_indices]
                ratio = torch.exp(log_ratio)

                mb_adv = b_advantages[mb_indices]
                if cfg['norm_adv']:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg['clip_coef'], 1 + cfg['clip_coef'])
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
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

                entropy_bonus = entropy.mean()
                total_loss = policy_loss - cfg['ent_coef'] * entropy_bonus + cfg['vf_coef'] * value_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg['max_grad_norm'])
                optimizer.step()

            # Optional: KL-divergence early stopping (more involved to track approx_kl per minibatch accurately)
            # if cfg.get('target_kl') and np.mean(approx_kls_epoch) > cfg['target_kl']: break

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

    # --- Save final model & cleanup ---
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
    print(f"\n--- Training Finished ---\nTotal time: {(time.time() - training_start_time)/60:.2f} minutes")