# Simplified PPO for CarRacing-v3 using YAML config, tqdm, saving, and video
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

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="PPO agent for CarRacing-v3 with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="cfg_ppo.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            # Ensure boolean values are parsed correctly
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded configuration from {config_path}")
        # Default values for new options if missing in file
        config.setdefault('save_model', False)
        config.setdefault('capture_video', False)
        config.setdefault('video_frequency', 50) # Default frequency if not set
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)

# --- Updated make_env to include video recording ---
def make_env(env_id, idx, gamma, capture_video, run_name, video_trigger):
    """Creates a function to generate an environment instance, potentially with video recording."""
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env) # Needed for episodic return tracking

        # --- Add video recording wrapper ---
        if capture_video and idx == 0:
            # Ensure the videos directory exists
            os.makedirs(f"videos/{run_name}", exist_ok=True)
            # Record video based on the trigger function
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=video_trigger, # Use the provided trigger
                disable_logger=True # Optional: disable verbose logging from RecordVideo
            )
        # --- End video recording addition ---

        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: obs / 255.0,
            observation_space=gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes weights and biases for a linear layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """PPO Agent with CNN feature extractor for image observations."""
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        cnn_input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(cnn_input_shape[0], 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape).permute(0, 3, 1, 2)
            flattened_size = self.cnn(dummy_input).shape[1]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(flattened_size, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(flattened_size, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(features)
        return action, log_prob, entropy, value

if __name__ == "__main__":
    cli_args = parse_args()
    config = load_config(cli_args.config)

    # --- Calculate derived parameters ---
    num_envs = config['num_envs']
    num_steps = config['num_steps']
    num_minibatches = config['num_minibatches']
    total_timesteps = config['total_timesteps']
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size

    # --- Create run name and directories ---
    run_name = f"{config['env_id']}__{config['seed']}__{int(time.time())}"
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True) # Create directory for saving models

    print(f"Running experiment: {run_name}")
    print(f"Saving models to: {run_dir}")
    if config['capture_video']:
        print(f"Saving videos to: videos/{run_name}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu'}")
    print(f"Hyperparameters: {config}")

    # Seeding
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config['torch_deterministic']

    device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

    # --- Video recording trigger ---
    # This function determines WHEN to record a video. Records every 'video_frequency' episodes.
    # Note: RecordVideo wrapper tracks episode counts internally.
    video_frequency = config['video_frequency']
    def video_trigger(episode_id):
         # episode_id is 0-indexed, record for episode 0 and then every video_frequency episodes
        return episode_id == 0 or (episode_id + 1) % video_frequency == 0

    # --- Environment setup with video trigger ---
    envs = gym.vector.SyncVectorEnv(
        [make_env(config['env_id'], i, config['gamma'], config['capture_video'], run_name, video_trigger) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'], eps=1e-5)

    # Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Training loop
    global_step = 0
    start_time = time.time()
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
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

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
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update phase
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

    # --- Add model saving logic ---
    if config['save_model']:
        # Ensure the directory exists (it should, but double-check)
        os.makedirs(run_dir, exist_ok=True)
        # Define the model path with .pt extension
        model_path = os.path.join(run_dir, f"{run_name}.pt")
        # Save the agent's state dictionary
        torch.save(agent.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
    # --- End model saving logic ---

    envs.close()
    print("\nTraining finished.")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
