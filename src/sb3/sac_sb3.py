import os
import gymnasium
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Configuration
env_str = "CarRacing-v3"
log_dir = os.path.join("logs", env_str)
video_dir = os.path.join("videos", env_str)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
gray_scale = True  # Convert observations to grayscale

# Use WarpFrame wrapper if gray_scale is enabled
wrapper_class = WarpFrame if gray_scale else None

# Create Training Environment
train_env = make_vec_env(env_str, n_envs=1, wrapper_class=wrapper_class)
train_env = VecFrameStack(train_env, n_stack=4)
train_env = VecTransposeImage(train_env)

# Create Evaluation Environment
eval_env = make_vec_env(env_str, n_envs=1, wrapper_class=wrapper_class)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env)

# Custom callback to record evaluation rewards during training
class RewardEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes=5, verbose=1):
        super(RewardEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timesteps = []
        self.mean_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env,
                                             n_eval_episodes=self.n_eval_episodes,
                                             render=False)
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(mean_reward)
            if self.verbose:
                print(f"Evaluation at timestep {self.num_timesteps}: mean reward = {mean_reward:.2f}")
        return True

# Initialize SAC model
model = SAC("CnnPolicy", train_env, verbose=1)

# Setup evaluation callback (adjust eval_freq as needed)
eval_callback = RewardEvalCallback(eval_env, eval_freq=2500, n_eval_episodes=5)

# Train the model
total_timesteps = 750000
model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)

# Save the model
model_path = os.path.join(log_dir, "sac_car_racing")
model.save(model_path)

# Evaluate the final model
final_mean_reward, final_std_reward = evaluate_policy(model, train_env, n_eval_episodes=20)
print(f"Final Model - Mean reward: {final_mean_reward:.2f} +/- {final_std_reward:.2f}")

# Record video of the trained model
video_env = make_vec_env(env_str, n_envs=1, seed=0, wrapper_class=wrapper_class)
video_env = VecFrameStack(video_env, n_stack=4)
video_env = VecTransposeImage(video_env)
video_env = VecVideoRecorder(video_env, video_dir,
                             video_length=10000,
                             record_video_trigger=lambda x: x == 0,
                             name_prefix="sac_car_racing")
obs = video_env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, info = video_env.step(action)
    video_env.render()
    if done:
        break
video_env.close()

# Plot evaluation results
plt.figure()
plt.plot(eval_callback.timesteps, eval_callback.mean_rewards, marker='o')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title(f"SAC Performance on {env_str}")
plt.grid()
plt.show()

# Close environments
train_env.close()
eval_env.close()
