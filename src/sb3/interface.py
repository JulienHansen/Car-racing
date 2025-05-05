import os
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime

from pathlib import Path

from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import constant as cst

os.makedirs(cst.LOG_DIR, exist_ok=True)
os.makedirs(cst.VIDEO_DIR, exist_ok=True)
gray_scale = True  

# Testing grayscale conversion to see differences
wrapper_class = WarpFrame if gray_scale else None

# Training Environment
train_env = make_vec_env(cst.LOG_DIR, n_envs=1, wrapper_class=wrapper_class)
train_env = VecFrameStack(train_env, n_stack=4)
train_env = VecTransposeImage(train_env)

# Testing Environment
eval_env = make_vec_env(cst.LOG_DIR, n_envs=1, wrapper_class=wrapper_class)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env)

verbose = 1

def create_path_if_missing(path: str):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        
def get_date_and_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%dT%H-%M-%S")

class RewardEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes=5, verbose=1, save_folder = ""):
        super(RewardEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []
        
        self.best_mean_reward = None
        create_path_if_missing(save_folder)
        self.save_folder = save_folder

    def _on_step(self) -> bool:
        # Evaluate every eval_freq calls
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env,
                                             n_eval_episodes=self.n_eval_episodes,
                                             render=False)
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(mean_reward)
            self.std_rewards.append(std_reward)
            if self.verbose:
                print(f"Evaluation at timestep {self.num_timesteps}: mean reward = {mean_reward:.2f}")
                
            if self.best_mean_reward is None or mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self._save_model("model_best")
                print("Best model saved")
        
        return True
    
    def _on_training_end(self):
        self._save_model("model_final")
        
        eval_path = os.path.join(self.save_folder, f"model_eval")
        np.savez(eval_path, timesteps=self.timesteps, 
                 mean_rewards=self.mean_rewards, 
                 std_rewards=self.std_rewards,
                )
        
    def _save_model(self, model_name):
        model_path = os.path.join(self.save_folder, model_name)
        self.model.save(model_path)

def get_model(model_name: str):
    verbose = 0
    if model_name =='ppo':
        return PPO('CnnPolicy', train_env, verbose=verbose, ent_coef=0.005)
    elif model_name == 'ddpg':
        return DDPG('CnnPolicy', train_env, verbose=verbose, buffer_size=10000)
    elif model_name == 'sac':
        return SAC("CnnPolicy", train_env, verbose=verbose, buffer_size=10000)
    else:
        raise Exception(f"{model_name} is not implemented")

def parse_args():
    parser = argparse.ArgumentParser(description='Train model from stable basline 3')
    
    parser.add_argument('-m', '--model', 
                        action='store', 
                        type=str, 
                        default='ppo', 
                        help="'ppo' | 'ddpg' | 'sac")
    
    parser.add_argument('-ts', '--timesteps',
                        action='store',
                        type=int,
                        default=75000,
                        help='Total number of timesteps used for model training')
    
    parser.add_argument('-ef', '--eval_freq',
                        action='store',
                        type=int,
                        default=2500,
                        help='Frequence of evaluations')
    
    parser.add_argument('-ee', '--eval_episode',
                        action='store',
                        type=int,
                        default=20,
                        help='Number of epoch of the evaluation')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    MODEL_NAME = args.model
    time_code = get_date_and_time()
    print(MODEL_NAME + "-" + time_code)
    
    model = get_model(MODEL_NAME)
    eval_callback = RewardEvalCallback(eval_env, eval_freq=args.eval_freq, n_eval_episodes=args.eval_episode, save_folder=f"{cst.LOG_DIR}/{MODEL_NAME}-{time_code}")

    total_timesteps = args.timesteps
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)

    final_mean_reward, final_std_reward = evaluate_policy(model, train_env, n_eval_episodes=args.eval_episode)
    print(f"Final Model - Mean reward: {final_mean_reward:.2f} +/- {final_std_reward:.2f}")

    """
    video_env = make_vec_env(env_str, n_envs=1, seed=0, wrapper_class=wrapper_class)
    video_env = VecFrameStack(video_env, n_stack=4)
    video_env = VecTransposeImage(video_env)
    video_env = VecVideoRecorder(video_env, cst.video_dir, video_length=10000, record_video_trigger=lambda x: x == 0,name_prefix=f"{MODEL_NAME}_car_racing")

    # Record a video
    obs = video_env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs)
        obs, reward, done, info = video_env.step(action)
        video_env.render()
        if done:
            break
    video_env.close()
    """

    """plt.figure()
    plt.plot(eval_callback.timesteps, eval_callback.mean_rewards, marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title(f"{MODEL_NAME} Performance on {env_str}")
    plt.grid()
    plt.show()"""

    train_env.close()
    eval_env.close()
