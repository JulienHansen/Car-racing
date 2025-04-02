import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

# --- Parameters ---
ENV_ID = 'CarRacing-v3'  # Environment identifier
ENV_KWARGS = {'continuous': True}  # Additional keyword arguments for the environment

# For testing purposes; increase this for full training
TOTAL_TIMESTEPS = 10000 
LOG_INTERVAL = 10
MODEL_FILENAME = "ppo_car_racing_v3"  # Filename to save the trained model
TENSORBOARD_LOG_DIR = "./ppo_car_racing_v3_tensorboard/"  # Directory for TensorBoard logs
N_ENVS = 4  # Number of parallel environments
FRAME_STACK = 4  # Number of frames to stack for the observation
LEARNING_RATE = 3e-4
BATCH_SIZE = 64 * N_ENVS

# --- Create Environment ---
try:
    env = make_vec_env(ENV_ID, n_envs=N_ENVS, env_kwargs=ENV_KWARGS)
except gym.error.NameNotFound as e:
    print(f"Error: Environment ID '{ENV_ID}' not found.")
    print("Please ensure 'CarRacing-v3' is correctly installed and registered with Gymnasium.")
    print(f"Original error: {e}")
    exit()

env = VecFrameStack(env, n_stack=FRAME_STACK)

os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# --- Initialize PPO Model ---
model = PPO(
    'CnnPolicy',
    env,
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG_DIR,
    learning_rate=LEARNING_RATE,
    n_steps=1024,
    batch_size=BATCH_SIZE,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
)

# --- Train the Model ---
print(f"Starting training on {ENV_ID} for {TOTAL_TIMESTEPS} timesteps...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    log_interval=LOG_INTERVAL
)
print("Training finished.")

# --- Save the Model ---
model.save(MODEL_FILENAME)
print(f"Model saved to {MODEL_FILENAME}.zip")

# --- Clean up ---
env.close()
print("Environment closed.")
