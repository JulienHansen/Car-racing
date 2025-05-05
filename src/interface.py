import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from load_agent import load_best_agent

from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import constant as cst


# Load the trained agent
env = gym.make(cst.env_str, render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)
observation, info = env.reset()

# load agent
MODEL_NAME = "ppo" # "ppo" | "ddpg" | "sac"
model = load_best_agent(cst.AGENT_PATH[MODEL_NAME])

# Run the agent in the environment
episode_over = False
rewards = []
while not episode_over:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    episode_over = terminated or truncated

# Print the total reward
print(f"Total reward: {sum(rewards)}")

# Close the environment
env.close()

plt.plot(np.arange(len(rewards)), rewards)
plt.xlabel("Time step")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show()
