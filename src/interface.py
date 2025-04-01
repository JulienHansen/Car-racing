import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from load_agent import load_best_agent


# Load the trained agent
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
observation, info = env.reset()

# load agent
model = load_best_agent()

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
