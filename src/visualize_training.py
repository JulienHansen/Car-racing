import matplotlib.pyplot as plt
import numpy as np

import constant as cst

MODEL_NAME = "ppo"

model_path = cst.AGENT_PATH[MODEL_NAME]

model_eval = np.load(f"{model_path}/model_eval.npz")
timesteps = model_eval["timesteps"]
mean_rewards = model_eval["mean_rewards"]
std_rewards = model_eval["std_rewards"]

plt.plot(timesteps, mean_rewards)
plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
plt.grid(alpha=0.3)

plt.ylabel("Reward")
plt.xlabel("Timestep")
plt.title(f"Evolution of reward for model {MODEL_NAME.upper()}")

plt.savefig(f"{cst.GRAPH_DIR}/{model_path.split('/')[-1]}")