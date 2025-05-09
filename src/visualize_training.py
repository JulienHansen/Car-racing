import matplotlib.pyplot as plt
import numpy as np
import argparse

import constant as cst

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train model from stable basline 3')

    parser.add_argument('-m', '--model',
                        action='store',
                        type=str,
                        default='ppo',
                        help="<model_name>-xxxx-xx-xxTxx-xx-xx")

    return parser.parse_args()


arg = parse_args()
model_path = f"{cst.LOG_DIR}/{arg.model}"

#path = f"logs/custom/model_eval.npz"
model_eval = np.load(model_path)

sample = 1
timesteps = model_eval["timesteps"][::sample]
mean_rewards = model_eval["mean_rewards"][::sample]
std_rewards = model_eval["std_rewards"][::sample]

plt.plot(timesteps, mean_rewards)
plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.5)
plt.grid(alpha=0.3)

plt.ylabel("Reward")
plt.xlabel("Timestep")
plt.title(f"Evolution of reward for model {model_path.split("-")[0].upper()}")

plt.savefig(f"{cst.GRAPH_DIR}/{arg.model}")