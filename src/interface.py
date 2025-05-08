import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from load_agent import load_best_agent

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import constant as cst


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train model from stable basline 3')

    parser.add_argument('-m', '--model',
                        action='store',
                        type=str,
                        default='ppo',
                        help="'ppo' | 'ddpg' | 'sac'")

    parser.add_argument('-sb', '--stable_baseline',
                        action='store',
                        type=bool,
                        default=True,
                        help="Use model trained with stable baseline or our custom implementation")

    return parser.parse_args()

# Load the trained agent
# env = gym.make(cst.env_str, render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)


if __name__ == '__main__':
    env = make_vec_env(cst.env_str, wrapper_class=WarpFrame,
                       env_kwargs={"render_mode": "human",
                                   "lap_complete_percent": 0.95,
                                   "domain_randomize": False,
                                   "continuous": True})
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    observation = env.reset()

    args = parse_args()
    MODEL_NAME = args.model
    sb3 = args.stable_baseline

    # load agent
    if sb3:
        path = cst.AGENT_PATH[MODEL_NAME]
    
    else:
        path = cst.C

    model = load_best_agent(cst.AGENT_PATH[MODEL_NAME])

    # Run the agent in the environment
    episode_over = False
    rewards = []
    while not episode_over:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, info = env.step(action)
        rewards.append(reward)
        episode_over = terminated

    # Print the total reward
    print(f"Total reward: {sum(rewards)}")

    # Close the environment
    env.close()

    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Reward over time")
    plt.show()
