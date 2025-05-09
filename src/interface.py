import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from load_agent import load_best_agent

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from algo.ppo import evaluate as ppo_evaluate

import constant as cst
import torch


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
                        default=False,
                        help="Use model trained with stable baseline or our custom implementation")

    return parser.parse_args()

def ppo_evaluate(agent, env_eval, device):
    agent.eval()

    obs, _ = env_eval.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    dones = np.zeros(env_eval.num_envs, dtype=bool)
    
    episode_rewards = []

    while not np.any(dones):
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs)
        next_obs, rewards, terminations, truncations, infos = env_eval.step(actions.cpu().numpy())
        dones = np.logical_or(terminations, truncations)
        
        episode_rewards.append(rewards[0])

    return episode_rewards

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
    
    print(sb3)

    # load agent
    if sb3:
        path = cst.AGENT_PATH[MODEL_NAME]
    else:
        path = cst.CUSTOM_AGENT_PATH[MODEL_NAME]

    model = load_best_agent(path, sb3)

    if sb3:
        # Run the agent in the environment
        episode_over = False
        rewards = []
        while not episode_over:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, info = env.step(action)
            rewards.append(reward)
            episode_over = terminated
    else:
        if MODEL_NAME == "ppo":
            rewards = ppo_evaluate(*model)

    # Print the total reward
    print(f"Total reward: {sum(rewards)}")

    # Close the environment
    env.close()

    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Reward over time")
    plt.show()
