@echo off

echo Running PPO model...
py Car-racing/src/sb3/interface.py --model "ppo" --timesteps 75000

echo Running DDPG model...
py Car-racing\src\sb3\interface.py --model "ddpg" --timesteps 75000

echo Running SAC model...
py Car-racing\src\sb3\interface.py --model "sac" --timesteps 75000