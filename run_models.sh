echo "Running PPO model"
py Car-racing/src/sb3/interface.py --model "ppo" --timesteps 25000

echo "Running DDPG model..."
py Car-racing/src/sb3/interface.py --model "ddpg" --timesteps 25000

echo "Running SAC model..."
py Car-racing/src/sb3/interface.py --model "sac" --timesteps 25000