# Car-racing
Repo for solving car-racing gymnasium environment through Reinforcement learning.

## Question 2 : External library algorithm
In order to start the training of a model presented in the report, you can use the command

```bash
python Car-racing/src/sb3/interface.py \
    --model "ppo" \
    --timesteps 7500 \
    --eval_freq 2500 \
    --eval_episode 20
```

The three last parameters are optional. As for the model, the accepted values are "ppo", "ddpg" and "sac"

### Starting PPO
```bash
python Car-racing/src/sb3/interface.py --model "ppo"
```

### Starting DDPG
```bash
python Car-racing/src/sb3/interface.py --model "ddpg"
```

### Starting SAC
```bash
python Car-racing/src/sb3/interface.py --model "sac"
```