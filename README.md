# Car-racing
Repo for solving car-racing gymnasium environment through Reinforcement learning.

## Question 2 : External library algorithm

### Training
In order to start the training of a model presented in the report, you can use the command

```bash
python src/sb3/interface.py \   
    --model "ppo" \             # "ppo" | "ddp" | "sac"
    --timesteps 7500 \          # number of timesteps used for training
    --eval_freq 2500 \          # frequence at which the model is evaluated
    --eval_episode 20           # number of episode on which the model is evaluated
```

The three last parameters are optional. As for the model, the accepted values are "ppo", "ddpg" and "sac"

* Starting PPO
```bash
python src/sb3/interface.py --model "ppo"
```

* Starting DDPG
```bash
python src/sb3/interface.py --model "ddpg"
```

* Starting SAC
```bash
python src/sb3/interface.py --model "sac"
```

### Evaluation
* Evaluating PPO
```bash
python src/interface.py --model "ppo" -sb
```
* Evaluating DDPG
```bash
python src/interface.py --model "ddpg" -sb
```
* Evaluating SAC
```bash
python src/interface.py --model "sac" -sb
```
## Question 3 : Personal algorithm

### Training

* Training DDPG
```bash
python src/algo/ddpg.py --config cfg_agent/cfg_ddpg.yaml
```
* Training PPO
```bash
python src/algo/ppo.py --config cfg_agent/cfg_ppo.yaml
```
* Training Beta PPO
```bash
python src/algo/beta_ppo.py --config cfg_agent/cfg_ppo.yaml
```


Add command

### Evalulation
* Evaluating PPO
```bash
python src/interface.py --model "ppo" 
```
* Evaluating PPO with beta distribution
```bash
python src/interface.py --model "beta_ppo" 
```
* Evaluating DDPG
```bash
python src/interface.py --model "ddpg" 
```