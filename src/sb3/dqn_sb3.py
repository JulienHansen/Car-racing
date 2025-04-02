import gymnasium as gym
from stable_baselines3 import DQN


env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

# Initialize DQN
model = DQN('CnnPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=1000000)

# Save the model
model.save("dqn_car_racing")