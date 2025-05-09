from pathlib import Path

from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC


def load_best_agent(path ,sb3=True):
    """
    Load the best agent from the specified path.

    This function is designed to load your pre-trained agent model so that it can be used to
    interact with the environment. Follow these steps to implement the function:

    1) Choose the right library for model loading:
       - Depending on whether you used PyTorch, TensorFlow, or another framework to train your model,
         import the corresponding library (e.g., `import torch` or `import tensorflow as tf`).

    2) Specify the correct file path:
       - Define the path where your trained model is saved.
       - Ensure the path is correct and accessible from your script.

    3) Load the model:
       - Use the appropriate loading function from your library.
         For example, with PyTorch you might use:
           ```python
           model = torch.load('path/to/your_model.pth')
           ```

    4) Ensure the model is callable:
       - The loaded model should be usable like a function. When you call:
           ```python
           action = model(observation)
           ```
         it should output an action based on the input observation.

    Returns:
        model: The loaded model. It must be callable so that when you pass an observation to it,
               it returns the corresponding action.

    Example usage:
        >>> model = load_best_agent()
        >>> observation = get_current_observation()  # Your method to fetch the current observation.
        >>> action = model(observation)
    """
    
    model_name = path.split("/")[-1].split("-")[0].strip()
    
    path = Path(path).joinpath("model_best.zip")
    assert path.exists(), f"{path} doesn't exist"
    
    if sb3:
      if model_name == "ppo":
        return PPO.load(path)
      elif model_name == "ddpg":
        return DDPG.load(path)
      elif model_name == "sac":
        return SAC.load(path)
      else:
        raise NotImplementedError(f"{model_name} is not implemented")
      
    if model_name == "ppo" or model_name == "beta_ppo":
      return None
    elif model_name == "ddpg":
      return None
    else:
      raise NotImplementedError(f"{model_name} is not implemented")

