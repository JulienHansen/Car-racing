#!/usr/bin/env python3
"""
Self-contained FQI (with extremely randomized tree) implementation for the CarRacing-v3 environment.
This implementation uses CNNs to process image observations and outputs a 3-dimensional action:
    - Steering in [-1, 1]
    - Gas and Brake in [0, 1]
"""

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt