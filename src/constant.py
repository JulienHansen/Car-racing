import os

env_str = "CarRacing-v3"

CFG_DIR = os.path.join(".", "src", "algo", "cfg_agent")
LOG_DIR = os.path.join(".", "logs", env_str)
VIDEO_DIR = os.path.join(".", "videos")
GRAPH_DIR = os.path.join(".", "graphs")

AGENT_PATH = {
    "ppo"  : f"{LOG_DIR}/ppo-2025-05-07T03-04-05",
    "sac"  : f"{LOG_DIR}/sac-2025-05-06T21-28-14",
    "ddpg" : f"{LOG_DIR}/ddpg-2025-05-06T16-28-43",
}

CUSTOM_AGENT_PATH = {
    "ppo"  : (f"{LOG_DIR}/ppo-custom", f"{CFG_DIR}/cfg_ppo_eval.yaml"),
    "beta_ppo"  : (f"{LOG_DIR}/beta_ppo-custom", f"{CFG_DIR}/cfg_ppo_eval.yaml"),
    "ddpg" : (f"{LOG_DIR}/ddpg-custom", f"{CFG_DIR}/cfg_ddpg.yaml"),
}