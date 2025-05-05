import os

env_str = "CarRacing-v3"

LOG_DIR = os.path.join(".", "logs", env_str)
VIDEO_DIR = os.path.join(".", "videos")
AGENT_PATH = {
    "ppo" : f"{LOG_DIR}/ppo-2025-05-05T10-35-48",
    "sac" : f"{LOG_DIR}/sac-2025-05-05T11-48-59",
}