import os
import sys
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import yaml
import numpy as np
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(directories):
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def save_model_info(model, path, metrics):
    info = {
        'model_summary': str(model.summary()),
        'metrics': metrics
    }
    with open(f"{path}/model_info.txt", 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

