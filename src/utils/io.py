import os
import pickle
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_object(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path: str) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
