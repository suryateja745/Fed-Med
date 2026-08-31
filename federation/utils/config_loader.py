"""
Config loader utility for FedMed Federated Learning.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fl_config.yaml"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    If no path is provided, loads the default fl_config.yaml.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_model_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract model configuration dictionary."""
    if config is None:
        config = load_config()
    return config.get("model", {})


def get_training_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract training configuration dictionary."""
    if config is None:
        config = load_config()
    return config.get("training", {})


def get_federation_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract federation round and server configuration dictionary."""
    if config is None:
        config = load_config()
    return config.get("federation", {})
