from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist and return it as a Path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device_config: str = "auto") -> torch.device:
    """
    Resolve the compute device from the config
    """
    if device_config != "auto":
        return torch.device(device_config)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(dtype_config: str = "auto") -> torch.dtype:
    """
    Resolve torch dtype from the config
    In auto mode we use float16 on CUDA and float32 elsewhere
    """
    if dtype_config == "float16":
        return torch.float16
    if dtype_config == "bfloat16":
        return torch.bfloat16
    if dtype_config == "float32":
        return torch.float32
    if dtype_config != "auto":
        raise ValueError(f"Unsupported dtype setting: {dtype_config}")

    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def save_jsonl(records: list[Dict[str, Any]], path: str | Path) -> None:
    """
    Save a list of dictionaries to a JSONL file
    """
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def project_root() -> Path:
    """
    Return the repository root, assuming this file is under src
    """
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """
    Resolve a project-relative path into an absolute path
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a compact summary of the current experiment setup
    """
    data_cfg = config.get("data", {})
    model_cfgs = config.get("models", [])
    enabled_models = [m["name"] for m in model_cfgs if m.get("enabled", False)]

    print("Configuration summary")
    print(f"  Dataset: {data_cfg.get('dataset_name')} / {data_cfg.get('dataset_config')}")
    print(f"  Split: {data_cfg.get('split')}")
    print(f"  Number of texts: {data_cfg.get('num_texts')}")
    print(f"  Enabled models: {enabled_models}")


def file_size_mb(path: str | Path) -> float:
    """
    Return file size in MB
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)