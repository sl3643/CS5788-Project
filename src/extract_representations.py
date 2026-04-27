from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from src.model_utils import (
    batch_texts,
    count_model_parameters,
    forward_hidden_states,
    load_model_and_tokenizer,
    tokenize_batch,
)
from src.pooling import pool_all_layers
from src.utils import ensure_dir, get_device, get_dtype, load_jsonl, resolve_path


def get_enabled_models(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Return model configs with enabled set to true
    """
    return [model_cfg for model_cfg in config.get("models", []) if model_cfg.get("enabled", False)]


def read_texts_from_jsonl(path: str | Path) -> list[str]:
    """
    Read processed texts from a JSONL file
    """
    records = load_jsonl(resolve_path(path))
    return [record["text"] for record in records]


def model_output_dir(config: Dict[str, Any], model_name: str) -> Path:
    """
    Return the output directory for one model
    """
    output_root = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    return output_root / "representations" / model_name


def save_model_metadata(
    output_dir: Path,
    model_cfg: Dict[str, Any],
    representations: np.ndarray,
    num_parameters: int,
    extraction_cfg: Dict[str, Any],
) -> None:
    """
    Save metadata describing extracted representations
    """
    metadata = {
        "name": model_cfg["name"],
        "hf_id": model_cfg["hf_id"],
        "family": model_cfg.get("family", "unknown"),
        "num_parameters": int(num_parameters),
        "num_layers": int(representations.shape[0]),
        "num_texts": int(representations.shape[1]),
        "hidden_dim": int(representations.shape[2]),
        "pooling": extraction_cfg.get("pooling", "mean"),
        "max_length": int(extraction_cfg.get("max_length", 128)),
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def extract_representations_for_model(
    config: Dict[str, Any],
    model_cfg: Dict[str, Any],
    texts: list[str],
) -> Path:
    """
    Extract and save pooled hidden representations for one model
    """
    extraction_cfg = config.get("extraction", {})
    device = get_device(extraction_cfg.get("device", "auto"))
    dtype = get_dtype(extraction_cfg.get("dtype", "auto"))
    batch_size = int(extraction_cfg.get("batch_size", 4))
    max_length = int(extraction_cfg.get("max_length", 128))
    pooling = extraction_cfg.get("pooling", "mean")

    output_dir = model_output_dir(config, model_cfg["name"])
    ensure_dir(output_dir)
    output_path = output_dir / f"reps_{pooling}.npz"

    print(f"\nExtracting representations for {model_cfg['name']} ({model_cfg['hf_id']})")
    print(f"Device: {device}")
    print(f"Pooling: {pooling}")

    model, tokenizer = load_model_and_tokenizer(model_cfg, device=device, dtype=dtype)
    num_parameters = count_model_parameters(model)
    print(f"Parameters: {num_parameters:,}")

    pooled_batches: list[np.ndarray] = []

    for text_batch in tqdm(
        batch_texts(texts, batch_size),
        total=(len(texts) + batch_size - 1) // batch_size,
        desc=f"{model_cfg['name']}",
    ):
        inputs = tokenize_batch(
            tokenizer=tokenizer,
            texts=text_batch,
            max_length=max_length,
            device=device,
        )
        hidden_states = forward_hidden_states(model, inputs)
        pooled = pool_all_layers(
            hidden_states=hidden_states,
            attention_mask=inputs["attention_mask"],
            pooling=pooling,
        )

        pooled_batches.append(pooled.detach().cpu().float().numpy())

    # Each batch has shape [num_layers, batch_size, hidden_dim]
    # Concatenate over the text dimension to get [num_layers, num_texts, hidden_dim]
    representations = np.concatenate(pooled_batches, axis=1)

    np.savez_compressed(
        output_path,
        representations=representations,
        model_name=model_cfg["name"],
        hf_id=model_cfg["hf_id"],
        pooling=pooling,
    )
    save_model_metadata(
        output_dir=output_dir,
        model_cfg=model_cfg,
        representations=representations,
        num_parameters=num_parameters,
        extraction_cfg=extraction_cfg,
    )

    print(f"Saved representations to {output_path}")
    print(f"Shape: {representations.shape}")

    del model
    del tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return output_path


def extract_all_representations(config: Dict[str, Any]) -> list[Path]:
    """
    Extract representations for all enabled models
    """
    data_cfg = config["data"]
    texts = read_texts_from_jsonl(data_cfg["processed_path"])
    model_cfgs = get_enabled_models(config)

    if not model_cfgs:
        raise ValueError("No enabled models found in config")

    print(f"Loaded {len(texts)} texts")
    print(f"Enabled models: {[model_cfg['name'] for model_cfg in model_cfgs]}")

    output_paths = []
    for model_cfg in model_cfgs:
        output_path = extract_representations_for_model(
            config=config,
            model_cfg=model_cfg,
            texts=texts,
        )
        output_paths.append(output_path)

    return output_paths