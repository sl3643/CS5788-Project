from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_representations import get_enabled_models, model_output_dir
from src.metrics.baselines import random_pair_baseline, shuffled_pair_baseline
from src.metrics.cka import layerwise_cka, summarize_cka_matrix
from src.metrics.nn_overlap import layerwise_nn_overlap, summarize_overlap_matrix
from src.utils import ensure_dir, load_config, resolve_path, set_seed


def load_representations(config: Dict[str, Any], model_cfg: Dict[str, Any]) -> np.ndarray:
    """
    Load saved representation tensor for one model
    """
    pooling = config.get("extraction", {}).get("pooling", "mean")
    rep_path = model_output_dir(config, model_cfg["name"]) / f"reps_{pooling}.npz"

    if not rep_path.exists():
        raise FileNotFoundError(
            f"Representation file not found for {model_cfg['name']}: {rep_path}\n"
            "Run scripts/02_extract_all.py first"
        )

    data = np.load(rep_path)
    return data["representations"]


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    """
    Save a metric matrix as both numpy and csv files
    """
    ensure_dir(path.parent)
    np.save(path.with_suffix(".npy"), matrix)
    pd.DataFrame(matrix).to_csv(path.with_suffix(".csv"), index=False)


def pair_name(model_a: str, model_b: str) -> str:
    """
    Create a stable name for a model pair
    """
    return f"{model_a}__vs__{model_b}"


def compute_pair_metrics(
    config: Dict[str, Any],
    model_a: Dict[str, Any],
    model_b: Dict[str, Any],
    reps_a: np.ndarray,
    reps_b: np.ndarray,
) -> dict[str, Any]:
    """
    Compute all metrics for one model pair
    """
    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    metrics_dir = ensure_dir(output_dir / "metrics")
    matrices_dir = ensure_dir(metrics_dir / "matrices")

    name_a = model_a["name"]
    name_b = model_b["name"]
    name_pair = pair_name(name_a, name_b)

    print(f"\nComputing metrics for {name_a} vs {name_b}")
    print(f"  {name_a} reps: {reps_a.shape}")
    print(f"  {name_b} reps: {reps_b.shape}")

    cka_matrix = layerwise_cka(reps_a, reps_b)
    save_matrix(cka_matrix, matrices_dir / f"cka__{name_pair}")
    cka_summary = summarize_cka_matrix(cka_matrix)

    nn_cfg = config.get("metrics", {}).get("nn_overlap", {})
    metric_name = nn_cfg.get("metric", "cosine")
    k_values = nn_cfg.get("k_values", [10])
    primary_k = int(k_values[-1])

    nn_matrix = layerwise_nn_overlap(
        reps_a,
        reps_b,
        k=primary_k,
        metric=metric_name,
    )
    save_matrix(nn_matrix, matrices_dir / f"nn_overlap_k{primary_k}__{name_pair}")
    nn_summary = summarize_overlap_matrix(nn_matrix)

    seed = int(config.get("project", {}).get("seed", 42))

    shuffled_a, shuffled_b = shuffled_pair_baseline(reps_a, reps_b, seed=seed)
    shuffled_cka_matrix = layerwise_cka(shuffled_a, shuffled_b)
    save_matrix(shuffled_cka_matrix, matrices_dir / f"cka_shuffled__{name_pair}")
    shuffled_cka_summary = summarize_cka_matrix(shuffled_cka_matrix)

    random_a, random_b = random_pair_baseline(reps_a, reps_b, seed=seed)
    random_cka_matrix = layerwise_cka(random_a, random_b)
    save_matrix(random_cka_matrix, matrices_dir / f"cka_random__{name_pair}")
    random_cka_summary = summarize_cka_matrix(random_cka_matrix)

    row = {
        "model_a": name_a,
        "model_b": name_b,
        "family_a": model_a.get("family", "unknown"),
        "family_b": model_b.get("family", "unknown"),
        "layers_a": int(reps_a.shape[0]),
        "layers_b": int(reps_b.shape[0]),
        "num_texts": int(reps_a.shape[1]),
        "dim_a": int(reps_a.shape[2]),
        "dim_b": int(reps_b.shape[2]),
        "mean_cka": cka_summary["mean_cka"],
        "max_cka": cka_summary["max_cka"],
        "best_cka_layer_a": cka_summary["best_layer_a"],
        "best_cka_layer_b": cka_summary["best_layer_b"],
        "mean_nn_overlap": nn_summary["mean_nn_overlap"],
        "max_nn_overlap": nn_summary["max_nn_overlap"],
        "best_nn_layer_a": nn_summary["best_nn_layer_a"],
        "best_nn_layer_b": nn_summary["best_nn_layer_b"],
        "nn_k": primary_k,
        "shuffled_mean_cka": shuffled_cka_summary["mean_cka"],
        "shuffled_max_cka": shuffled_cka_summary["max_cka"],
        "random_mean_cka": random_cka_summary["mean_cka"],
        "random_max_cka": random_cka_summary["max_cka"],
    }

    print(f"  Mean CKA: {row['mean_cka']:.4f}")
    print(f"  Max CKA: {row['max_cka']:.4f}")
    print(f"  Mean NN overlap@{primary_k}: {row['mean_nn_overlap']:.4f}")
    print(f"  Shuffled mean CKA: {row['shuffled_mean_cka']:.4f}")
    print(f"  Random mean CKA: {row['random_mean_cka']:.4f}")

    return row


def compute_all_metrics(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute metrics for all enabled model pairs
    """
    model_cfgs = get_enabled_models(config)
    if len(model_cfgs) < 2:
        raise ValueError("Need at least two enabled models to compute pairwise metrics")

    reps_by_model = {
        model_cfg["name"]: load_representations(config, model_cfg)
        for model_cfg in model_cfgs
    }

    rows = []
    for model_a, model_b in itertools.combinations(model_cfgs, 2):
        reps_a = reps_by_model[model_a["name"]]
        reps_b = reps_by_model[model_b["name"]]
        row = compute_pair_metrics(config, model_a, model_b, reps_a, reps_b)
        rows.append(row)

    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    metrics_dir = ensure_dir(output_dir / "metrics")
    summary_df = pd.DataFrame(rows)

    summary_path = metrics_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    cka_path = metrics_dir / "cka_results.csv"
    summary_df[
        [
            "model_a",
            "model_b",
            "mean_cka",
            "max_cka",
            "best_cka_layer_a",
            "best_cka_layer_b",
            "shuffled_mean_cka",
            "random_mean_cka",
        ]
    ].to_csv(cka_path, index=False)

    nn_path = metrics_dir / "nn_overlap_results.csv"
    summary_df[
        [
            "model_a",
            "model_b",
            "mean_nn_overlap",
            "max_nn_overlap",
            "best_nn_layer_a",
            "best_nn_layer_b",
            "nn_k",
        ]
    ].to_csv(nn_path, index=False)

    print("\nSaved metric summaries")
    print(f"  {summary_path}")
    print(f"  {cka_path}")
    print(f"  {nn_path}")

    return summary_df


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Compute representation similarity metrics")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run metric computation
    """
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    summary_df = compute_all_metrics(config)
    print("\nMetric computation complete")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()