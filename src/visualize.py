from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP

from src.utils import ensure_dir, resolve_path


def clean_model_name(name: str) -> str:
    """
    Convert model names into safe filename fragments
    """
    return name.replace("/", "_").replace(" ", "_")


def load_metric_matrix(metrics_dir: Path, matrix_name: str) -> np.ndarray:
    """
    Load a saved metric matrix from outputs/metrics/matrices
    """
    path = metrics_dir / "matrices" / f"{matrix_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Metric matrix not found: {path}")
    return np.load(path)


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Save a heatmap for a layer-wise metric matrix
    """
    ensure_dir(output_path.parent)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar=True,
        square=False,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_cka_heatmaps(config: Dict) -> list[Path]:
    """
    Make one CKA heatmap per model pair
    """
    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    metrics_dir = output_dir / "metrics"
    figures_dir = ensure_dir(output_dir / "figures")
    summary_path = metrics_dir / "summary_table.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary table not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    output_paths: list[Path] = []

    for _, row in summary.iterrows():
        model_a = row["model_a"]
        model_b = row["model_b"]
        pair = f"{model_a}__vs__{model_b}"
        matrix = load_metric_matrix(metrics_dir, f"cka__{pair}")

        output_path = figures_dir / f"cka_{clean_model_name(model_a)}_vs_{clean_model_name(model_b)}.png"
        plot_heatmap(
            matrix=matrix,
            title=f"Layer-wise CKA: {model_a} vs {model_b}",
            xlabel=f"{model_b} layer",
            ylabel=f"{model_a} layer",
            output_path=output_path,
        )
        output_paths.append(output_path)

    return output_paths


def make_baseline_comparison(config: Dict) -> Path:
    """
    Make a bar plot comparing real CKA against shuffled and random baselines
    """
    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    metrics_dir = output_dir / "metrics"
    figures_dir = ensure_dir(output_dir / "figures")
    summary_path = metrics_dir / "summary_table.csv"

    summary = pd.read_csv(summary_path)
    rows = []
    for _, row in summary.iterrows():
        pair = f"{row['model_a']} vs {row['model_b']}"
        rows.append({"pair": pair, "setting": "real", "mean_cka": row["mean_cka"]})
        rows.append({"pair": pair, "setting": "shuffled", "mean_cka": row["shuffled_mean_cka"]})
        rows.append({"pair": pair, "setting": "random", "mean_cka": row["random_mean_cka"]})

    plot_df = pd.DataFrame(rows)
    output_path = figures_dir / "baseline_comparison.png"

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="pair", y="mean_cka", hue="setting")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.title("Mean CKA vs baselines")
    plt.xlabel("Model pair")
    plt.ylabel("Mean CKA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def make_nn_overlap_comparison(config: Dict) -> Path:
    """
    Make a bar plot of nearest-neighbor overlap by model pair
    """
    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    metrics_dir = output_dir / "metrics"
    figures_dir = ensure_dir(output_dir / "figures")
    summary_path = metrics_dir / "summary_table.csv"

    summary = pd.read_csv(summary_path)
    summary = summary.copy()
    summary["pair"] = summary["model_a"] + " vs " + summary["model_b"]

    output_path = figures_dir / "nn_overlap_comparison.png"

    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x="pair", y="mean_nn_overlap")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.title("Mean nearest-neighbor overlap")
    plt.xlabel("Model pair")
    plt.ylabel("Mean NN overlap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def load_final_layer_representations(config: Dict) -> dict[str, np.ndarray]:
    """
    Load final-layer sentence representations for all enabled models
    """
    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    pooling = config.get("extraction", {}).get("pooling", "mean")
    reps_dir = output_dir / "representations"

    final_reps: dict[str, np.ndarray] = {}
    for model_cfg in config.get("models", []):
        if not model_cfg.get("enabled", False):
            continue

        model_name = model_cfg["name"]
        rep_path = reps_dir / model_name / f"reps_{pooling}.npz"
        if not rep_path.exists():
            raise FileNotFoundError(f"Representation file not found: {rep_path}")

        data = np.load(rep_path)
        reps = data["representations"]
        final_reps[model_name] = reps[-1]

    return final_reps


def make_pca_final_layer_plot(config: Dict) -> Path:
    """
    Make a PCA visualization using final-layer sentence representations
    """
    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    figures_dir = ensure_dir(output_dir / "figures")
    final_reps = load_final_layer_representations(config)

    rows = []
    for model_name, reps in final_reps.items():
        # PCA requires a shared feature dimension, so fit PCA separately per model and compare geometry qualitatively
        projected = PCA(n_components=2, random_state=config.get("project", {}).get("seed", 42)).fit_transform(reps)
        for idx, point in enumerate(projected):
            rows.append(
                {
                    "model": model_name,
                    "text_id": idx,
                    "x": point[0],
                    "y": point[1],
                }
            )

    plot_df = pd.DataFrame(rows)
    output_path = figures_dir / "pca_final_layers.png"

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="model", alpha=0.75, s=35)
    plt.title("PCA of final-layer sentence representations")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def make_umap_final_layer_plot(config: Dict) -> Path:
    """
    Make a UMAP visualization using final-layer sentence representations
    """
    output_dir = resolve_path(config.get("project", {}).get("output_dir", "outputs"))
    figures_dir = ensure_dir(output_dir / "figures")
    final_reps = load_final_layer_representations(config)
    seed = int(config.get("project", {}).get("seed", 42))

    rows = []
    for model_name, reps in final_reps.items():
        n_neighbors = min(15, max(2, reps.shape[0] - 1))
        projected = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        ).fit_transform(reps)
        for idx, point in enumerate(projected):
            rows.append(
                {
                    "model": model_name,
                    "text_id": idx,
                    "x": point[0],
                    "y": point[1],
                }
            )

    plot_df = pd.DataFrame(rows)
    output_path = figures_dir / "umap_final_layers.png"

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="model", alpha=0.75, s=35)
    plt.title("UMAP of final-layer sentence representations")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def make_all_figures(config: Dict) -> list[Path]:
    """
    Generate all currently supported figures
    """
    output_paths = []
    output_paths.extend(make_cka_heatmaps(config))
    output_paths.append(make_baseline_comparison(config))
    output_paths.append(make_nn_overlap_comparison(config))
    output_paths.append(make_pca_final_layer_plot(config))
    output_paths.append(make_umap_final_layer_plot(config))
    return output_paths