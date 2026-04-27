from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def pairwise_distances(x: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise distances between examples
    """
    if metric == "cosine":
        return cosine_distances(x)

    if metric == "euclidean":
        return euclidean_distances(x)

    raise ValueError(f"Unsupported distance metric: {metric}")


def nearest_neighbors(x: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """
    Return k nearest-neighbor indices for each example

    Args:
        x: Representation matrix with shape [num_examples, hidden_dim]
        k: Number of nearest neighbors to return
        metric: Distance metric name

    Returns:
        Integer array with shape [num_examples, k]
    """
    if x.ndim != 2:
        raise ValueError("nearest_neighbors expects a 2D representation matrix")

    num_examples = x.shape[0]
    if k >= num_examples:
        raise ValueError(f"k must be smaller than num_examples, got k={k}, num_examples={num_examples}")

    distances = pairwise_distances(x, metric=metric)

    # Exclude each example itself from its neighbor list
    np.fill_diagonal(distances, np.inf)

    return np.argsort(distances, axis=1)[:, :k]


def nn_overlap_single_layer(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> float:
    """
    Compute nearest-neighbor overlap between two representation spaces
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("nn_overlap_single_layer expects two 2D arrays")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"NN overlap requires the same number of examples, got {x.shape[0]} and {y.shape[0]}"
        )

    neighbors_x = nearest_neighbors(x, k=k, metric=metric)
    neighbors_y = nearest_neighbors(y, k=k, metric=metric)

    overlaps = []
    for row_x, row_y in zip(neighbors_x, neighbors_y):
        overlap = len(set(row_x.tolist()).intersection(set(row_y.tolist()))) / k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def layerwise_nn_overlap(
    reps_a: np.ndarray,
    reps_b: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute layer-wise nearest-neighbor overlap between two models

    Args:
        reps_a: Array with shape [layers_a, num_examples, dim_a]
        reps_b: Array with shape [layers_b, num_examples, dim_b]
        k: Number of nearest neighbors
        metric: Distance metric name

    Returns:
        Overlap matrix with shape [layers_a, layers_b]
    """
    if reps_a.ndim != 3 or reps_b.ndim != 3:
        raise ValueError("layerwise_nn_overlap expects arrays with shape [layers, examples, dim]")

    if reps_a.shape[1] != reps_b.shape[1]:
        raise ValueError(
            f"Layer-wise NN overlap requires same number of examples, got {reps_a.shape[1]} and {reps_b.shape[1]}"
        )

    num_layers_a = reps_a.shape[0]
    num_layers_b = reps_b.shape[0]
    scores = np.zeros((num_layers_a, num_layers_b), dtype=np.float64)

    for i in range(num_layers_a):
        for j in range(num_layers_b):
            scores[i, j] = nn_overlap_single_layer(
                reps_a[i],
                reps_b[j],
                k=k,
                metric=metric,
            )

    return scores


def summarize_overlap_matrix(overlap_matrix: np.ndarray) -> dict[str, float | int]:
    """
    Summarize a layer-wise nearest-neighbor overlap matrix
    """
    if overlap_matrix.ndim != 2:
        raise ValueError("summarize_overlap_matrix expects a 2D matrix")

    best_flat_index = int(np.argmax(overlap_matrix))
    best_layer_a, best_layer_b = np.unravel_index(best_flat_index, overlap_matrix.shape)

    return {
        "mean_nn_overlap": float(np.mean(overlap_matrix)),
        "max_nn_overlap": float(np.max(overlap_matrix)),
        "min_nn_overlap": float(np.min(overlap_matrix)),
        "best_nn_layer_a": int(best_layer_a),
        "best_nn_layer_b": int(best_layer_b),
    }
