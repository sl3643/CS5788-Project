from __future__ import annotations

import numpy as np


def center_columns(x: np.ndarray) -> np.ndarray:
    """
    Center each feature dimension by subtracting the sample mean
    """
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute linear CKA between two representation matrices

    Args:
        x: Representation matrix with shape [num_examples, dim_x]
        y: Representation matrix with shape [num_examples, dim_y]
        eps: Small constant for numerical stability

    Returns:
        Scalar CKA value
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear_cka expects two 2D arrays")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"CKA requires the same number of examples, got {x.shape[0]} and {y.shape[0]}"
        )

    x = center_columns(x.astype(np.float64, copy=False))
    y = center_columns(y.astype(np.float64, copy=False))

    xy = x.T @ y
    numerator = np.linalg.norm(xy, ord="fro") ** 2

    xx = x.T @ x
    yy = y.T @ y
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")

    return float(numerator / (denominator + eps))


def layerwise_cka(reps_a: np.ndarray, reps_b: np.ndarray) -> np.ndarray:
    """
    Compute layer-wise CKA between two models

    Args:
        reps_a: Array with shape [layers_a, num_examples, dim_a]
        reps_b: Array with shape [layers_b, num_examples, dim_b]

    Returns:
        CKA matrix with shape [layers_a, layers_b]
    """
    if reps_a.ndim != 3 or reps_b.ndim != 3:
        raise ValueError("layerwise_cka expects arrays with shape [layers, examples, dim]")

    if reps_a.shape[1] != reps_b.shape[1]:
        raise ValueError(
            f"Layer-wise CKA requires same number of examples, got {reps_a.shape[1]} and {reps_b.shape[1]}"
        )

    num_layers_a = reps_a.shape[0]
    num_layers_b = reps_b.shape[0]
    scores = np.zeros((num_layers_a, num_layers_b), dtype=np.float64)

    for i in range(num_layers_a):
        for j in range(num_layers_b):
            scores[i, j] = linear_cka(reps_a[i], reps_b[j])

    return scores


def summarize_cka_matrix(cka_matrix: np.ndarray) -> dict[str, float | int]:
    """
    Summarize a layer-wise CKA matrix with useful aggregate statistics
    """
    if cka_matrix.ndim != 2:
        raise ValueError("summarize_cka_matrix expects a 2D matrix")

    best_flat_index = int(np.argmax(cka_matrix))
    best_layer_a, best_layer_b = np.unravel_index(best_flat_index, cka_matrix.shape)

    return {
        "mean_cka": float(np.mean(cka_matrix)),
        "max_cka": float(np.max(cka_matrix)),
        "min_cka": float(np.min(cka_matrix)),
        "best_layer_a": int(best_layer_a),
        "best_layer_b": int(best_layer_b),
    }