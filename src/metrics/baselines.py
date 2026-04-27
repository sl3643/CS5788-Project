from __future__ import annotations

import numpy as np


def shuffle_examples(reps: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Shuffle the example dimension of a representation tensor

    Args:
        reps: Array with shape [num_layers, num_examples, hidden_dim]
        seed: Random seed

    Returns:
        Array with the same shape as reps, with examples permuted
    """
    if reps.ndim != 3:
        raise ValueError("shuffle_examples expects an array with shape [layers, examples, dim]")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(reps.shape[1])
    return reps[:, permutation, :]


def random_gaussian_like(reps: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Generate a random Gaussian representation tensor with the same shape as reps

    The random tensor is standardized per layer to roughly match the scale of centered features
    """
    if reps.ndim != 3:
        raise ValueError("random_gaussian_like expects an array with shape [layers, examples, dim]")

    rng = np.random.default_rng(seed)
    random_reps = rng.standard_normal(size=reps.shape).astype(np.float32)

    layer_mean = random_reps.mean(axis=(1, 2), keepdims=True)
    layer_std = random_reps.std(axis=(1, 2), keepdims=True) + 1e-12
    random_reps = (random_reps - layer_mean) / layer_std

    return random_reps


def shuffled_pair_baseline(
    reps_a: np.ndarray,
    reps_b: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a pair where the second model's examples are shuffled
    """
    return reps_a, shuffle_examples(reps_b, seed=seed)


def random_pair_baseline(
    reps_a: np.ndarray,
    reps_b: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a pair of random Gaussian representation tensors matching the original shapes
    """
    random_a = random_gaussian_like(reps_a, seed=seed)
    random_b = random_gaussian_like(reps_b, seed=seed + 1)
    return random_a, random_b