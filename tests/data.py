"""Shared synthetic samples for the tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type GroupArray = NDArray[np.integer | np.str_]
type IndexArray = NDArray[np.intp]
type NumericArray = NDArray[np.integer | np.floating]
type LabelArray = NDArray[np.integer | np.bool_ | np.str_]


def linear_data(
    *,
    num_samples: int = 120,
    seed: int = 0,
) -> tuple[NumericArray, NDArray[np.float64]]:
    """Return 3-D features and continuous targets."""
    rng: np.random.Generator = np.random.default_rng(seed)
    features: NumericArray = rng.standard_normal((num_samples, 3))
    return features, rng.standard_normal(num_samples)


def separable_blobs(
    *,
    num_classes: int = 3,
    num_samples_per_class: int = 30,
    seed: int = 0,
) -> tuple[NumericArray, NDArray[np.int64]]:
    """Return 2-D Gaussian blobs and their integer labels."""
    rng: np.random.Generator = np.random.default_rng(seed)
    features: list[NumericArray] = [
        rng.standard_normal((num_samples_per_class, 2)) + 8.0 * label
        for label in range(num_classes)
    ]
    labels: NDArray[np.int64] = np.repeat(
        np.arange(num_classes, dtype=np.int64), num_samples_per_class
    )
    return np.vstack(features), labels
