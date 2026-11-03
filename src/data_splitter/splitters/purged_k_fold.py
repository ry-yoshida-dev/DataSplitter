from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator
from ..types import IndexArray


@dataclass(frozen=True)
class PurgedKFoldSplitter(LabelFreeSplitter):
    """
    K-fold over ordered samples that removes training samples next to the test fold.

    Unlike forward chaining, every sample is tested once, because training samples
    may come from both sides of the test fold. Removing the samples within ``gap`` of
    it stops autocorrelation, such as overlapping windows, from leaking test
    information into training. Use forward chaining when the future must never
    inform the past.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    gap : int
        Number of samples removed from training on each side of the test fold.
    """

    n_splits: int = 5
    gap: int = 0

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(n_splits=self.n_splits, gap=self.gap)
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition ordered samples into contiguous folds.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples in order; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold, in order.

        Raises
        ------
        ValueError
            If there are fewer samples than folds, or the gap leaves a fold without
            training samples.
        """
        num_samples: int = len(X)
        if num_samples < self.n_splits:
            raise ValueError(
                f"X must have at least {self.n_splits} samples, got {num_samples}"
            )
        indices: IndexArray = np.arange(num_samples, dtype=np.intp)
        splits: list[DataSplit] = []
        for test_indices in np.array_split(indices, self.n_splits):
            first: int = test_indices.item(0) - self.gap
            last: int = test_indices.item(-1) + self.gap
            train_indices: IndexArray = indices[(indices < first) | (indices > last)]
            if train_indices.size == 0:
                raise ValueError(
                    f"gap={self.gap} leaves a fold without training samples"
                )
            splits.append(
                DataSplit(train_indices=train_indices, test_indices=test_indices)
            )
        return tuple(splits)
