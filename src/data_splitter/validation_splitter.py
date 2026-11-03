from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data_split import DataSplit
from .label_free_splitter import LabelFreeSplitter
from .splitter import DatasetSplitter
from .types import IndexArray
from .validated_split import ValidatedSplit


@dataclass(frozen=True)
class ValidationSplitter:
    """
    Train/validation/test partitions from two strategies, also for nested validation.

    The outer strategy sets the test samples aside. The inner strategy then divides
    the remaining samples, so that tuning on the validation samples never touches
    the test samples. One split results per combination of outer and inner partition.

    Strategies holding per-sample ``groups`` cannot be the inner strategy, because
    those groups would not match the remaining samples.

    Attributes
    ----------
    outer : DatasetSplitter
        The strategy that sets the test samples aside.
    inner : DatasetSplitter
        The strategy that divides the rest into training and validation samples.
    """

    outer: DatasetSplitter
    inner: DatasetSplitter

    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[ValidatedSplit, ...]:
        """
        Partition samples into train, validation, and test indices.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets; required if a strategy needs labels.

        Returns
        -------
        tuple[ValidatedSplit, ...]
            One split per combination of outer and inner partition.

        Raises
        ------
        ValueError
            If y is None but a strategy needs labels, or the strategies reject the
            samples.
        """
        results: list[ValidatedSplit] = []
        for outer_split in self._apply(self.outer, X, y):
            remaining: IndexArray = outer_split.train_indices
            inner_y: NDArray[np.generic] | None = None if y is None else y[remaining]
            for inner_split in self._apply(self.inner, remaining, inner_y):
                results.append(
                    ValidatedSplit(
                        train_indices=remaining[inner_split.train_indices],
                        validation_indices=remaining[inner_split.test_indices],
                        test_indices=outer_split.test_indices,
                    )
                )
        return tuple(results)

    @staticmethod
    def _apply(
        splitter: DatasetSplitter,
        X: NDArray[np.generic],
        y: NDArray[np.generic] | None,
    ) -> tuple[DataSplit, ...]:
        if y is not None:
            return splitter.split(X, y)
        if isinstance(splitter, LabelFreeSplitter):
            return splitter.split(X)
        raise ValueError(f"y is required by {type(splitter).__name__}")
