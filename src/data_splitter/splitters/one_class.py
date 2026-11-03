from dataclasses import dataclass
from typing import cast, override

import numpy as np
from numpy.typing import NDArray

from ..data_split import DataSplit
from ..splitter import DatasetSplitter
from ..types import BoolArray, IndexArray


@dataclass(frozen=True)
class OneClassSplitter(DatasetSplitter):
    """
    Keeps only one class in training, for one-class and anomaly detection models.

    Test partitions stay untouched, so they still measure how well other classes
    are told apart.

    Attributes
    ----------
    base : DatasetSplitter
        The strategy that makes the partitions.
    training_label : bool | int | str
        The class kept in training, such as the label of normal samples.
    """

    base: DatasetSplitter
    training_label: bool | int | str = False

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic]
    ) -> tuple[DataSplit, ...]:
        """
        Partition samples, then remove other classes from training.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic]
            The class labels.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per partition of the base strategy.

        Raises
        ------
        ValueError
            If a partition has no training samples of the class, or for the
            reasons of the base strategy.
        """
        is_kept: BoolArray = cast(BoolArray, y == self.training_label)
        splits: list[DataSplit] = []
        for split in self.base.split(X, y):
            train_indices: IndexArray = split.train_indices[
                is_kept[split.train_indices]
            ]
            if train_indices.size == 0:
                raise ValueError(
                    f"a partition has no training samples of class {self.training_label!r}"
                )
            splits.append(
                DataSplit(train_indices=train_indices, test_indices=split.test_indices)
            )
        return tuple(splits)
