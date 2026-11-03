from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import PredefinedSplit

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter


@dataclass(frozen=True, eq=False)
class PredefinedSplitter(LabelFreeSplitter):
    """
    Partitions fixed in advance, such as an official validation set.

    Attributes
    ----------
    test_folds : NDArray[np.integer]
        Test fold of each sample, with shape (n_samples,). Samples with -1 are
        always used for training.
    """

    test_folds: NDArray[np.integer]

    def __post_init__(self) -> None:
        object.__setattr__(self, "test_folds", np.asarray(self.test_folds))
        if not np.issubdtype(self.test_folds.dtype, np.integer):
            raise TypeError(
                f"test_folds must contain integers, got dtype {self.test_folds.dtype}"
            )
        if np.any(self.test_folds < -1):
            raise ValueError("test_folds must not contain values below -1")
        folds: NDArray[np.integer] = np.unique(self.test_folds)
        num_test_folds: int = int(np.count_nonzero(folds >= 0))
        if num_test_folds == 0:
            raise ValueError("test_folds must contain at least one test fold")
        if folds.size == 1:
            raise ValueError("test_folds must leave samples for training")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into the predefined folds.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per test fold.

        Raises
        ------
        ValueError
            If test_folds does not have one entry per sample.
        """
        if self.test_folds.shape != (len(X),):
            raise ValueError(
                f"test_folds must have shape ({len(X)},), got {self.test_folds.shape}"
            )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in PredefinedSplit(self.test_folds).split()
        )
