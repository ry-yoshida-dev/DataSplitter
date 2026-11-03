import math
from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import LeavePOut

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator


@dataclass(frozen=True)
class LeavePOutSplitter(LabelFreeSplitter):
    """
    Cross-validation that tests on every combination of a number of samples.

    Test partitions overlap, so a sample is tested several times, and the number
    of folds grows combinatorially; ``max_splits`` stops it before any fold is
    built. With ``p`` of 1, prefer LeaveOneOutSplitter.

    Attributes
    ----------
    p : int
        Number of samples in each test partition.
    max_splits : int
        Largest number of folds allowed.
    """

    p: int = 2
    max_splits: int = 10000

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            p=self.p,
            max_splits=self.max_splits,
        )
        if self.p <= 0:
            raise ValueError("p must be greater than 0")
        if self.max_splits <= 0:
            raise ValueError("max_splits must be greater than 0")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one fold per combination of samples.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per combination.

        Raises
        ------
        ValueError
            If there are not more samples than p, or if the folds would exceed
            max_splits.
        """
        num_splits: int = math.comb(len(X), self.p)
        if num_splits > self.max_splits:
            # The count itself can exceed Python's integer-to-string digit limit.
            raise ValueError(
                f"leaving {self.p} of {len(X)} samples out exceeds max_splits={self.max_splits}"
            )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in LeavePOut(p=self.p).split(X)
        )
