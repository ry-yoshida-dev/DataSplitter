from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedKFold

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator
from ..random_state import DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class RepeatedKFoldSplitter(LabelFreeSplitter):
    """
    K-fold cross-validation repeated with a different shuffle each time.

    Every sample is tested once per repeat, so test partitions overlap across
    repeats and the standard deviation over folds is not that of independent runs.

    Attributes
    ----------
    n_splits : int
        Number of folds per repeat.
    n_repeats : int
        Number of repeats.
    random_state : int | None
        Seed for the shuffles.
    """

    n_splits: int = 5
    n_repeats: int = 10
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_repeats <= 0:
            raise ValueError("n_repeats must be greater than 0")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into repeated folds.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold of every repeat.
        """
        splitter: RepeatedKFold = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X)
        )
