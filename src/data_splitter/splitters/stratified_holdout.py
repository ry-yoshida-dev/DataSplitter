from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedShuffleSplit

from ..data_split import DataSplit
from ..parameter_validator import ParameterValidator
from ..random_state import DEFAULT_RANDOM_STATE
from ..splitter import DatasetSplitter


@dataclass(frozen=True)
class StratifiedHoldoutSplitter(DatasetSplitter):
    """
    Single random train/test partition that preserves class proportions.

    Attributes
    ----------
    test_fraction : float
        Fraction of the samples in the test partition.
    random_state : int | None
        Seed for the partition.
    """

    test_fraction: float = 0.2
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(random_state=self.random_state)
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in the range (0, 1)")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic]
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one stratified train/test split.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic]
            The class labels.

        Returns
        -------
        tuple[DataSplit, ...]
            The single split.
        """
        splitter: StratifiedShuffleSplit = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_fraction,
            random_state=self.random_state,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X, y)
        )
