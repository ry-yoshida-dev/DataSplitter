from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import ShuffleSplit

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator
from ..random_state import DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class HoldoutSplitter(LabelFreeSplitter):
    """
    Single random train/test partition.

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
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one train/test split.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            The single split.
        """
        splitter: ShuffleSplit = ShuffleSplit(
            n_splits=1,
            test_size=self.test_fraction,
            random_state=self.random_state,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X)
        )
