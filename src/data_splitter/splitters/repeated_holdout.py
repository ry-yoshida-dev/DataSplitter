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
class RepeatedHoldoutSplitter(LabelFreeSplitter):
    """
    Independently drawn random train/test partitions.

    A sample can be tested any number of times, including never, so test
    partitions may overlap.

    Attributes
    ----------
    n_repeats : int
        Number of partitions.
    test_fraction : float
        Fraction of the samples in each test partition.
    random_state : int | None
        Seed for the partitions.
    """

    n_repeats: int = 10
    test_fraction: float = 0.2
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        if self.n_repeats <= 0:
            raise ValueError("n_repeats must be greater than 0")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in the range (0, 1)")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset repeatedly at random.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per repeat.
        """
        splitter: ShuffleSplit = ShuffleSplit(
            n_splits=self.n_repeats,
            test_size=self.test_fraction,
            random_state=self.random_state,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X)
        )
