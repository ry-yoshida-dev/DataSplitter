from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator
from ..random_state import DEFAULT_RANDOM_STATE
from ..types import GroupArray


@dataclass(frozen=True, eq=False)
class RepeatedGroupHoldoutSplitter(LabelFreeSplitter):
    """
    Independently drawn random partitions that never split a group.

    The test fraction applies to groups, so the number of test samples varies
    when groups differ in size. A group can be tested any number of times,
    including never.

    Attributes
    ----------
    groups : GroupArray
        Group identifier of each sample, with shape (n_samples,).
    n_repeats : int
        Number of partitions.
    test_fraction : float
        Fraction of the groups in each test partition.
    random_state : int | None
        Seed for the partitions.
    """

    groups: GroupArray
    n_repeats: int = 10
    test_fraction: float = 0.2
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", np.asarray(self.groups))
        ParameterValidator.require_group_identifiers(self.groups)
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
        Partition a dataset repeatedly at random, keeping groups together.

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

        Raises
        ------
        ValueError
            If groups does not have one entry per sample or has a single value.
        """
        if self.groups.shape != (len(X),):
            raise ValueError(
                f"groups must have shape ({len(X)},), got {self.groups.shape}"
            )
        splitter: GroupShuffleSplit = GroupShuffleSplit(
            n_splits=self.n_repeats,
            test_size=self.test_fraction,
            random_state=self.random_state,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X, y, self.groups)
        )
