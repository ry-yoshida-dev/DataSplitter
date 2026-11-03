import math
from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import LeavePGroupsOut

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator
from ..types import GroupArray


@dataclass(frozen=True, eq=False)
class LeavePGroupsOutSplitter(LabelFreeSplitter):
    """
    Cross-validation that tests on every combination of a number of groups.

    Test partitions overlap, so a sample is tested several times, and the number
    of folds grows combinatorially; ``max_splits`` stops it before any fold is
    built.

    Attributes
    ----------
    groups : GroupArray
        Group identifier of each sample, with shape (n_samples,).
    n_groups : int
        Number of groups in each test partition.
    max_splits : int
        Largest number of folds allowed.
    """

    groups: GroupArray
    n_groups: int = 2
    max_splits: int = 10000

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", np.asarray(self.groups))
        ParameterValidator.require_group_identifiers(self.groups)
        ParameterValidator.require_integers(
            n_groups=self.n_groups,
            max_splits=self.max_splits,
        )
        if self.n_groups <= 0:
            raise ValueError("n_groups must be greater than 0")
        if self.max_splits <= 0:
            raise ValueError("max_splits must be greater than 0")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one fold per combination of groups.

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
            If groups does not have one entry per sample, if there are not more
            groups than n_groups, or if the folds would exceed max_splits.
        """
        if self.groups.shape != (len(X),):
            raise ValueError(
                f"groups must have shape ({len(X)},), got {self.groups.shape}"
            )
        num_unique_groups: int = np.unique(self.groups).size
        num_splits: int = math.comb(num_unique_groups, self.n_groups)
        if num_splits > self.max_splits:
            # The count itself can exceed Python's integer-to-string digit limit.
            raise ValueError(
                f"leaving {self.n_groups} of {num_unique_groups} groups out exceeds max_splits={self.max_splits}"
            )
        splitter: LeavePGroupsOut = LeavePGroupsOut(n_groups=self.n_groups)
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X, y, self.groups)
        )
