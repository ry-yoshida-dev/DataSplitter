from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold

from ..data_split import DataSplit
from ..parameter_validator import ParameterValidator
from ..random_state import DEFAULT_RANDOM_STATE
from ..splitter import DatasetSplitter
from ..types import GroupArray


@dataclass(frozen=True, eq=False)
class StratifiedGroupKFoldSplitter(DatasetSplitter):
    """
    K-fold cross-validation that never splits a group across train and test.

    Use this when samples are correlated within groups, such as several images
    of one subject; random splitting would otherwise leak information.

    Attributes
    ----------
    groups : GroupArray
        Group identifier of each sample, with shape (n_samples,).
    n_splits : int
        Number of folds.
    is_shuffled : bool
        Whether groups are shuffled before folding.
    random_state : int | None
        Seed for the shuffle; ignored when is_shuffled is False.

    Raises
    ------
    TypeError
        If groups does not contain integer or string identifiers.
    """

    groups: GroupArray
    n_splits: int = 5
    is_shuffled: bool = True
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", np.asarray(self.groups))
        ParameterValidator.require_group_identifiers(self.groups)
        ParameterValidator.require_integers(
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic]
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into group-preserving folds.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic]
            The class labels.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold.

        Raises
        ------
        ValueError
            If groups does not have one entry per sample.
        """
        if self.groups.shape != (len(X),):
            raise ValueError(
                f"groups must have shape ({len(X)},), got {self.groups.shape}"
            )
        splitter: StratifiedGroupKFold = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.is_shuffled,
            random_state=self.random_state if self.is_shuffled else None,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X, y, self.groups)
        )
