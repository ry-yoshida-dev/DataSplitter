from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import LeaveOneGroupOut

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator
from ..types import GroupArray


@dataclass(frozen=True, eq=False)
class LeaveOneGroupOutSplitter(LabelFreeSplitter):
    """
    Cross-validation that tests on each group in turn.

    Use this to estimate performance on an unseen group, such as a new subject.
    The number of folds equals the number of groups.

    Attributes
    ----------
    groups : GroupArray
        Group identifier of each sample, with shape (n_samples,).
    """

    groups: GroupArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", np.asarray(self.groups))
        ParameterValidator.require_group_identifiers(self.groups)

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one fold per group.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per group.

        Raises
        ------
        ValueError
            If groups does not have one entry per sample or has a single value.
        """
        if self.groups.shape != (len(X),):
            raise ValueError(
                f"groups must have shape ({len(X)},), got {self.groups.shape}"
            )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in LeaveOneGroupOut().split(X, y, self.groups)
        )
