from dataclasses import dataclass
from typing import cast, override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import TimeSeriesSplit

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter
from ..parameter_validator import ParameterValidator
from ..types import BoolArray, GroupArray, IndexArray


@dataclass(frozen=True, eq=False)
class GroupTimeSeriesSplitter(LabelFreeSplitter):
    """
    Forward-chaining folds over consecutive groups, such as the frames of video clips.

    Each group stays whole on one side, and training groups always precede test
    groups. Samples must be in temporal order, so that every group is one consecutive
    run.

    Attributes
    ----------
    groups : GroupArray
        Group identifier of each sample, with shape (n_samples,).
    n_splits : int
        Number of folds.
    gap : int
        Number of groups excluded between training and test.
    """

    groups: GroupArray
    n_splits: int = 5
    gap: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", np.asarray(self.groups))
        ParameterValidator.require_group_identifiers(self.groups)
        ParameterValidator.require_integers(n_splits=self.n_splits, gap=self.gap)
        if self.groups.ndim != 1 or self.groups.size == 0:
            raise ValueError(
                f"groups must be non-empty and 1-D, got shape {self.groups.shape}"
            )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition samples into forward-chaining folds of whole groups.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples in temporal order; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold, oldest first.

        Raises
        ------
        ValueError
            If groups does not have one entry per sample, a group is not one
            consecutive run, or there are too few groups for the folds.
        """
        if self.groups.shape != (len(X),):
            raise ValueError(
                f"groups must have shape ({len(X)},), got {self.groups.shape}"
            )
        is_new_run: BoolArray = cast(BoolArray, self.groups[1:] != self.groups[:-1])
        run_of_sample: IndexArray = np.concatenate(
            ([0], np.cumsum(is_new_run, dtype=np.intp))
        )
        num_runs: int = run_of_sample.item(-1) + 1
        if num_runs != np.unique(self.groups).size:
            raise ValueError("each group must be one consecutive run of samples")
        splitter: TimeSeriesSplit = TimeSeriesSplit(
            n_splits=self.n_splits, gap=self.gap
        )
        run_indices: IndexArray = np.arange(num_runs, dtype=np.intp)
        splits: list[DataSplit] = []
        for train_runs, test_runs in splitter.split(run_indices):
            # Runs are ordered, so sample positions follow from run boundaries.
            train_stop: int = int(
                np.searchsorted(run_of_sample, train_runs.item(-1), side="right")
            )
            test_start: int = int(
                np.searchsorted(run_of_sample, test_runs.item(0), side="left")
            )
            test_stop: int = int(
                np.searchsorted(run_of_sample, test_runs.item(-1), side="right")
            )
            splits.append(
                DataSplit(
                    train_indices=np.arange(train_stop, dtype=np.intp),
                    test_indices=np.arange(test_start, test_stop, dtype=np.intp),
                )
            )
        return tuple(splits)
