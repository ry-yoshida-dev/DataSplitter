from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import LeaveOneOut

from ..data_split import DataSplit
from ..label_free_splitter import LabelFreeSplitter


@dataclass(frozen=True)
class LeaveOneOutSplitter(LabelFreeSplitter):
    """
    Cross-validation that tests on each sample in turn.

    Each test partition has one sample, so metrics that need several samples or
    classes are undefined in every fold.
    """

    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one fold per sample.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per sample.

        Raises
        ------
        ValueError
            If the dataset has fewer than two samples.
        """
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in LeaveOneOut().split(X)
        )
