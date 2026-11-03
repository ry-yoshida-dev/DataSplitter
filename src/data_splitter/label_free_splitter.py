from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import numpy as np
from numpy.typing import NDArray

from .data_split import DataSplit
from .splitter import DatasetSplitter


class LabelFreeSplitter(DatasetSplitter, ABC):
    """
    Base class for strategies that also work on unlabeled samples.
    """

    @abstractmethod
    @override
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic] | None = None
    ) -> tuple[DataSplit, ...]:
        """
        Partition samples into train and test indices.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic] | None
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per partition.
        """
