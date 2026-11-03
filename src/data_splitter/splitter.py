from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .data_split import DataSplit


class DatasetSplitter(ABC):
    """
    Base class for strategies that partition samples into train and test indices.

    Code that always has labels depends on this class. Strategies that do not need
    labels derive from LabelFreeSplitter instead, which also accepts unlabeled data.
    """

    @abstractmethod
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic]
    ) -> tuple[DataSplit, ...]:
        """
        Partition samples into train and test indices.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples; only their number is used.
        y : NDArray[np.generic]
            The targets.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per partition.
        """
