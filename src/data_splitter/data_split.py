from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .index_validator import IndexValidator
from .types import IndexArray


@dataclass(frozen=True, eq=False)
class DataSplit:
    """
    Sample indices of one train/test partition.

    The indices are read-only, so a split cannot change after it was made.

    Attributes
    ----------
    train_indices : IndexArray
        Indices of the training samples.
    test_indices : IndexArray
        Indices of the test samples.

    Raises
    ------
    TypeError
        If indices are not integers.
    ValueError
        If indices are not 1-D.
    """

    train_indices: IndexArray
    test_indices: IndexArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "train_indices",
            IndexValidator.to_read_only(self.train_indices, "train_indices"),
        )
        object.__setattr__(
            self,
            "test_indices",
            IndexValidator.to_read_only(self.test_indices, "test_indices"),
        )

    @property
    def num_train(self) -> int:
        """
        Get the number of training samples.

        Returns
        -------
        int
            The number of training samples.
        """
        return self.train_indices.size

    @property
    def num_test(self) -> int:
        """
        Get the number of test samples.

        Returns
        -------
        int
            The number of test samples.
        """
        return self.test_indices.size

    def select_train[ScalarT: np.generic](
        self, values: NDArray[ScalarT]
    ) -> NDArray[ScalarT]:
        """
        Select the training samples of an array.

        Parameters
        ----------
        values : NDArray[ScalarT]
            The array with samples along its first axis.

        Returns
        -------
        NDArray[ScalarT]
            The training samples.
        """
        return values[self.train_indices]

    def select_test[ScalarT: np.generic](
        self, values: NDArray[ScalarT]
    ) -> NDArray[ScalarT]:
        """
        Select the test samples of an array.

        Parameters
        ----------
        values : NDArray[ScalarT]
            The array with samples along its first axis.

        Returns
        -------
        NDArray[ScalarT]
            The test samples.
        """
        return values[self.test_indices]
