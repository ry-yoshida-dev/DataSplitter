from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .index_validator import IndexValidator
from .types import IndexArray


@dataclass(frozen=True, eq=False)
class ValidatedSplit:
    """
    Sample indices of one train/validation/test partition.

    Attributes
    ----------
    train_indices : IndexArray
        Indices of the training samples.
    validation_indices : IndexArray
        Indices of the validation samples.
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
    validation_indices: IndexArray
    test_indices: IndexArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "train_indices",
            IndexValidator.to_read_only(self.train_indices, "train_indices"),
        )
        object.__setattr__(
            self,
            "validation_indices",
            IndexValidator.to_read_only(self.validation_indices, "validation_indices"),
        )
        object.__setattr__(
            self,
            "test_indices",
            IndexValidator.to_read_only(self.test_indices, "test_indices"),
        )

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

    def select_validation[ScalarT: np.generic](
        self, values: NDArray[ScalarT]
    ) -> NDArray[ScalarT]:
        """
        Select the validation samples of an array.

        Parameters
        ----------
        values : NDArray[ScalarT]
            The array with samples along its first axis.

        Returns
        -------
        NDArray[ScalarT]
            The validation samples.
        """
        return values[self.validation_indices]

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
