from __future__ import annotations

import numpy as np

from .types import IndexArray


class IndexValidator:
    """
    Runtime checks of sample indices.
    """

    @staticmethod
    def to_read_only(indices: IndexArray, name: str) -> IndexArray:
        """
        Check indices and return them in a read-only view.

        Parameters
        ----------
        indices : IndexArray
            The sample indices.
        name : str
            The name used in error messages.

        Returns
        -------
        IndexArray
            The read-only indices.

        Raises
        ------
        TypeError
            If the indices are not integers.
        ValueError
            If the indices are not 1-D.
        """
        array: IndexArray = np.asarray(indices)
        if array.size > 0 and not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must contain integers, got dtype {array.dtype}")
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got {array.ndim} dimensions")
        view: IndexArray = array.astype(np.intp, copy=False).view()
        view.setflags(write=False)
        return view
