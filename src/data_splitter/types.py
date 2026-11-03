"""Type aliases for sample indices, counts, and group identifiers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type BoolArray = NDArray[np.bool_]
type IndexArray = NDArray[np.intp]
type CountArray = NDArray[np.int64]
type GroupArray = NDArray[np.integer | np.str_]
