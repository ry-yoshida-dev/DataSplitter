from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

class StratifiedShuffleSplit:
    def __init__(
        self,
        n_splits: int = 10,
        *,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class StratifiedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class StratifiedGroupKFold:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike | None, groups: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class TimeSeriesSplit:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
    ) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class ShuffleSplit:
    def __init__(
        self,
        n_splits: int = 10,
        *,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class KFold:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class GroupKFold:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike | None, groups: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class RepeatedKFold:
    def __init__(
        self,
        *,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class RepeatedStratifiedKFold:
    def __init__(
        self,
        *,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class LeaveOneOut:
    def __init__(self) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class LeaveOneGroupOut:
    def __init__(self) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike | None, groups: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class LeavePGroupsOut:
    def __init__(self, n_groups: int) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike | None, groups: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class LeavePOut:
    def __init__(self, p: int) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class PredefinedSplit:
    def __init__(self, test_fold: ArrayLike) -> None: ...
    def split(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class GroupShuffleSplit:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike | None, groups: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...
