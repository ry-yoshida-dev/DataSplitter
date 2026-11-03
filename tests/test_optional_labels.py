from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from data_splitter import (
    DatasetSplitter,
    DataSplit,
    GroupKFoldSplitter,
    GroupTimeSeriesSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    LabelFreeSplitter,
    LeaveOneGroupOutSplitter,
    LeaveOneOutSplitter,
    LeavePGroupsOutSplitter,
    LeavePOutSplitter,
    OneClassSplitter,
    PredefinedSplitter,
    PurgedKFoldSplitter,
    RepeatedGroupHoldoutSplitter,
    RepeatedHoldoutSplitter,
    RepeatedKFoldSplitter,
    RepeatedStratifiedKFoldSplitter,
    StratifiedGroupKFoldSplitter,
    StratifiedHoldoutSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
)

GROUPS: NDArray[np.int64] = np.repeat(np.arange(4, dtype=np.int64), 3)
UNLABELED_STRATEGIES: tuple[LabelFreeSplitter, ...] = (
    KFoldSplitter(n_splits=3),
    HoldoutSplitter(),
    RepeatedKFoldSplitter(n_splits=3, n_repeats=2),
    RepeatedHoldoutSplitter(n_repeats=2),
    LeaveOneOutSplitter(),
    LeavePOutSplitter(p=2),
    GroupKFoldSplitter(groups=GROUPS, n_splits=2),
    LeaveOneGroupOutSplitter(groups=GROUPS),
    LeavePGroupsOutSplitter(groups=GROUPS, n_groups=2),
    RepeatedGroupHoldoutSplitter(groups=GROUPS, n_repeats=2),
    PredefinedSplitter(test_folds=np.array([0] * 6 + [-1] * 6)),
    TimeSeriesSplitter(n_splits=3),
    PurgedKFoldSplitter(n_splits=3),
    GroupTimeSeriesSplitter(groups=GROUPS, n_splits=2),
)


class TestOptionalLabels(unittest.TestCase):
    def test_strategies_split_unlabeled_samples_of_any_shape(self) -> None:
        images: NDArray[np.uint8] = np.zeros((12, 4, 4, 3), dtype=np.uint8)
        for splitter in UNLABELED_STRATEGIES:
            with self.subTest(splitter=type(splitter).__name__):
                splits: tuple[DataSplit, ...] = splitter.split(images)
                self.assertGreater(len(splits), 0)
                for split in splits:
                    self.assertEqual(
                        np.intersect1d(split.train_indices, split.test_indices).size, 0
                    )

    def test_strategies_that_need_labels_are_not_label_free(self) -> None:
        for splitter in (
            StratifiedKFoldSplitter(),
            StratifiedHoldoutSplitter(),
            RepeatedStratifiedKFoldSplitter(),
            StratifiedGroupKFoldSplitter(groups=GROUPS),
            OneClassSplitter(base=KFoldSplitter()),
        ):
            with self.subTest(splitter=type(splitter).__name__):
                self.assertIsInstance(splitter, DatasetSplitter)
                self.assertNotIsInstance(splitter, LabelFreeSplitter)
