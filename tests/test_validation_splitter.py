from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from data_splitter import (
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    ValidatedSplit,
    ValidationSplitter,
)
from tests.data import IndexArray

X: NDArray[np.float64] = np.zeros((60, 2))
Y: NDArray[np.int64] = np.repeat(np.array([0, 1, 2], dtype=np.int64), 20)


class TestValidationSplitter(unittest.TestCase):
    def test_partitions_are_disjoint_and_complete(self) -> None:
        split: ValidatedSplit = ValidationSplitter(
            outer=HoldoutSplitter(test_fraction=0.2),
            inner=HoldoutSplitter(test_fraction=0.25),
        ).split(X)[0]
        combined: IndexArray = np.concatenate(
            (split.train_indices, split.validation_indices, split.test_indices)
        )
        np.testing.assert_array_equal(np.sort(combined), np.arange(60))
        self.assertEqual(
            (
                split.train_indices.size,
                split.validation_indices.size,
                split.test_indices.size,
            ),
            (36, 12, 12),
        )

    def test_nested_cross_validation_multiplies_the_folds(self) -> None:
        splits: tuple[ValidatedSplit, ...] = ValidationSplitter(
            outer=KFoldSplitter(n_splits=5), inner=KFoldSplitter(n_splits=3)
        ).split(X)
        self.assertEqual(len(splits), 15)
        for split in splits:
            self.assertEqual(
                np.intersect1d(split.validation_indices, split.test_indices).size, 0
            )
            self.assertEqual(
                np.intersect1d(split.train_indices, split.test_indices).size, 0
            )

    def test_stratified_inner_strategy_sees_the_labels_of_remaining_samples(
        self,
    ) -> None:
        splits: tuple[ValidatedSplit, ...] = ValidationSplitter(
            outer=StratifiedKFoldSplitter(n_splits=5),
            inner=StratifiedKFoldSplitter(n_splits=2),
        ).split(X, Y)
        for split in splits:
            counts: NDArray[np.int64] = np.bincount(split.select_validation(Y))
            self.assertEqual(int(counts.max() - counts.min()), 0)

    def test_requires_labels_for_strategies_that_need_them(self) -> None:
        for splitter in (
            ValidationSplitter(outer=StratifiedKFoldSplitter(), inner=KFoldSplitter()),
            ValidationSplitter(outer=KFoldSplitter(), inner=StratifiedKFoldSplitter()),
        ):
            with (
                self.subTest(splitter=splitter),
                self.assertRaisesRegex(ValueError, "y is required"),
            ):
                _ = splitter.split(X)

    def test_temporal_order_is_kept_on_both_levels(self) -> None:
        splits: tuple[ValidatedSplit, ...] = ValidationSplitter(
            outer=TimeSeriesSplitter(n_splits=2), inner=TimeSeriesSplitter(n_splits=2)
        ).split(X)
        for split in splits:
            self.assertLess(split.train_indices.max(), split.validation_indices.min())
            self.assertLess(split.validation_indices.max(), split.test_indices.min())

    def test_group_strategy_works_as_outer_but_not_as_inner(self) -> None:
        groups: NDArray[np.int64] = np.arange(60, dtype=np.int64) % 6
        grouped: GroupKFoldSplitter = GroupKFoldSplitter(groups=groups, n_splits=3)
        outer: tuple[ValidatedSplit, ...] = ValidationSplitter(
            outer=grouped, inner=HoldoutSplitter()
        ).split(X)
        self.assertEqual(len(outer), 3)
        with self.assertRaisesRegex(ValueError, "groups must have shape"):
            _ = ValidationSplitter(outer=KFoldSplitter(), inner=grouped).split(X)

    def test_indices_are_read_only(self) -> None:
        split: ValidatedSplit = ValidationSplitter(
            outer=HoldoutSplitter(), inner=HoldoutSplitter()
        ).split(X)[0]
        with self.assertRaises(ValueError):
            split.validation_indices[0] = 0
