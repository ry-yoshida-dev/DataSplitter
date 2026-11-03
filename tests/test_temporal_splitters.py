from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from data_splitter import DataSplit, GroupTimeSeriesSplitter, PurgedKFoldSplitter
from tests.data import GroupArray, IndexArray

X: NDArray[np.float64] = np.zeros((20, 2))


class TestPurgedKFoldSplitter(unittest.TestCase):
    def test_tests_every_sample_once_in_order(self) -> None:
        splits: tuple[DataSplit, ...] = PurgedKFoldSplitter(n_splits=4).split(X)
        tested: IndexArray = np.concatenate([split.test_indices for split in splits])
        self.assertEqual(tested.tolist(), list(range(20)))

    def test_removes_training_samples_within_the_gap_on_both_sides(self) -> None:
        split: DataSplit = PurgedKFoldSplitter(n_splits=4, gap=2).split(X)[1]
        self.assertEqual(split.test_indices.tolist(), [5, 6, 7, 8, 9])
        self.assertEqual(split.train_indices.tolist(), [0, 1, 2, *range(12, 20)])

    def test_zero_gap_trains_on_everything_else(self) -> None:
        for split in PurgedKFoldSplitter(n_splits=4).split(X):
            self.assertEqual(split.num_train + split.num_test, 20)

    def test_rejects_too_few_samples_and_excessive_gaps(self) -> None:
        with self.assertRaises(ValueError):
            _ = PurgedKFoldSplitter(n_splits=5).split(np.zeros((4, 1)))
        with self.assertRaisesRegex(ValueError, "without training samples"):
            _ = PurgedKFoldSplitter(n_splits=2, gap=20).split(X)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            _ = PurgedKFoldSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = PurgedKFoldSplitter(gap=-1)
        with self.assertRaises(TypeError):
            _ = PurgedKFoldSplitter(gap=cast(int, 1.5))


class TestGroupTimeSeriesSplitter(unittest.TestCase):
    def test_groups_stay_whole_and_training_precedes_testing(self) -> None:
        groups: NDArray[np.int64] = np.repeat(np.arange(5, dtype=np.int64), 4)
        for split in GroupTimeSeriesSplitter(groups=groups, n_splits=3).split(X):
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )
            self.assertLess(split.train_indices.max(), split.test_indices.min())
            self.assertTrue(
                np.array_equal(np.sort(split.train_indices), split.train_indices)
            )
            self.assertEqual(split.num_test % 4, 0)

    def test_handles_groups_of_different_sizes_and_string_identifiers(self) -> None:
        groups: GroupArray = np.array(["b"] * 3 + ["a"] * 2 + ["d"] * 1 + ["c"] * 4)
        splits: tuple[DataSplit, ...] = GroupTimeSeriesSplitter(
            groups=groups, n_splits=3
        ).split(np.zeros((10, 1)))
        self.assertEqual(splits[0].train_indices.tolist(), [0, 1, 2])
        self.assertEqual(splits[0].test_indices.tolist(), [3, 4])
        self.assertEqual(splits[2].test_indices.tolist(), [6, 7, 8, 9])

    def test_gap_excludes_whole_groups(self) -> None:
        groups: NDArray[np.int64] = np.repeat(np.arange(5, dtype=np.int64), 4)
        split: DataSplit = GroupTimeSeriesSplitter(
            groups=groups, n_splits=2, gap=1
        ).split(X)[-1]
        self.assertEqual(np.unique(groups[split.test_indices]).tolist(), [4])
        self.assertEqual(int(groups[split.train_indices].max()), 2)

    def test_rejects_groups_that_are_not_one_run(self) -> None:
        groups: NDArray[np.int64] = np.array(
            [0, 0, 1, 1, 0, 2, 2, 3, 3, 4], dtype=np.int64
        )
        with self.assertRaisesRegex(ValueError, "consecutive run"):
            _ = GroupTimeSeriesSplitter(groups=groups, n_splits=2).split(
                np.zeros((10, 1))
            )

    def test_rejects_mismatched_and_invalid_groups(self) -> None:
        with self.assertRaises(ValueError):
            _ = GroupTimeSeriesSplitter(groups=np.arange(5)).split(X)
        with self.assertRaises(ValueError):
            _ = GroupTimeSeriesSplitter(groups=np.zeros((0,), dtype=np.int64))
        with self.assertRaises(TypeError):
            _ = GroupTimeSeriesSplitter(groups=cast(GroupArray, np.zeros(5)))

    def test_rejects_too_few_groups(self) -> None:
        with self.assertRaises(ValueError):
            _ = GroupTimeSeriesSplitter(
                groups=np.repeat(np.arange(2), 10), n_splits=5
            ).split(X)
