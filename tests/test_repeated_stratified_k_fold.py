from __future__ import annotations

import unittest
from typing import cast

import numpy as np

from data_splitter import (
    DataSplit,
    RepeatedStratifiedKFoldSplitter,
)
from tests.data import IndexArray, LabelArray, NumericArray, separable_blobs


class TestRepeatedStratifiedKFoldSplitter(unittest.TestCase):
    def test_each_repeat_covers_all_samples_once(self) -> None:
        X, y = separable_blobs()
        splits: tuple[DataSplit, ...] = RepeatedStratifiedKFoldSplitter(
            n_splits=3, n_repeats=2
        ).split(X, y)
        self.assertEqual(len(splits), 6)
        for repeat in (splits[:3], splits[3:]):
            test_indices: IndexArray = np.concatenate(
                [split.test_indices for split in repeat]
            )
            self.assertEqual(np.sort(test_indices).tolist(), list(range(y.size)))

    def test_folds_preserve_class_proportions(self) -> None:
        X, y = separable_blobs()
        for split in RepeatedStratifiedKFoldSplitter(n_splits=3, n_repeats=2).split(
            X, y
        ):
            self.assertEqual(np.bincount(y[split.test_indices]).tolist(), [10, 10, 10])

    def test_repeats_differ_and_are_reproducible(self) -> None:
        X, y = separable_blobs()
        first: tuple[DataSplit, ...] = RepeatedStratifiedKFoldSplitter(
            n_splits=3, n_repeats=2, random_state=1
        ).split(X, y)
        second: tuple[DataSplit, ...] = RepeatedStratifiedKFoldSplitter(
            n_splits=3, n_repeats=2, random_state=1
        ).split(X, y)
        np.testing.assert_array_equal(first[0].test_indices, second[0].test_indices)
        self.assertFalse(np.array_equal(first[0].test_indices, first[3].test_indices))

    def test_rare_class_raises(self) -> None:
        X: NumericArray = np.zeros((7, 1))
        y: LabelArray = np.array([0, 0, 0, 0, 0, 1, 1])
        with self.assertRaises(ValueError):
            _ = RepeatedStratifiedKFoldSplitter(n_splits=3).split(X, y)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = RepeatedStratifiedKFoldSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = RepeatedStratifiedKFoldSplitter(n_repeats=0)
        with self.assertRaises(TypeError):
            _ = RepeatedStratifiedKFoldSplitter(n_repeats=cast(int, 2.5))


if __name__ == "__main__":
    _ = unittest.main()
