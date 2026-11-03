from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from data_splitter import (
    DataSplit,
    HoldoutSplitter,
    KFoldSplitter,
    OneClassSplitter,
    StratifiedKFoldSplitter,
)

X: NDArray[np.float64] = np.zeros((40, 2))
IS_ANOMALOUS: NDArray[np.bool_] = np.tile(np.array([True, False, False, False]), 10)


class TestOneClassSplitter(unittest.TestCase):
    def test_training_holds_only_the_class_and_tests_stay_untouched(self) -> None:
        base: StratifiedKFoldSplitter = StratifiedKFoldSplitter(n_splits=5)
        splits: tuple[DataSplit, ...] = OneClassSplitter(base=base).split(
            X, IS_ANOMALOUS
        )
        for split, base_split in zip(splits, base.split(X, IS_ANOMALOUS), strict=True):
            self.assertFalse(split.select_train(IS_ANOMALOUS).any())
            np.testing.assert_array_equal(split.test_indices, base_split.test_indices)
            self.assertTrue(split.select_test(IS_ANOMALOUS).any())

    def test_keeps_a_chosen_integer_or_string_label(self) -> None:
        integers: NDArray[np.int64] = np.arange(40, dtype=np.int64) % 2
        split: DataSplit = OneClassSplitter(
            base=HoldoutSplitter(), training_label=1
        ).split(X, integers)[0]
        self.assertEqual(np.unique(split.select_train(integers)).tolist(), [1])
        names: NDArray[np.str_] = np.array(["ok", "bad"] * 20)
        named: DataSplit = OneClassSplitter(
            base=HoldoutSplitter(), training_label="ok"
        ).split(X, names)[0]
        self.assertEqual(np.unique(named.select_train(names)).tolist(), ["ok"])

    def test_rejects_partitions_without_the_class(self) -> None:
        all_anomalous: NDArray[np.bool_] = np.ones(40, dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "no training samples"):
            _ = OneClassSplitter(base=KFoldSplitter()).split(X, all_anomalous)
