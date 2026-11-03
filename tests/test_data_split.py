from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from data_splitter import DataSplit
from tests.data import IndexArray


class TestDataSplit(unittest.TestCase):
    def test_indices_are_read_only(self) -> None:
        split: DataSplit = DataSplit(np.array([0, 1]), np.array([2]))
        with self.assertRaises(ValueError):
            split.train_indices[0] = 5

    def test_source_array_stays_writable(self) -> None:
        train: IndexArray = np.array([0, 1], dtype=np.intp)
        _ = DataSplit(train, np.array([2]))
        train[0] = 5
        self.assertEqual(train.item(0), 5)

    def test_counts_and_selection(self) -> None:
        split: DataSplit = DataSplit(np.array([2, 0]), np.array([1]))
        values: NDArray[np.float64] = np.array([10.0, 20.0, 30.0])
        self.assertEqual((split.num_train, split.num_test), (2, 1))
        self.assertEqual(split.select_train(values).tolist(), [30.0, 10.0])
        self.assertEqual(split.select_test(values).tolist(), [20.0])

    def test_selects_along_the_first_axis_of_any_shape(self) -> None:
        images: NDArray[np.uint8] = np.zeros((3, 4, 4), dtype=np.uint8)
        split: DataSplit = DataSplit(np.array([0, 1]), np.array([2]))
        self.assertEqual(split.select_train(images).shape, (2, 4, 4))

    def test_rejects_invalid_indices(self) -> None:
        with self.assertRaises(TypeError):
            _ = DataSplit(cast(IndexArray, np.array([0.5])), np.array([1]))
        with self.assertRaises(ValueError):
            _ = DataSplit(np.array([[0, 1]]), np.array([2]))
