from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.typing import NDArray

from data_splitter import (
    DataSplit,
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    LeaveOneGroupOutSplitter,
    LeaveOneOutSplitter,
    LeavePGroupsOutSplitter,
    LeavePOutSplitter,
    PredefinedSplitter,
    RepeatedGroupHoldoutSplitter,
    RepeatedHoldoutSplitter,
    RepeatedKFoldSplitter,
    TimeSeriesSplitter,
)
from tests.data import linear_data


class TestHoldoutSplitter(unittest.TestCase):
    def test_partition_sizes(self) -> None:
        X, y = linear_data(num_samples=100)
        splits: tuple[DataSplit, ...] = HoldoutSplitter(test_fraction=0.25).split(X, y)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0].test_indices.size, 25)
        self.assertEqual(splits[0].train_indices.size, 75)

    def test_invalid_fraction_raises(self) -> None:
        for test_fraction in (0.0, 1.0):
            with self.subTest(test_fraction=test_fraction):
                with self.assertRaises(ValueError):
                    _ = HoldoutSplitter(test_fraction=test_fraction)


class TestKFoldSplitter(unittest.TestCase):
    def test_test_partitions_cover_all_samples_once(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = KFoldSplitter(n_splits=3).split(X, y)
        test_indices: NDArray[np.intp] = np.concatenate(
            [split.test_indices for split in splits]
        )
        self.assertEqual(np.sort(test_indices).tolist(), list(range(y.size)))
        for split in splits:
            self.assertEqual(
                np.intersect1d(split.train_indices, split.test_indices).size, 0
            )

    def test_is_reproducible(self) -> None:
        X, y = linear_data()
        first: tuple[DataSplit, ...] = KFoldSplitter(random_state=1).split(X, y)
        second: tuple[DataSplit, ...] = KFoldSplitter(random_state=1).split(X, y)
        np.testing.assert_array_equal(first[0].test_indices, second[0].test_indices)

    def test_unshuffled_ignores_random_state(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = KFoldSplitter(
            is_shuffled=False, random_state=1
        ).split(X, y)
        self.assertEqual(len(splits), 5)

    def test_invalid_n_splits_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = KFoldSplitter(n_splits=1)


class TestGroupKFoldSplitter(unittest.TestCase):
    def test_groups_never_cross_partitions(self) -> None:
        X, y = linear_data()
        groups: NDArray[np.int64] = np.arange(y.size) % 12
        splits: tuple[DataSplit, ...] = GroupKFoldSplitter(
            groups=groups, n_splits=3
        ).split(X, y)
        for split in splits:
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )

    def test_group_length_mismatch_raises(self) -> None:
        X, y = linear_data()
        splitter: GroupKFoldSplitter = GroupKFoldSplitter(groups=np.array([0, 1]))
        with self.assertRaises(ValueError):
            _ = splitter.split(X, y)


class TestTimeSeriesSplitter(unittest.TestCase):
    def test_training_precedes_test_with_gap(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = TimeSeriesSplitter(n_splits=3, gap=2).split(
            X, y
        )
        for split in splits:
            self.assertEqual(
                int(split.train_indices.max()) + 3, int(split.test_indices.min())
            )

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(gap=-1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(test_size=0)


class TestRepeatedKFoldSplitter(unittest.TestCase):
    def test_each_repeat_covers_all_samples_once(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = RepeatedKFoldSplitter(
            n_splits=3, n_repeats=2
        ).split(X, y)
        self.assertEqual(len(splits), 6)
        for repeat in (splits[:3], splits[3:]):
            test_indices: NDArray[np.intp] = np.concatenate(
                [split.test_indices for split in repeat]
            )
            self.assertEqual(np.sort(test_indices).tolist(), list(range(y.size)))

    def test_repeats_differ_and_are_reproducible(self) -> None:
        X, y = linear_data()
        first: tuple[DataSplit, ...] = RepeatedKFoldSplitter(
            n_splits=3, n_repeats=2, random_state=1
        ).split(X, y)
        second: tuple[DataSplit, ...] = RepeatedKFoldSplitter(
            n_splits=3, n_repeats=2, random_state=1
        ).split(X, y)
        np.testing.assert_array_equal(first[0].test_indices, second[0].test_indices)
        self.assertFalse(np.array_equal(first[0].test_indices, first[3].test_indices))

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = RepeatedKFoldSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = RepeatedKFoldSplitter(n_repeats=0)
        with self.assertRaises(TypeError):
            _ = RepeatedKFoldSplitter(n_repeats=True)


class TestRepeatedHoldoutSplitter(unittest.TestCase):
    def test_partition_sizes(self) -> None:
        X, y = linear_data(num_samples=100)
        splits: tuple[DataSplit, ...] = RepeatedHoldoutSplitter(
            n_repeats=3, test_fraction=0.25
        ).split(X, y)
        self.assertEqual(len(splits), 3)
        for split in splits:
            self.assertEqual(split.test_indices.size, 25)
            self.assertEqual(split.train_indices.size, 75)
            self.assertEqual(
                np.intersect1d(split.train_indices, split.test_indices).size, 0
            )
        self.assertFalse(np.array_equal(splits[0].test_indices, splits[1].test_indices))

    def test_invalid_arguments_raise(self) -> None:
        for test_fraction in (0.0, 1.0, math.nan):
            with self.subTest(test_fraction=test_fraction):
                with self.assertRaises(ValueError):
                    _ = RepeatedHoldoutSplitter(test_fraction=test_fraction)
        with self.assertRaises(ValueError):
            _ = RepeatedHoldoutSplitter(n_repeats=0)
        with self.assertRaises(TypeError):
            _ = RepeatedHoldoutSplitter(n_repeats=True)


class TestLeaveOneGroupOutSplitter(unittest.TestCase):
    def test_tests_on_each_group_once(self) -> None:
        X, y = linear_data(num_samples=30)
        groups: NDArray[np.int64] = np.arange(30) % 5
        splits: tuple[DataSplit, ...] = LeaveOneGroupOutSplitter(groups=groups).split(
            X, y
        )
        self.assertEqual(len(splits), 5)
        for split in splits:
            self.assertEqual(np.unique(groups[split.test_indices]).size, 1)
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )
        tested: NDArray[np.intp] = np.concatenate(
            [split.test_indices for split in splits]
        )
        self.assertEqual(np.sort(tested).tolist(), list(range(30)))

    def test_invalid_groups_raise(self) -> None:
        X, y = linear_data(num_samples=4)
        with self.assertRaises(TypeError):
            _ = LeaveOneGroupOutSplitter(groups=np.array([0.5, 1.5]))
        with self.assertRaises(ValueError):
            _ = LeaveOneGroupOutSplitter(groups=np.array([0, 1])).split(X, y)
        with self.assertRaises(ValueError):
            _ = LeaveOneGroupOutSplitter(groups=np.zeros(4, dtype=np.int64)).split(X, y)


class TestLeavePGroupsOutSplitter(unittest.TestCase):
    def test_tests_on_every_combination(self) -> None:
        X, y = linear_data(num_samples=30)
        groups: NDArray[np.int64] = np.arange(30) % 5
        splits: tuple[DataSplit, ...] = LeavePGroupsOutSplitter(
            groups=groups, n_groups=2
        ).split(X, y)
        self.assertEqual(len(splits), math.comb(5, 2))
        for split in splits:
            self.assertEqual(np.unique(groups[split.test_indices]).size, 2)
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )

    def test_too_many_folds_raise_before_any_fold_is_built(self) -> None:
        X: NDArray[np.float64] = np.zeros((20000, 1))
        y: NDArray[np.float64] = np.zeros(20000)
        splitter: LeavePGroupsOutSplitter = LeavePGroupsOutSplitter(
            groups=np.arange(20000), n_groups=10000
        )
        with self.assertRaisesRegex(ValueError, "max_splits"):
            _ = splitter.split(X, y)

    def test_invalid_arguments_raise(self) -> None:
        X, y = linear_data(num_samples=4)
        groups: NDArray[np.int64] = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            _ = LeavePGroupsOutSplitter(groups=groups, n_groups=0)
        with self.assertRaises(ValueError):
            _ = LeavePGroupsOutSplitter(groups=groups, max_splits=0)
        with self.assertRaises(TypeError):
            _ = LeavePGroupsOutSplitter(groups=groups, n_groups=True)
        with self.assertRaises(ValueError):
            _ = LeavePGroupsOutSplitter(groups=groups, n_groups=2).split(X, y)


class TestLeavePOutSplitter(unittest.TestCase):
    def test_tests_on_every_combination(self) -> None:
        X, y = linear_data(num_samples=6)
        splits: tuple[DataSplit, ...] = LeavePOutSplitter(p=2).split(X, y)
        self.assertEqual(len(splits), math.comb(6, 2))
        combinations: set[bytes] = set()
        for split in splits:
            self.assertEqual(split.test_indices.size, 2)
            self.assertEqual(split.train_indices.size, 4)
            combinations.add(split.test_indices.tobytes())
        self.assertEqual(len(combinations), math.comb(6, 2))

    def test_too_many_folds_raise_before_any_fold_is_built(self) -> None:
        X: NDArray[np.float64] = np.zeros((20000, 1))
        y: NDArray[np.float64] = np.zeros(20000)
        with self.assertRaisesRegex(ValueError, "max_splits"):
            _ = LeavePOutSplitter(p=10000).split(X, y)
        self.assertEqual(
            len(LeavePOutSplitter(p=1, max_splits=1000).split(X[:1000], y[:1000])), 1000
        )

    def test_invalid_arguments_raise(self) -> None:
        X, y = linear_data(num_samples=4)
        with self.assertRaises(ValueError):
            _ = LeavePOutSplitter(p=0)
        with self.assertRaises(ValueError):
            _ = LeavePOutSplitter(max_splits=0)
        with self.assertRaises(TypeError):
            _ = LeavePOutSplitter(p=True)
        with self.assertRaises(ValueError):
            _ = LeavePOutSplitter(p=4).split(X, y)


class TestPredefinedSplitter(unittest.TestCase):
    def test_uses_the_given_folds(self) -> None:
        X, y = linear_data(num_samples=6)
        splits: tuple[DataSplit, ...] = PredefinedSplitter(
            test_folds=np.array([-1, -1, 0, 0, 1, 1])
        ).split(X, y)
        self.assertEqual(
            [split.test_indices.tolist() for split in splits], [[2, 3], [4, 5]]
        )
        self.assertEqual(splits[0].train_indices.tolist(), [0, 1, 4, 5])

    def test_single_validation_set(self) -> None:
        X, y = linear_data(num_samples=4)
        splits: tuple[DataSplit, ...] = PredefinedSplitter(
            test_folds=np.array([-1, -1, -1, 0])
        ).split(X, y)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0].test_indices.tolist(), [3])

    def test_invalid_folds_raise(self) -> None:
        X, y = linear_data(num_samples=3)
        with self.assertRaises(TypeError):
            _ = PredefinedSplitter(test_folds=np.array([0.5, -1.0]))
        invalid: tuple[list[int], ...] = ([-2, 0], [-1, -1], [0, 0])
        for folds in invalid:
            with self.subTest(folds=folds):
                with self.assertRaises(ValueError):
                    _ = PredefinedSplitter(test_folds=np.array(folds))
        with self.assertRaises(ValueError):
            _ = PredefinedSplitter(test_folds=np.array([-1, 0])).split(X, y)


class TestRepeatedGroupHoldoutSplitter(unittest.TestCase):
    def test_groups_never_cross_partitions(self) -> None:
        X, y = linear_data(num_samples=40)
        groups: NDArray[np.int64] = np.arange(40) % 10
        splits: tuple[DataSplit, ...] = RepeatedGroupHoldoutSplitter(
            groups=groups, n_repeats=4, test_fraction=0.2
        ).split(X, y)
        self.assertEqual(len(splits), 4)
        for split in splits:
            self.assertEqual(np.unique(groups[split.test_indices]).size, 2)
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )
        self.assertFalse(np.array_equal(splits[0].test_indices, splits[1].test_indices))

    def test_is_reproducible(self) -> None:
        X, y = linear_data(num_samples=40)
        splitter: RepeatedGroupHoldoutSplitter = RepeatedGroupHoldoutSplitter(
            groups=np.arange(40) % 10
        )
        self.assertTrue(
            np.array_equal(
                splitter.split(X, y)[0].test_indices,
                splitter.split(X, y)[0].test_indices,
            )
        )

    def test_invalid_arguments_raise(self) -> None:
        X, y = linear_data(num_samples=4)
        groups: NDArray[np.int64] = np.array([0, 0, 1, 1])
        for test_fraction in (0.0, 1.0, math.nan):
            with self.subTest(test_fraction=test_fraction):
                with self.assertRaises(ValueError):
                    _ = RepeatedGroupHoldoutSplitter(
                        groups=groups, test_fraction=test_fraction
                    )
        with self.assertRaises(ValueError):
            _ = RepeatedGroupHoldoutSplitter(groups=groups, n_repeats=0)
        with self.assertRaises(TypeError):
            _ = RepeatedGroupHoldoutSplitter(groups=np.array([0.5, 1.5]))
        with self.assertRaises(ValueError):
            _ = RepeatedGroupHoldoutSplitter(groups=np.array([0, 1])).split(X, y)
        with self.assertRaises(ValueError):
            _ = RepeatedGroupHoldoutSplitter(groups=np.zeros(4, dtype=np.int64)).split(
                X, y
            )


class TestLeaveOneOutSplitter(unittest.TestCase):
    def test_each_sample_is_tested_once(self) -> None:
        X, y = linear_data(num_samples=10)
        splits: tuple[DataSplit, ...] = LeaveOneOutSplitter().split(X, y)
        self.assertEqual(
            sorted(split.test_indices.item() for split in splits), list(range(10))
        )
        for split in splits:
            self.assertEqual(split.train_indices.size, 9)

    def test_single_sample_raises(self) -> None:
        X, y = linear_data(num_samples=1)
        with self.assertRaises(ValueError):
            _ = LeaveOneOutSplitter().split(X, y)


if __name__ == "__main__":
    _ = unittest.main()
