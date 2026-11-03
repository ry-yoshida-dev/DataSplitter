# DataSplitter

## Overview

DataSplitter (`data_splitter`) partitions samples into train and test indices. Every strategy is a validated, frozen dataclass whose `split` returns one read-only `DataSplit` per partition. Only indices are returned, so samples can be anything indexable by them.

| How samples relate | Strategies |
| --- | --- |
| Independent | `KFoldSplitter`, `HoldoutSplitter`, `RepeatedKFoldSplitter`, `RepeatedHoldoutSplitter` |
| Very few | `LeaveOneOutSplitter`, `LeavePOutSplitter` |
| Correlated within groups | `GroupKFoldSplitter`, `LeaveOneGroupOutSplitter`, `LeavePGroupsOutSplitter`, `RepeatedGroupHoldoutSplitter` |
| Imbalanced classes | `StratifiedKFoldSplitter`, `StratifiedHoldoutSplitter`, `RepeatedStratifiedKFoldSplitter`, `StratifiedGroupKFoldSplitter` |
| Fixed in advance | `PredefinedSplitter` |
| Temporal | `TimeSeriesSplitter`, `PurgedKFoldSplitter`, `GroupTimeSeriesSplitter` |
| One class in training | `OneClassSplitter` around any strategy |

For module-level design notes, see [src/data_splitter/README.md](src/data_splitter/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick example

Labels are optional unless the strategy preserves class proportions:

```python
from data_splitter import KFoldSplitter, StratifiedKFoldSplitter

for split in KFoldSplitter(n_splits=5).split(X):
    X_train, X_test = X[split.train_indices], X[split.test_indices]

splits = StratifiedKFoldSplitter(n_splits=5).split(X, y)
```

Strategies that need labels take `y` as a required argument, so forgetting it is a type error. Selecting samples is part of a split:

```python
for split in KFoldSplitter().split(X):
    X_train = split.select_train(X)
    y_test = split.select_test(y)
```

Set test samples aside and tune on validation samples, once or in nested cross-validation:

```python
from data_splitter import HoldoutSplitter, KFoldSplitter, ValidationSplitter

once = ValidationSplitter(outer=HoldoutSplitter(), inner=HoldoutSplitter()).split(X)
nested = ValidationSplitter(
    outer=KFoldSplitter(), inner=KFoldSplitter(n_splits=3)
).split(X)
for split in nested:
    X_validation = split.select_validation(X)
```

Train anomaly detectors on normal samples only, and test on everything:

```python
from data_splitter import OneClassSplitter, StratifiedKFoldSplitter

splits = OneClassSplitter(base=StratifiedKFoldSplitter(), training_label=False).split(
    X, is_anomalous
)
```

Keep correlated samples together, or respect temporal order:

```python
from data_splitter import GroupKFoldSplitter, TimeSeriesSplitter

by_subject = GroupKFoldSplitter(groups=subject_ids).split(X)
forward = TimeSeriesSplitter(n_splits=5, gap=1).split(X)
```

Keep whole clips on one side of time, or test on all of an autocorrelated sequence:

```python
from data_splitter import GroupTimeSeriesSplitter, PurgedKFoldSplitter

by_clip = GroupTimeSeriesSplitter(groups=clip_ids).split(frames)
purged = PurgedKFoldSplitter(n_splits=5, gap=30).split(windows)
```

## Notes

- Strategies that do not need labels derive from `LabelFreeSplitter` and accept unlabeled samples. Code that always has labels can depend on `DatasetSplitter`, which covers every strategy.
- `TimeSeriesSplitter`, `PurgedKFoldSplitter`, and `GroupTimeSeriesSplitter` expect samples already in temporal order. `GroupTimeSeriesSplitter` rejects groups that are not one consecutive run.
- `PurgedKFoldSplitter` trains on samples from both sides of the test fold. Use forward chaining when the future must never inform the past.
- `ValidationSplitter` cannot take a strategy holding per-sample `groups` as its inner strategy, because those groups would not match the remaining samples; it raises `ValueError`.
- `OneClassSplitter` raises `ValueError` when a partition has no training samples of the class.
- `LeavePOutSplitter` and `LeavePGroupsOutSplitter` test on every combination, so their partitions overlap and grow combinatorially; `split` raises `ValueError` when they would exceed `max_splits`, 10000 by default.
- Repeated strategies produce overlapping test partitions.
- Strategies reject non-integer values at construction with `TypeError`, and values out of range with `ValueError`.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request, along with the tests on the minimum supported dependency versions.

## License

Released under the [MIT License](LICENSE).
