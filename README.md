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

Each strategy is described in [src/data_splitter/splitters/](src/data_splitter/splitters/README.md).

## Installation

```bash
pip install .
```

## Usage

Labels are optional unless the strategy preserves class proportions:

```python
from data_splitter import KFoldSplitter, StratifiedKFoldSplitter

for split in KFoldSplitter(n_splits=5).split(X):
    X_train, X_test = X[split.train_indices], X[split.test_indices]

splits = StratifiedKFoldSplitter(n_splits=5).split(X, y)
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

Keep correlated samples together, or respect temporal order:

```python
from data_splitter import GroupKFoldSplitter, TimeSeriesSplitter

by_subject = GroupKFoldSplitter(groups=subject_ids).split(X)
forward = TimeSeriesSplitter(n_splits=5, gap=1).split(X)
```
