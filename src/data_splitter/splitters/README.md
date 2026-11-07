# splitters

## Overview

Concrete `DatasetSplitter` strategies, one per file.

## Components

| Component | Description |
|-----------|-------------|
| [k_fold.py](./k_fold.py) | `KFoldSplitter`, the common default. |
| [holdout.py](./holdout.py) | `HoldoutSplitter`, a single partition. |
| [repeated_k_fold.py](./repeated_k_fold.py) | `RepeatedKFoldSplitter`, k-fold repeated with new shuffles. |
| [repeated_holdout.py](./repeated_holdout.py) | `RepeatedHoldoutSplitter`, independently drawn random partitions. |
| [leave_one_out.py](./leave_one_out.py) | `LeaveOneOutSplitter`, which tests on each sample in turn. |
| [leave_p_out.py](./leave_p_out.py) | `LeavePOutSplitter`, which tests on every combination of samples. |
| [group_k_fold.py](./group_k_fold.py) | `GroupKFoldSplitter`, which keeps each group on one side. |
| [leave_one_group_out.py](./leave_one_group_out.py) | `LeaveOneGroupOutSplitter`, which tests on each group in turn. |
| [leave_p_groups_out.py](./leave_p_groups_out.py) | `LeavePGroupsOutSplitter`, which tests on every combination of groups. |
| [repeated_group_holdout.py](./repeated_group_holdout.py) | `RepeatedGroupHoldoutSplitter`, random partitions that keep each group on one side. |
| [stratified_k_fold.py](./stratified_k_fold.py) | `StratifiedKFoldSplitter`, which preserves class proportions in each fold. |
| [stratified_holdout.py](./stratified_holdout.py) | `StratifiedHoldoutSplitter`, a single partition preserving class proportions. |
| [repeated_stratified_k_fold.py](./repeated_stratified_k_fold.py) | `RepeatedStratifiedKFoldSplitter`, stratified k-fold repeated with new shuffles. |
| [stratified_group_k_fold.py](./stratified_group_k_fold.py) | `StratifiedGroupKFoldSplitter`, stratified folds that keep each group on one side. |
| [predefined.py](./predefined.py) | `PredefinedSplitter`, partitions fixed in advance. |
| [time_series.py](./time_series.py) | `TimeSeriesSplitter`, which always tests on later samples. |
| [purged_k_fold.py](./purged_k_fold.py) | `PurgedKFoldSplitter`, ordered folds that remove training samples next to the test fold. |
| [group_time_series.py](./group_time_series.py) | `GroupTimeSeriesSplitter`, forward chaining over consecutive groups. |
| [one_class.py](./one_class.py) | `OneClassSplitter`, which keeps one class in training. |
