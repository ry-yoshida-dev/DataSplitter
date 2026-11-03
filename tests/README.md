# Tests

## Overview

Unit tests for the `data_splitter` package.

## Components

| Component | Description |
| --- | --- |
| [`data.py`](data.py) | Shared synthetic samples and type aliases. |
| [`test_data_split.py`](test_data_split.py) | Tests for `DataSplit`. |
| [`test_validation_splitter.py`](test_validation_splitter.py) | Tests for train/validation/test partitions and nested validation. |
| [`test_temporal_splitters.py`](test_temporal_splitters.py) | Tests for `PurgedKFoldSplitter` and `GroupTimeSeriesSplitter`. |
| [`test_one_class.py`](test_one_class.py) | Tests for `OneClassSplitter`. |
| [`test_splitters.py`](test_splitters.py) | Tests for the strategies that ignore labels. |
| [`test_optional_labels.py`](test_optional_labels.py) | Tests for unlabeled samples of any shape and for which strategies need labels. |
| [`test_stratified_k_fold.py`](test_stratified_k_fold.py) | Tests for `StratifiedKFoldSplitter`. |
| [`test_stratified_holdout.py`](test_stratified_holdout.py) | Tests for `StratifiedHoldoutSplitter`. |
| [`test_repeated_stratified_k_fold.py`](test_repeated_stratified_k_fold.py) | Tests for `RepeatedStratifiedKFoldSplitter`. |
| [`test_stratified_group_k_fold.py`](test_stratified_group_k_fold.py) | Tests for `StratifiedGroupKFoldSplitter`. |

## Examples

```bash
uv run pytest
```
