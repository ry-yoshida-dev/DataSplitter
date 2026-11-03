# data_splitter

## Overview

Strategies that partition samples into train and test indices, independent of what the samples are or how they are used afterwards.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for indices, counts, and group identifiers. |
| [random_state.py](./random_state.py) | `DEFAULT_RANDOM_STATE`, the default seed of splitters. |
| [parameter_validator.py](./parameter_validator.py) | `ParameterValidator`, runtime checks for non-integer parameters and non-discrete group identifiers. |
| [index_validator.py](./index_validator.py) | `IndexValidator`, runtime checks of sample indices. |
| [data_split.py](./data_split.py) | `DataSplit`, the read-only indices of one train/test partition. |
| [validated_split.py](./validated_split.py) | `ValidatedSplit`, the read-only indices of one train/validation/test partition. |
| [splitter.py](./splitter.py) | `DatasetSplitter`, the abstract splitting strategy for labeled samples. |
| [label_free_splitter.py](./label_free_splitter.py) | `LabelFreeSplitter`, the abstract splitting strategy that also accepts unlabeled samples. |
| [validation_splitter.py](./validation_splitter.py) | `ValidationSplitter`, which combines two strategies into train/validation/test partitions. |
| [splitters/](./splitters/README.md) | Concrete splitting strategies. |
