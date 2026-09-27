"""Tests for DatasetAdapter and the low-level Adapter helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole, SmallDataset
from hypex.dataset.dataset import DatasetAdapter
from hypex.utils.adapter import Adapter
from hypex.utils.errors import InvalidArgumentError


def test_to_dataset_dispatch_scalar() -> None:
    """Scalars are wrapped into a 1x1 dataset."""
    ds = DatasetAdapter.to_dataset(5, roles=FeatureRole())
    assert len(ds) == 1


def test_to_dataset_dispatch_dict() -> None:
    """Dicts become column-oriented datasets."""
    ds = DatasetAdapter.to_dataset(
        {"a": [1, 2]}, roles={"a": FeatureRole()}, small=False
    )
    assert isinstance(ds, Dataset)
    assert len(ds) == 2


def test_to_dataset_dispatch_list() -> None:
    """Lists become single-column datasets."""
    ds = DatasetAdapter.to_dataset(
        [1, 2, 3], roles={"a": FeatureRole()}, small=False
    )
    assert len(ds) == 3


def test_to_dataset_dispatch_dataframe() -> None:
    """pandas DataFrames are wrapped directly."""
    df = pd.DataFrame({"a": [1, 2]})
    ds = DatasetAdapter.to_dataset(df, roles={"a": FeatureRole()}, small=False)
    assert len(ds) == 2


def test_to_dataset_dispatch_ndarray() -> None:
    """2-D numpy arrays become datasets with role-named columns."""
    arr = np.array([[1, 2], [3, 4]])
    ds = DatasetAdapter.to_dataset(
        arr, roles={"c1": FeatureRole(), "c2": FeatureRole()}, small=False
    )
    assert ds.shape[0] == 2


def test_to_dataset_dispatch_existing_dataset() -> None:
    """An existing Dataset passes through unchanged when small=False."""
    source = Dataset(
        roles={"a": FeatureRole()}, data=pd.DataFrame({"a": [1]})
    )
    result = DatasetAdapter.to_dataset(source, roles={"a": FeatureRole()}, small=False)
    assert isinstance(result, Dataset)


def test_to_dataset_invalid_data_raises() -> None:
    """Unsupported data types raise InvalidArgumentError."""
    with pytest.raises(InvalidArgumentError):
        DatasetAdapter.to_dataset(object(), roles={"a": FeatureRole()})


def test_value_to_dataset() -> None:
    """value_to_dataset wraps a scalar with the given role name."""
    ds = DatasetAdapter.value_to_dataset(3.14, roles=FeatureRole())
    assert len(ds) == 1


def test_adapter_to_list() -> None:
    """Adapter.to_list normalises scalars, lists and None."""
    assert Adapter.to_list(None) == []
    assert Adapter.to_list("x") == ["x"]
    assert Adapter.to_list([1, 2]) == [1, 2]


@pytest.mark.parametrize(
    "data,expected",
    [([], None), ([1], 1)],
)
def test_list_to_single(data, expected) -> None:
    """list_to_single unwraps zero/one-element lists."""
    assert Adapter.list_to_single(data) == expected


def test_list_to_single_multiple_raises() -> None:
    """list_to_single rejects multi-element lists."""
    with pytest.raises(ValueError):
        Adapter.list_to_single([1, 2])