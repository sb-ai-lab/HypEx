"""Tests for Dataset.groupby and GroupedDataset aggregations."""
from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import FeatureRole, GroupingRole
from hypex.utils import BackendsEnum


def _grouped(make_dataset):
    """Dataset grouped into two groups of two rows each."""
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [1.0, 3.0, 10.0, 30.0],
            "w": [2.0, 4.0, 20.0, 40.0],
        }
    )
    roles = {"g": GroupingRole(), "v": FeatureRole(), "w": FeatureRole()}
    return make_dataset(df, roles)


@pytest.mark.xfail(
    strict=True,
    raises=(FutureWarning, TypeError),
    reason="Issue: count_groups does int(df[cols].nunique()) on a Series: FutureWarning on pandas, TypeError on Spark, so len(GroupedDataset) breaks",
)
def test_groupby_returns_grouped_dataset(make_dataset) -> None:
    """groupby yields a GroupedDataset with the correct group count."""
    ds = _grouped(make_dataset)
    grouped = ds.groupby("g")
    assert len(grouped) == 2


def test_grouped_mean(make_dataset) -> None:
    """Grouped mean aggregates each group correctly."""
    grouped = _grouped(make_dataset).groupby("g")
    result = grouped.mean()

    assert len(result) == 2
    assert "v" in result.columns

@pytest.mark.parametrize("reducer", ["count", "sum", "min", "max", "first", "last", "median"])
def test_grouped_reducers(make_dataset, reducer) -> None:
    """All basic reducers run without error and return one row per group."""
    grouped = _grouped(make_dataset).groupby("g")
    result = getattr(grouped, reducer)()
    assert len(result) == 2


def test_grouped_std_var(make_dataset, backend) -> None:
    """std and var reducers produce one row per group."""
    grouped = _grouped(make_dataset).groupby("g")
    assert len(grouped.std()) == 2
    if backend == BackendsEnum.spark:
        pytest.skip("Spark var mapping issue")
    assert len(grouped.var()) == 2


def test_grouped_reducer_with_column_subset(make_dataset) -> None:
    """Reducers accept explicit column arguments."""
    grouped = _grouped(make_dataset).groupby("g")
    result = grouped.mean("v")
    assert "v" in result.columns


def test_grouped_agg_str(make_dataset) -> None:
    """agg accepts a single function name."""
    grouped = _grouped(make_dataset).groupby("g")
    result = grouped.agg("mean")
    assert len(result) == 2


def test_grouped_value_counts(make_dataset) -> None:
    """value_counts produces per-group category counts."""
    df = pd.DataFrame({"g": ["a", "a", "b"], "c": ["x", "x", "y"]})
    ds = make_dataset(df, {"g": GroupingRole(), "c": FeatureRole()})
    grouped = ds.groupby("g")
    result = grouped.value_counts()
    assert len(result) == 2


@pytest.mark.xfail(
    strict=True,
    raises=(FutureWarning, TypeError),
    reason="Issue: count_groups does int(df[cols].nunique()) on a Series: FutureWarning on pandas, TypeError on Spark, so len(GroupedDataset) breaks",
)
def test_grouped_size_len_iter(make_dataset) -> None:
    """size, __len__ and iteration are consistent."""
    grouped = _grouped(make_dataset).groupby("g")

    assert len(grouped) == 2

    keys = [key for key, _ in grouped]
    assert sorted(keys) == ["a", "b"]


@pytest.mark.xfail(
    strict=True,
    raises=(FutureWarning, TypeError),
    reason="Issue: count_groups does int(df[cols].nunique()) on a Series: FutureWarning on pandas, TypeError on Spark, so len(GroupedDataset) breaks",
)
def test_groupby_single_group(make_dataset) -> None:
    """groupby works when all rows belong to one group."""
    df = pd.DataFrame({"g": ["a", "a"], "v": [1.0, 2.0]})
    ds = make_dataset(df, {"g": GroupingRole(), "v": FeatureRole()})
    grouped = ds.groupby("g")
    assert len(grouped) == 1


def test_groupby_group_of_one_row(make_dataset) -> None:
    """A group with a single row still aggregates without errors."""
    df = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
    ds = make_dataset(df, {"g": GroupingRole(), "v": FeatureRole()})
    grouped = ds.groupby("g")
    result = grouped.count()
    assert len(result) == 2
