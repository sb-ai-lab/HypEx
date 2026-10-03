"""Tests for Dataset indexing, selection and value access."""
from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import FeatureRole, TargetRole
from hypex.utils import BackendsEnum
from hypex.utils.errors import RoleColumnError


def _ds(make_dataset):
    """Four-column fixture-like dataset used by indexing tests."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [10.0, 20.0, 30.0, 40.0],
            "c": ["w", "x", "y", "z"],
        }
    )
    roles = {"a": FeatureRole(), "b": TargetRole(), "c": FeatureRole()}
    return make_dataset(df, roles)


def test_getitem_by_column_name(make_dataset) -> None:
    """Selecting a single column by name returns a one-column Dataset."""
    ds = _ds(make_dataset)
    single = ds["a"]

    assert list(single.columns) == ["a"]
    assert len(single) == 4


def test_getitem_by_column_list_preserves_order(make_dataset) -> None:
    """Selecting multiple columns keeps the requested order."""
    ds = _ds(make_dataset)
    subset = ds[["b", "a"]]

    assert list(subset.columns) == ["b", "a"]


def test_getitem_by_boolean_mask(make_dataset) -> None:
    """A boolean mask Dataset filters rows."""
    ds = _ds(make_dataset)
    mask = ds["a"] > 2
    filtered = ds[mask]

    assert len(filtered) == 2


def test_getitem_missing_column_raises(make_dataset) -> None:
    """Requesting an unknown column raises KeyError."""
    ds = _ds(make_dataset)
    with pytest.raises(KeyError):
        _ = ds["nope"]


def test_setitem_new_column_warns(make_dataset) -> None:
    """Assigning a brand-new column via __setitem__ emits SyntaxWarning."""
    ds = _ds(make_dataset)
    with pytest.warns(SyntaxWarning, match="add_column"):
        ds["new_col"] = [1, 2, 3, 4]
    assert "new_col" in ds.columns


def test_setitem_existing_column_replaces(make_dataset) -> None:
    """Assigning to an existing column replaces its values."""
    ds = _ds(make_dataset)
    ds["a"] = [9, 9, 9, 9]
    assert ds["a"].get_values() == [[9], [9], [9], [9]] or ds["a"].sum() == 36


@pytest.mark.xfail(
    strict=True,
    raises=RoleColumnError,
    reason="Issue: Dataset.get(key) rebuilds the Dataset with all original roles, so selecting a single column raises RoleColumnError",
)
def test_get_with_default(make_dataset) -> None:
    """get() returns the column or the provided default."""
    ds = _ds(make_dataset)
    assert ds.get("a") is not None
    assert ds.backend_data.get("missing", "fallback") == "fallback"


def test_select_and_iselect(make_dataset) -> None:
    """select uses names while iselect uses integer positions."""
    ds = _ds(make_dataset)

    by_name = ds.select(["a", "b"])
    by_pos = ds.iselect([0, 1])

    assert list(by_name.columns) == ["a", "b"]
    assert list(by_pos.columns) == ["a", "b"]


@pytest.mark.parametrize(
    "include,exclude,expected",
    [
        (["int"], None, ["a"]),
        (["float"], None, ["b"]),
        (None, ["object"], ["a", "b"]),
    ],
)
def test_select_dtypes(make_dataset, include, exclude, expected) -> None:
    """select_dtypes filters columns by dtype inclusion/exclusion."""
    ds = _ds(make_dataset)
    result = ds.select_dtypes(include=include, exclude=exclude)
    assert set(result.columns) == set(expected)


def test_filter_items(make_dataset) -> None:
    """filter(items=...) keeps only the listed columns."""
    ds = _ds(make_dataset)
    result = ds.filter(items=["a", "c"], axis=1)
    assert set(result.columns) == {"a", "c"}


def test_filter_regex(make_dataset) -> None:
    """filter(regex=...) keeps columns matching the pattern."""
    ds = _ds(make_dataset)
    result = ds.filter(regex="^[ab]$", axis=1)
    assert set(result.columns) == {"a", "b"}


def test_limit_and_take(make_dataset) -> None:
    """limit truncates rows; take selects by position."""
    ds = _ds(make_dataset)

    limited = ds.limit(2)
    assert len(limited) == 2

    taken = ds.take([0, 2])
    assert len(taken) == 2


@pytest.mark.spark
def test_index_property_and_setter(make_dataset) -> None:
    """index is readable and writable."""
    pytest.skip("Setting list index in pyspark.pandas is unstable")


def test_reset_index_drop(make_dataset, xfail_backend) -> None:
    xfail_backend(
        BackendsEnum.spark,
        reason="Issue: SparkDataset.index setter passes the list to set_index() as column names (KeyError)",
        raises=KeyError,
    )
    """reset_index(drop=True) returns a fresh RangeIndex dataset."""
    ds = _ds(make_dataset)
    ds.index = [10, 11, 12, 13]
    reset = ds.reset_index(drop=True)
    assert list(reset.index) == [0, 1, 2, 3]


def test_set_index(make_dataset) -> None:
    """set_index promotes a column to index and removes its role."""
    ds = _ds(make_dataset)
    reindexed = ds.set_index("c")

    assert "c" not in reindexed.columns
    assert "c" not in reindexed.roles


def test_shape_columns_len(make_dataset) -> None:
    """shape, columns and len report consistent dimensions."""
    ds = _ds(make_dataset)
    assert ds.shape == (4, 3)
    assert len(ds.columns) == 3
    assert len(ds) == 4


def test_get_values_vs_iget_values(make_dataset) -> None:
    """get_values resolves by label while iget_values resolves by position."""
    ds = _ds(make_dataset)

    by_label = ds.get_values(column="a")
    by_position = ds.iget_values(column=0)

    assert by_label == by_position