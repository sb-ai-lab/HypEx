"""Tests for Dataset mutation operations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)
from hypex.utils.errors import ConcatBackendError, ConcatDataError, DataTypeError


def _ds(make_dataset):
    """Standard three-column dataset."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "t": [0, 1, 0]})
    roles = {"x": FeatureRole(), "y": TargetRole(), "t": TreatmentRole()}
    return make_dataset(df, roles)


# ── add_column ────────────────────────────────────────────────────────────
def test_add_column_from_scalar(make_dataset) -> None:
    """add_column broadcasts a scalar value."""
    ds = _ds(make_dataset)
    ds.add_column(data=7, role={"new": InfoRole()})
    assert "new" in ds.columns


def test_add_column_from_list(make_dataset) -> None:
    """add_column accepts a plain list of values."""
    ds = _ds(make_dataset)
    ds.add_column(data=[10, 20, 30], role={"new": FeatureRole()})
    assert "new" in ds.columns
    assert len(ds) == 3


def test_add_column_from_dataset(make_dataset) -> None:
    """add_column accepts another Dataset when role is None."""
    ds = _ds(make_dataset)
    extra = make_dataset(pd.DataFrame({"new": [1, 2, 3]}), {"new": FeatureRole()})
    ds.add_column(data=extra)
    assert "new" in ds.columns


def test_add_column_duplicate_name_raises(make_dataset) -> None:
    """Adding a column with an existing name raises ValueError."""
    ds = _ds(make_dataset)
    with pytest.raises(ValueError):
        ds.add_column(data=[1, 2, 3], role={"x": FeatureRole()})


def test_add_column_requires_abcrole(make_dataset) -> None:
    """Role values must be ABCRole instances."""
    ds = _ds(make_dataset)
    with pytest.raises(TypeError):
        ds.add_column(data=[1, 2, 3], role={"new": "feature"})


# ── append ────────────────────────────────────────────────────────────────
def test_append_concatenates_rows(make_dataset) -> None:
    """append stacks rows and merges roles."""
    ds = _ds(make_dataset)
    other = make_dataset(
        pd.DataFrame({"x": [9], "y": [9.0], "t": [1]}),
        {"x": FeatureRole(), "y": TargetRole(), "t": TreatmentRole()},
    )
    combined = ds.append(other, reset_index=True)
    assert len(combined) == 4


def test_append_non_dataset_raises(make_dataset) -> None:
    """append with a non-Dataset object raises ConcatDataError."""
    ds = _ds(make_dataset)
    with pytest.raises(ConcatDataError):
        ds.append(pd.DataFrame({"x": [1]}))


@pytest.mark.spark
def test_append_cross_backend_raises(make_dataset, spark_session) -> None:
    """Appending across backends raises ConcatBackendError."""
    pytest.importorskip("pyspark")
    df = pd.DataFrame({"x": [1, 2]})
    roles = {"x": FeatureRole()}

    pandas_ds = Dataset(roles=roles, data=df)
    spark_ds = Dataset(roles=roles, data=df, backend="spark", session=spark_session)

    with pytest.raises(ConcatBackendError):
        pandas_ds.append(spark_ds)


# ── astype ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("target_type", [float, str])
def test_astype_converts(make_dataset, target_type) -> None:
    """astype converts column dtype and updates role data_type."""
    ds = _ds(make_dataset)
    casted = ds.astype({"x": target_type})
    assert casted.roles["x"].data_type is target_type


def test_astype_missing_column_raises(make_dataset) -> None:
    """astype raises KeyError for an unknown column in raise mode."""
    ds = _ds(make_dataset)
    with pytest.raises(KeyError):
        ds.astype({"ghost": float}, errors="raise")


# ── apply / map ───────────────────────────────────────────────────────────
def test_apply_on_empty_short_circuits() -> None:
    """apply on an empty dataset returns a deepcopy without calling func."""
    calls = []

    def _tracker(column):
        calls.append(1)
        return column

    empty = Dataset.create_empty(roles={"x": FeatureRole()})
    result = empty.apply(func=_tracker, role={"x": FeatureRole()})

    assert calls == []
    assert result.is_empty()


def test_map_transforms_values(make_dataset) -> None:
    """map applies an element-wise function."""
    ds = _ds(make_dataset)
    doubled = ds.map(lambda v: v * 2)
    assert len(doubled) == 3


# ── rename / replace / drop ───────────────────────────────────────────────
def test_rename_preserves_roles(make_dataset) -> None:
    """rename remaps both column names and their roles."""
    ds = _ds(make_dataset)
    renamed = ds.rename({"x": "x2"})

    assert "x2" in renamed.columns
    assert isinstance(renamed.roles["x2"], FeatureRole)
    assert "x" not in renamed.roles


def test_replace_values(make_dataset) -> None:
    """replace substitutes matching values."""
    ds = _ds(make_dataset)
    replaced = ds.replace(to_replace=1, value=100)
    assert len(replaced) == 3


def test_drop_column(make_dataset) -> None:
    """drop removes a column and its role."""
    ds = _ds(make_dataset)
    dropped = ds.drop(columns="x")
    assert "x" not in dropped.columns
    assert "x" not in dropped.roles


# ── fillna / dropna / isna ────────────────────────────────────────────────
def test_fillna_scalar(make_dataset) -> None:
    """fillna replaces NaN with a scalar."""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    ds = make_dataset(df, {"x": FeatureRole()})
    filled = ds.fillna(values=0)
    assert len(filled) == 3


def test_fillna_requires_argument(make_dataset) -> None:
    """fillna without values or method raises ValueError."""
    ds = _ds(make_dataset)
    with pytest.raises(ValueError):
        ds.fillna()


@pytest.mark.parametrize("how", ["any", "all"])
def test_dropna_how(make_dataset, how) -> None:
    """dropna supports how='any' and how='all'."""
    df = pd.DataFrame({"x": [1.0, np.nan], "y": [np.nan, np.nan]})
    ds = make_dataset(df, {"x": FeatureRole(), "y": FeatureRole()})

    result = ds.dropna(how=how)
    assert len(result) == (1 if how == "any" else 0)


def test_isna_and_na_counts(make_dataset) -> None:
    """isna and na_counts report missing values."""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    ds = make_dataset(df, {"x": FeatureRole()})

    assert ds.count_nulls()["x"] == 1
    assert len(ds.isna()) == 3


# ── explode / list_to_columns ─────────────────────────────────────────────
def test_explode(make_dataset) -> None:
    """explode expands list values into separate rows."""
    df = pd.DataFrame({"x": [[1, 2], [3]]})
    ds = make_dataset(df, {"x": FeatureRole()})
    exploded = ds.explode("x")
    assert len(exploded) == 3


def test_list_to_columns(make_dataset) -> None:
    """list_to_columns splits a list column into positional columns."""
    df = pd.DataFrame({"x": [[1, 2], [3, 4]]})
    ds = make_dataset(df, {"x": FeatureRole()})
    expanded = ds.list_to_columns("x")
    assert "x" not in expanded.columns
    assert len(expanded.columns) == 2


# ── roles management ──────────────────────────────────────────────────────
def test_replace_roles(make_dataset) -> None:
    """replace_roles swaps roles for selected columns."""
    ds = _ds(make_dataset)
    ds.replace_roles({"x": InfoRole()})
    assert isinstance(ds.roles["x"], InfoRole)


def test_tmp_roles_lifecycle(make_dataset) -> None:
    """tmp_roles can be set, read and cleared."""
    ds = _ds(make_dataset)
    ds.tmp_roles = {"x": TargetRole()}
    assert isinstance(ds.tmp_roles["x"], TargetRole)

    ds.tmp_roles = {}
    assert ds.tmp_roles == {}


@pytest.mark.parametrize(
    "roles,expected",
    [
        (FeatureRole(), ["x"]),
        (TargetRole(), ["y"]),
        ([FeatureRole(), TargetRole()], ["x", "y"]),
    ],
)
def test_search_columns(make_dataset, roles, expected) -> None:
    """search_columns finds columns by one or many roles."""
    ds = _ds(make_dataset)
    assert sorted(ds.search_columns(roles)) == sorted(expected)


def test_search_columns_by_type(make_dataset) -> None:
    """search_columns_by_type filters by role data_type."""
    ds = _ds(make_dataset)
    float_cols = ds.search_columns_by_type(float)
    assert float_cols == ["y"]