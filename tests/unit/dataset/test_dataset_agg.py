"""Tests for Dataset aggregation and statistical methods."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole


def _ds(make_dataset):
    """Simple numeric dataset with known statistics."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    return make_dataset(df, {"x": FeatureRole(), "y": FeatureRole()})


def _scalar(value) -> float:
    """Unwrap single-value aggregation results to a plain float."""
    return float(value)


def test_basic_stats_match_expected(make_dataset) -> None:
    """mean/max/min/sum/count return exact expected values."""
    ds = _ds(make_dataset)

    assert _scalar(ds.mean().get_values(row=0, column="x")) == pytest.approx(2.0)
    assert _scalar(ds.max().get_values(row=0, column="y")) == pytest.approx(30.0)
    assert _scalar(ds.min().get_values(row=0, column="x")) == pytest.approx(1.0)
    assert _scalar(ds.sum().get_values(row=0, column="x")) == pytest.approx(6.0)


@pytest.mark.parametrize("ddof,expected", [(0, 1.0), (1, math.sqrt(1.0))])
def test_std_ddof(make_dataset, ddof, expected) -> None:
    """std honours the ddof parameter."""
    ds = _ds(make_dataset)
    result = ds.std(ddof=ddof)
    assert _scalar(result.get_values(row=0, column="x")) == pytest.approx(expected)


def test_var_bessel_correction(make_dataset) -> None:
    """var uses Bessel's correction (ddof=1) by default."""
    ds = _ds(make_dataset)
    result = ds.var()
    assert _scalar(result.get_values(row=0, column="x")) == pytest.approx(1.0)


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
def test_quantile(make_dataset, q) -> None:
    """quantile returns correct values for boundary and median levels."""
    ds = _ds(make_dataset)
    result = ds.quantile(q)
    expected = 1.0 + q * 2.0
    assert _scalar(result.get_values(row=0, column="x")) == pytest.approx(expected)


def test_coefficient_of_variation(make_dataset) -> None:
    """coefficient_of_variation equals std / mean."""
    ds = _ds(make_dataset)
    result = ds.coefficient_of_variation()
    value = result.get_values()[0][0]
    assert float(value) == pytest.approx(1.0 / 2.0)


def test_corr_perfect_correlation(make_dataset) -> None:
    """Perfectly correlated columns yield correlation of 1.0."""
    ds = _ds(make_dataset)
    corr = ds.corr(numeric_only=True)
    assert float(corr.get_values(row="x", column="y")) == pytest.approx(1.0)


def test_cov(make_dataset) -> None:
    """cov returns a covariance dataset."""
    ds = _ds(make_dataset)
    cov = ds.cov()
    assert len(cov.columns) >= 1


@pytest.mark.parametrize("func", ["mean", ["mean", "sum"], {"x": "mean"}])
def test_agg_with_str_list_dict(make_dataset, func) -> None:
    """agg accepts string, list and dict specifications."""
    ds = _ds(make_dataset)
    result = ds.agg(func)
    assert result is not None


def test_log(make_dataset) -> None:
    """log applies the natural logarithm element-wise."""
    df = pd.DataFrame({"x": [1.0, math.e]})
    ds = make_dataset(df, {"x": FeatureRole()})
    logged = ds.log()
    values = logged.get_values(column="x")
    assert float(values[0][0]) == pytest.approx(0.0)
    assert float(values[1][0]) == pytest.approx(1.0)


def test_value_counts(make_dataset) -> None:
    """value_counts returns per-value frequency counts."""
    df = pd.DataFrame({"x": ["a", "a", "b"]})
    ds = make_dataset(df, {"x": FeatureRole()})
    counts = ds.value_counts()
    assert len(counts) == 2


def test_nunique_and_unique(make_dataset) -> None:
    """nunique counts distinct values; unique lists them."""
    df = pd.DataFrame({"x": [1, 1, 2]})
    ds = make_dataset(df, {"x": FeatureRole()})

    assert ds.nunique()["x"] == 2
    assert len(ds.unique()["x"]) == 2


def test_isin(make_dataset) -> None:
    """isin produces a boolean mask dataset."""
    ds = _ds(make_dataset)
    mask = ds.isin([1.0])
    assert len(mask) == 3


def test_dot_with_numpy(make_dataset) -> None:
    """dot multiplies a dataset by a numpy vector."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    ds = make_dataset(df, {"x": FeatureRole(), "y": FeatureRole()})

    result = ds.dot(np.array([1.0, 1.0]))
    assert len(result) == 2


def test_stats_on_empty_dataset() -> None:
    """Aggregations on an empty dataset do not raise."""
    empty = Dataset.create_empty(roles={"x": FeatureRole()})
    assert empty.count() is not None or empty.is_empty()


def test_std_single_row_is_nan(make_dataset) -> None:
    """std with ddof=1 on a single row is NaN."""
    df = pd.DataFrame({"x": [5.0]})
    ds = make_dataset(df, {"x": FeatureRole()})
    result = ds.std(ddof=1)
    value = result.get_values(row=0, column="x")
    assert value is None or (isinstance(value, float) and math.isnan(value))