"""Documented divergences between the Pandas and Spark backends.

Each test states the *desired* contract (identical behaviour) and is marked
``xfail(strict=True)`` for the backend that currently violates it, so that a
fix flips the test to XPASS(strict) and forces the marker to be removed.
Tests without ``xfail`` pin down divergences that are intentional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole
from hypex.utils import BackendsEnum

pytestmark = pytest.mark.spark

MOD = 10_000_000


def _build(df: pd.DataFrame, backend: BackendsEnum, session) -> Dataset:
    roles = {c: FeatureRole() for c in df.columns}
    return Dataset(
        roles=roles,
        data=df,
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def _pdf(ds: Dataset) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def _spark_xfail(reason: str):
    return pytest.param(
        BackendsEnum.spark,
        marks=pytest.mark.xfail(strict=True, reason=reason),
        id="spark",
    )


PANDAS = pytest.param(BackendsEnum.pandas, id="pandas")


# ---------------------------------------------------------------------------
# labels_dict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "backend_kind",
    [
        PANDAS,
        _spark_xfail("Issue: SparkDataset.labels_dict raises NotImplementedError"),
    ],
)
def test_labels_dict_diverges(backend_kind, spark_session) -> None:
    """``labels_dict`` should be readable on every backend."""
    ds = _build(
        pd.DataFrame({"x": [1, 2], "s": ["a", "b"]}), backend_kind, spark_session
    )
    assert isinstance(ds.labels_dict, dict)


# ---------------------------------------------------------------------------
# Hash split
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="Issue: Pandas uses MD5, Spark uses Murmur3 -> different labels for the same seed",
)
def test_hash_split_diverges_across_backends(spark_session) -> None:
    """Same index + seed must yield the same labels (docstring promises it)."""
    df = pd.DataFrame({"x": np.arange(50, dtype=float)})
    kwargs = dict(edges=[MOD // 2, MOD], labels=["A", "B"], random_state=42)
    left = _pdf(
        _build(df, BackendsEnum.pandas, spark_session).random_split_labels(**kwargs)
    )
    right = _pdf(
        _build(df, BackendsEnum.spark, spark_session).random_split_labels(**kwargs)
    )
    assert left["split"].sort_index().tolist() == right["split"].sort_index().tolist()


# ---------------------------------------------------------------------------
# NaN vs null
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "backend_kind", [PANDAS, pytest.param(BackendsEnum.spark, id="spark")]
)
def test_nan_vs_null_in_dropna(backend_kind, spark_session) -> None:
    """dropna removes both float NaN and string null on both backends."""
    df = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, 4.0], "s": ["a", "b", None, "d"], "k": [1, 2, 3, 4]}
    )
    ds = _build(df, backend_kind, spark_session)
    assert sorted(_pdf(ds.dropna())["k"].tolist()) == [1, 4]
    assert sorted(_pdf(ds.dropna(subset=["s"]))["k"].tolist()) == [1, 2, 4]
    assert sorted(_pdf(ds.dropna(subset=["x"]))["k"].tolist()) == [1, 3, 4]


@pytest.mark.parametrize(
    "backend_kind",
    [
        PANDAS,
        _spark_xfail(
            "Issue: pyspark.pandas propagates NaN in mean/sum instead of skipping"
        ),
    ],
)
def test_nan_vs_null_in_stats(backend_kind, spark_session) -> None:
    """Aggregations skip NaN (pandas semantics) on every backend."""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, 3.0]})
    ds = _build(df, backend_kind, spark_session)
    frame = _pdf(ds.mean())
    series = frame.iloc[0] if len(frame) == 1 else frame.iloc[:, 0]
    assert float(series["x"]) == pytest.approx(2.0)
    frame = _pdf(ds.sum())
    series = frame.iloc[0] if len(frame) == 1 else frame.iloc[:, 0]
    assert float(series["x"]) == pytest.approx(4.0)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: Spark count() includes NaN and drops non-numeric columns",
)
def test_count_ignores_nan_on_spark(spark_session) -> None:
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "s": ["a", "b", "c"]})
    ds = _build(df, BackendsEnum.spark, spark_session)
    frame = _pdf(ds.count())
    series = frame.iloc[0] if len(frame) == 1 else frame.iloc[:, 0]
    assert int(series["x"]) == 2
    assert "s" in series.index


# ---------------------------------------------------------------------------
# Aggregation layout / semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "backend_kind",
    [
        PANDAS,
        _spark_xfail(
            "Issue: Spark aggregation result is transposed, get_values(row=, column=) -> KeyError"
        ),
    ],
)
def test_aggregation_layout_diverges(backend_kind, spark_session) -> None:
    ds = _build(
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}),
        backend_kind,
        spark_session,
    )
    assert float(ds.mean().get_values(row="mean", column="x")) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "backend_kind",
    [PANDAS, _spark_xfail("Issue: SparkDataset.std ignores ddof (always ddof=1)")],
)
def test_std_ddof_zero_diverges(backend_kind, spark_session) -> None:
    ds = _build(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}),
        backend_kind,
        spark_session,
    )
    frame = _pdf(ds.std(ddof=0))
    series = frame.iloc[0] if len(frame) == 1 else frame.iloc[:, 0]
    assert float(series["x"]) == pytest.approx(np.std([1, 2, 3, 4]))


@pytest.mark.parametrize(
    "backend_kind",
    [
        PANDAS,
        _spark_xfail(
            "Issue: pyspark.pandas quantile is approximate (lower value, transposed layout)"
        ),
    ],
)
def test_quantile_median_diverges(backend_kind, spark_session) -> None:
    ds = _build(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]}),
        backend_kind,
        spark_session,
    )
    assert float(
        _pdf(ds.quantile(0.5)).to_numpy(dtype=float).ravel()[0]
    ) == pytest.approx(2.5)


@pytest.mark.parametrize(
    "backend_kind",
    [
        PANDAS,
        _spark_xfail(
            "Issue: NaN poisons groupby mean on Spark instead of being skipped"
        ),
    ],
)
def test_groupby_mean_nan_diverges(backend_kind, spark_session) -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "x": [1.0, np.nan, 5.0]})
    ds = _build(df, backend_kind, spark_session)
    result = _pdf(ds.groupby("g").mean()).sort_index()
    assert float(result["x"].iloc[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# File errors
# ---------------------------------------------------------------------------
def _file_error(backend: BackendsEnum, path: str, session):
    with pytest.raises(Exception) as info:
        Dataset(
            roles={},
            data=path,
            backend=backend,
            session=session if backend == BackendsEnum.spark else None,
        )
    return info.value


def test_file_error_types_diverge(tmp_path, spark_session) -> None:
    """Missing file -> FileNotFoundError on both; directory message differs."""
    missing = str(tmp_path / "missing.csv")
    for backend in (BackendsEnum.pandas, BackendsEnum.spark):
        assert isinstance(
            _file_error(backend, missing, spark_session), FileNotFoundError
        )

    pandas_err = _file_error(BackendsEnum.pandas, str(tmp_path), spark_session)
    spark_err = _file_error(BackendsEnum.spark, str(tmp_path), spark_session)
    assert isinstance(pandas_err, ValueError) and isinstance(spark_err, ValueError)
    assert "Unsupported file extension" in str(pandas_err)
    assert "not a file" in str(spark_err)


def test_unsupported_extension_raises_on_both(tmp_path, spark_session) -> None:
    path = tmp_path / "data.unknown"
    path.write_text("a,b\n1,2\n")
    for backend in (BackendsEnum.pandas, BackendsEnum.spark):
        assert isinstance(_file_error(backend, str(path), spark_session), ValueError)
