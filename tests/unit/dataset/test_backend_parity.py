"""Backend parity tests: identical data on Pandas and Spark gives identical results.

Every test builds the same ``pd.DataFrame`` on both backends and compares the
results of one ``Dataset`` method. Known, intentional divergences live in
``test_backend_divergences.py`` and are not exercised here (inputs avoid NaN
in aggregations for that reason).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole
from hypex.utils import BackendsEnum

pytestmark = pytest.mark.spark


def _to_pandas(ds: Dataset) -> pd.DataFrame:
    """Return the backend data of ``ds`` as a pandas frame (Spark -> pandas)."""
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def _agg_series(ds: Dataset) -> pd.Series:
    """Normalise an aggregation result to a ``Series`` indexed by column name.

    Pandas returns one row (columns = source columns); Spark returns one
    column (rows = source columns). The layout difference itself is covered
    in ``test_backend_divergences.py``.
    """
    frame = _to_pandas(ds)
    series = frame.iloc[0] if len(frame) == 1 else frame.iloc[:, 0]
    return series.astype(float)


RTOL = 1e-6
ATOL = 1e-6


@pytest.fixture
def source_df() -> pd.DataFrame:
    """Small NaN-free frame with numeric and categorical columns."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [10.0, 20.0, 15.0, 40.0, 35.0, 60.0],
            "k": [1, 2, 3, 4, 5, 6],
            "g": ["a", "b", "a", "b", "a", "b"],
        }
    )


@pytest.fixture
def pair(source_df, spark_session):
    """Return ``(pandas_ds, spark_ds)`` built from the same frame."""
    roles = {c: FeatureRole() for c in source_df.columns}
    pandas_ds = Dataset(roles=dict(roles), data=source_df, backend=BackendsEnum.pandas)
    spark_ds = Dataset(
        roles=dict(roles),
        data=source_df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    return pandas_ds, spark_ds


def _frames(ds_pair, *, sort_cols=None):
    """Convert both datasets of a pair to comparable pandas frames."""
    frames = []
    for ds in ds_pair:
        pdf = _to_pandas(ds) if isinstance(ds, Dataset) else ds
        pdf = pdf.copy()
        if sort_cols:
            pdf = pdf.sort_values(sort_cols)
        frames.append(pdf.reset_index(drop=True))
    return frames


def _assert_frames_close(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Compare two frames column-wise with float tolerance, ignoring column order."""
    assert sorted(left.columns) == sorted(right.columns)
    assert len(left) == len(right)
    for col in left.columns:
        a, b = left[col], right[col]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            np.testing.assert_allclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=RTOL, atol=ATOL
            )
        else:
            assert a.tolist() == b.tolist(), f"column {col!r} differs"


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def test_shape_len_columns_match(pair) -> None:
    p, s = pair
    assert p.shape == s.shape
    assert len(p) == len(s)
    assert sorted(p.columns) == sorted(s.columns)


def test_index_values_match(pair) -> None:
    p, s = pair
    assert list(_to_pandas(p).index) == list(_to_pandas(s).index)


@pytest.mark.parametrize("cols", [["x"], ["y", "x"], ["x", "g"]])
def test_getitem_columns_match(pair, cols) -> None:
    p, s = pair
    left, right = _frames((p[cols], s[cols]))
    assert list(left.columns) == list(right.columns) == cols
    _assert_frames_close(left, right)


def test_get_values_scalar_match(pair) -> None:
    p, s = pair
    left = _agg_series(p[["x", "y"]].mean())
    right = _agg_series(s[["x", "y"]].mean())
    assert left["x"] == pytest.approx(3.5, rel=RTOL)
    assert left["x"] == pytest.approx(right["x"], rel=RTOL)


# ---------------------------------------------------------------------------
# Calc
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["mean", "max", "min", "sum", "std", "var"])
def test_scalar_aggregations_match(pair, method) -> None:
    p, s = pair
    left = _agg_series(getattr(p[["x", "y"]], method)())
    right = _agg_series(getattr(s[["x", "y"]], method)())
    for col in ("x", "y"):
        assert left[col] == pytest.approx(right[col], rel=RTOL, abs=ATOL)


def test_std_default_ddof_match(pair) -> None:
    """Default ``ddof=1`` matches; non-default ddof diverges (see divergences)."""
    p, s = pair
    left = _agg_series(p[["x", "y"]].std())
    right = _agg_series(s[["x", "y"]].std())
    assert left["y"] == pytest.approx(right["y"], rel=RTOL)


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_quantile_boundaries_match(pair, q) -> None:
    """Boundary quantiles are exact on both backends.

    Interior quantiles diverge (layout and interpolation), see
    ``test_backend_divergences.py``.
    """
    p, s = pair
    left = _to_pandas(p[["x", "y"]].quantile(q)).to_numpy(dtype=float).ravel()
    right = _to_pandas(s[["x", "y"]].quantile(q)).to_numpy(dtype=float).ravel()
    np.testing.assert_allclose(np.sort(left), np.sort(right), rtol=RTOL)


def test_corr_match(pair) -> None:
    p, s = pair
    left = _to_pandas(p[["x", "y", "k"]].corr(numeric_only=True))
    right = _to_pandas(s[["x", "y", "k"]].corr(numeric_only=True))
    np.testing.assert_allclose(
        left.loc[["x", "y", "k"], ["x", "y", "k"]].to_numpy(dtype=float),
        right.loc[["x", "y", "k"], ["x", "y", "k"]].to_numpy(dtype=float),
        rtol=RTOL,
        atol=ATOL,
    )


def test_dot_match(pair) -> None:
    p, s = pair
    weights = np.array([[2.0], [0.5]])
    left = _to_pandas(p[["x", "y"]].dot(weights)).to_numpy(dtype=float)
    right = _to_pandas(s[["x", "y"]].dot(weights)).to_numpy(dtype=float)
    np.testing.assert_allclose(left, right, rtol=RTOL)


def test_na_counts_match(source_df, spark_session) -> None:
    df = source_df.copy()
    df.loc[1, "x"] = np.nan
    df.loc[3, "y"] = np.nan
    roles = {c: FeatureRole() for c in df.columns}
    p = Dataset(roles=dict(roles), data=df, backend=BackendsEnum.pandas)
    s = Dataset(
        roles=dict(roles), data=df, backend=BackendsEnum.spark, session=spark_session
    )
    left = _to_pandas(p.na_counts()).iloc[0]
    right = _to_pandas(s.na_counts()).iloc[0]
    assert left[["x", "y", "k", "g"]].tolist() == right[["x", "y", "k", "g"]].tolist()
    assert left["x"] == 1 and left["y"] == 1


def test_dropna_match(source_df, spark_session) -> None:
    df = source_df.copy()
    df.loc[2, "x"] = np.nan
    roles = {c: FeatureRole() for c in df.columns}
    p = Dataset(roles=dict(roles), data=df, backend=BackendsEnum.pandas)
    s = Dataset(
        roles=dict(roles), data=df, backend=BackendsEnum.spark, session=spark_session
    )
    left, right = _frames((p.dropna(), s.dropna()), sort_cols="k")
    assert len(left) == len(right) == len(df) - 1
    _assert_frames_close(left, right)


def test_fillna_match(source_df, spark_session) -> None:
    df = source_df.copy()
    df.loc[2, "x"] = np.nan
    roles = {c: FeatureRole() for c in df.columns}
    p = Dataset(roles=dict(roles), data=df, backend=BackendsEnum.pandas)
    s = Dataset(
        roles=dict(roles), data=df, backend=BackendsEnum.spark, session=spark_session
    )
    left, right = _frames((p.fillna(0.0), s.fillna(0.0)), sort_cols="k")
    assert left["x"].tolist()[2] == 0.0
    _assert_frames_close(left, right)


@pytest.mark.parametrize("ascending", [True, False])
def test_sort_values_match(pair, ascending) -> None:
    p, s = pair
    left = _to_pandas(p.sort(by="y", ascending=ascending))
    right = _to_pandas(s.sort(by="y", ascending=ascending))
    assert left["k"].tolist() == right["k"].tolist()


# ---------------------------------------------------------------------------
# Groupby
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("func", ["mean", "sum", "max", "min"])
def test_groupby_agg_match(pair, func) -> None:
    p, s = pair
    left = _to_pandas(getattr(p.groupby("g"), func)()).sort_index()
    right = _to_pandas(getattr(s.groupby("g"), func)()).sort_index()
    for col in ("x", "y"):
        np.testing.assert_allclose(
            left[col].to_numpy(dtype=float),
            right[col].to_numpy(dtype=float),
            rtol=RTOL,
        )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["inner", "left"])
def test_merge_on_column_match(pair, spark_session, how) -> None:
    p, s = pair
    other_df = pd.DataFrame({"k": [1, 2, 3, 99], "z": [0.1, 0.2, 0.3, 0.9]})
    roles = {"k": FeatureRole(), "z": FeatureRole()}
    p_other = Dataset(roles=dict(roles), data=other_df, backend=BackendsEnum.pandas)
    s_other = Dataset(
        roles=dict(roles),
        data=other_df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    left, right = _frames(
        (p.merge(p_other, on="k", how=how), s.merge(s_other, on="k", how=how)),
        sort_cols="k",
    )
    assert len(left) == len(right)
    assert left["z"].isna().sum() == right["z"].isna().sum()
    assert left["k"].tolist() == right["k"].tolist()
    np.testing.assert_allclose(
        left["z"].to_numpy(dtype=float),
        right["z"].to_numpy(dtype=float),
        rtol=RTOL,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 3, 6])
def test_sample_n_size_and_membership_match(pair, n) -> None:
    p, s = pair
    left = _to_pandas(p.sample(n=n, random_state=7))
    right = _to_pandas(s.sample(n=n, random_state=7))
    assert len(left) == len(right) == n
    # Row *selection* is backend specific; both must only return source rows.
    assert set(left["k"]) <= set(range(1, 7))
    assert set(right["k"]) <= set(range(1, 7))
    assert left["k"].is_unique and right["k"].is_unique


def test_sample_is_reproducible_per_backend(pair) -> None:
    p, s = pair
    for ds in (p, s):
        first = _to_pandas(ds.sample(n=3, random_state=11))["k"].tolist()
        second = _to_pandas(ds.sample(n=3, random_state=11))["k"].tolist()
        assert first == second


def test_random_split_labels_structure_match(pair) -> None:
    """Both backends label rows with the same label set and keep the index."""
    p, s = pair
    mod = 10_000_000
    kwargs = dict(edges=[mod // 2, mod], labels=["A", "B"], random_state=42)
    left = _to_pandas(p.random_split_labels(**kwargs))
    right = _to_pandas(s.random_split_labels(**kwargs))
    assert set(left["split"]) <= {"A", "B"}
    assert set(right["split"]) <= {"A", "B"}
    assert len(left) == len(right) == 6
    assert sorted(left.index) == sorted(right.index)


def _indexed_spark_ds(df: pd.DataFrame, spark_session, partitions: int = 3) -> Dataset:
    """Spark Dataset keeping ``df.index`` on a deterministic multi-partition frame."""
    from pyspark.sql import types as T

    idx_col = "__idx__"
    rows = [(int(i), *map(float, r)) for i, r in zip(df.index, df.to_numpy())]
    schema = T.StructType(
        [T.StructField(idx_col, T.LongType())]
        + [T.StructField(c, T.DoubleType()) for c in df.columns]
    )
    sdf = spark_session.createDataFrame(
        spark_session.sparkContext.parallelize(rows, partitions), schema
    )
    psdf = sdf.pandas_api(index_col=idx_col)
    psdf.index.name = None
    ds = Dataset(
        roles={c: FeatureRole() for c in df.columns},
        data=psdf,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    assert ds.backend_data.data.to_spark().rdd.getNumPartitions() == partitions
    return ds


def test_index_setter_list_maps_rows_positionally(source_df, spark_session) -> None:
    """A list assigned to ``index`` labels the same rows on both backends."""
    df = source_df[["x", "y"]].copy()
    df.index = [50, 10, 30, 20, 60, 40]
    pandas_ds = Dataset(
        roles={c: FeatureRole() for c in df.columns},
        data=df,
        backend=BackendsEnum.pandas,
    )
    spark_ds = _indexed_spark_ds(df, spark_session)
    new = [100, 101, 102, 103, 104, 105]
    pandas_ds.index = new
    spark_ds.index = new

    left = _to_pandas(pandas_ds).sort_values("x")
    right = _to_pandas(spark_ds).sort_values("x")
    assert left.index.tolist() == right.index.tolist()
    np.testing.assert_allclose(left["x"], right["x"])
    np.testing.assert_allclose(left["y"], right["y"])


def test_index_setter_does_not_mutate_shared_frame(source_df, spark_session) -> None:
    """Assigning an index must not change a ps frame shared with another Dataset."""
    df = source_df[["x", "y"]].copy()
    df.index = [50, 10, 30, 20, 60, 40]
    spark_ds = _indexed_spark_ds(df, spark_session)
    raw = spark_ds.backend_data.data
    before = sorted(raw.to_pandas().index.tolist())

    spark_ds.index = [100, 101, 102, 103, 104, 105]

    assert sorted(raw.to_pandas().index.tolist()) == before
