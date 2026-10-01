"""Backend contracts of extensions: abstract methods and aggregation parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole, TargetRole
from hypex.extensions import (
    BiasExtension,
    DummyEncoderExtension,
    FaissExtension,
    GroupChi2TestExtension,
    MatchingMetricsExtension,
)
from hypex.extensions.abstract import CompareExtension, Extension, MLExtension
from hypex.extensions.stats_hypothesis_testing import StatsAggregationExtension
from hypex.utils import BackendsEnum
from hypex.utils.errors import AbstractMethodError

ALL_STATS = ["mean", "std", "var", "count", "sum", "min", "max"]


def _ds(df, backend, session=None, roles=None) -> Dataset:
    roles = roles or {c: FeatureRole() for c in df.columns}
    return Dataset(
        roles=roles, data=df.copy(), backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


# ---------------------------------------------------------------------------
# Abstract methods
# ---------------------------------------------------------------------------
def test_base_extension_calc_not_implemented() -> None:
    class _Ext(Extension):
        pass

    with pytest.raises(NotImplementedError):
        _Ext().calc(None)


def test_compare_extension_calc_not_implemented() -> None:
    class _Ext(CompareExtension):
        pass

    with pytest.raises(NotImplementedError):
        _Ext().calc(None)


def test_ml_extension_requires_fit_and_predict() -> None:
    with pytest.raises(TypeError):
        MLExtension()  # type: ignore[abstract]


def test_ml_extension_dispatches_modes() -> None:
    calls = []

    class _Ml(MLExtension):
        def fit(self, X, Y=None, **kwargs):
            calls.append("fit")

        def predict(self, X, **kwargs):
            calls.append("predict")

    ext = _Ml()
    ext.calc("data", mode="fit")
    ext.calc("data", mode="auto")
    ext.calc("data", mode="predict")
    ext.calc("data")
    assert calls == ["fit", "fit", "predict", "predict"]


def test_faiss_master_calc_is_abstract() -> None:
    with pytest.raises(TypeError):
        FaissExtension()  # type: ignore[abstract]


def test_bias_master_methods_raise_not_implemented() -> None:
    ext = BiasExtension(None, None)
    with pytest.raises(NotImplementedError):
        ext.calc(None)
    with pytest.raises(NotImplementedError):
        ext._calc_coefs(None)
    with pytest.raises(NotImplementedError):
        BiasExtension.prepare_data(None)
    with pytest.raises(NotImplementedError):
        BiasExtension.calc_bias(None, None, None)


def test_matching_metrics_master_hooks_raise() -> None:
    ext = MatchingMetricsExtension(None, None, "att", 1)
    with pytest.raises(NotImplementedError):
        MatchingMetricsExtension._prepare_data(None, [], [])
    with pytest.raises((NotImplementedError, NotADirectoryError)):
        ext._calc_stats_and_weights(None)


def test_group_chi2_matrix_preparation_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        GroupChi2TestExtension().matrix_preparation(None, None)


def test_dummy_encoder_master_has_no_calc() -> None:
    with pytest.raises(NotImplementedError):
        DummyEncoderExtension().calc(None)


def test_abstract_method_error_is_not_implemented_error() -> None:
    assert issubclass(AbstractMethodError, NotImplementedError)


# ---------------------------------------------------------------------------
# StatsAggregationExtension
# ---------------------------------------------------------------------------
@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "g": ["a"] * 30 + ["b"] * 40 + ["c"] * 20,
            "x": rng.normal(0, 1, 90),
            "y": rng.normal(5, 2, 90),
        }
    )


def test_pandas_aggregation_matches_groupby(frame) -> None:
    result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.pandas), ["g"], ["x", "y"], ALL_STATS
    )
    for group, part in frame.groupby("g"):
        for col in ("x", "y"):
            stats = result[group][col]
            assert stats["mean"] == pytest.approx(part[col].mean(), abs=1e-9)
            assert stats["std"] == pytest.approx(part[col].std(ddof=1), abs=1e-9)
            assert stats["var"] == pytest.approx(part[col].var(ddof=1), abs=1e-9)
            assert stats["count"] == len(part)
            assert stats["sum"] == pytest.approx(part[col].sum(), abs=1e-9)
            assert stats["min"] == part[col].min()
            assert stats["max"] == part[col].max()


def test_pandas_aggregation_orders_groups_alphabetically(frame) -> None:
    result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.pandas), ["g"], ["x"], ["mean"]
    )
    assert list(result) == ["a", "b", "c"]


@pytest.mark.spark
def test_spark_aggregation_matches_pandas(frame, spark_session) -> None:
    pandas_result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.pandas), ["g"], ["x", "y"], ALL_STATS
    )
    spark_result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.spark, spark_session), ["g"], ["x", "y"], ALL_STATS
    )
    assert set(spark_result) == set(pandas_result)
    for group in pandas_result:
        for col in ("x", "y"):
            for stat in ALL_STATS:
                assert float(spark_result[group][col][stat]) == pytest.approx(
                    float(pandas_result[group][col][stat]), rel=1e-6, abs=1e-9
                )


@pytest.mark.spark
def test_spark_aggregation_skips_nan_like_pandas(frame, spark_session) -> None:
    frame = frame.copy()
    frame.loc[[0, 1, 40], "x"] = np.nan
    pandas_result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.pandas), ["g"], ["x"], ["mean", "count", "var"]
    )
    spark_result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.spark, spark_session), ["g"], ["x"], ["mean", "count", "var"]
    )
    for group in pandas_result:
        for stat in ("mean", "count", "var"):
            assert float(spark_result[group]["x"][stat]) == pytest.approx(
                float(pandas_result[group]["x"][stat]), rel=1e-6
            )


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    reason="Issue: Spark aggregation returns groups in collect() order, but StatsComparator "
    "treats the first key as baseline ('alphabetically smallest')",
)
def test_spark_aggregation_orders_groups_alphabetically(spark_session) -> None:
    keys = [f"g{i:02d}" for i in range(12)]
    frame = pd.DataFrame({"g": keys * 5, "x": np.arange(60, dtype=float)})
    result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.spark, spark_session), ["g"], ["x"], ["mean"]
    )
    assert list(result) == sorted(result)


def test_multi_column_grouping_uses_tuple_keys() -> None:
    frame = pd.DataFrame(
        {"g1": ["a", "a", "b", "b"], "g2": [1, 2, 1, 2], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    result = StatsAggregationExtension().calc(
        _ds(frame, BackendsEnum.pandas), ["g1", "g2"], ["x"], ["mean"]
    )
    assert result[("a", 1)]["x"]["mean"] == 1.0
    assert result[("b", 2)]["x"]["mean"] == 4.0
