import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from hypex.dataset import Dataset, InfoRole, TargetRole
from hypex.extensions.stats_hypothesis_testing import (
    StatsAggregationExtension,
    StatsChi2TestExtension,
)
from hypex.utils import BackendsEnum


def _roles():
    return {"g": InfoRole(), "y": TargetRole()}


def _datasets(spark_session, pdf):
    pandas_ds = Dataset(roles=_roles(), data=pdf)
    spark_ds = Dataset(
        roles=_roles(),
        data=pdf,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    return pandas_ds, spark_ds


def test_stats_aggregation_spark_group_order(spark_session):
    pdf = pd.DataFrame({"g": np.arange(60) % 6, "y": np.arange(60.0)})
    pandas_ds, spark_ds = _datasets(spark_session, pdf)
    stats = ["mean", "count"]
    expected = StatsAggregationExtension().calc(pandas_ds, ["g"], ["y"], stats=stats)
    actual = StatsAggregationExtension().calc(spark_ds, ["g"], ["y"], stats=stats)
    assert list(actual) == [0, 1, 2, 3, 4, 5]
    assert list(actual) == list(expected)
    for key in expected:
        for stat in stats:
            assert actual[key]["y"][stat] == pytest.approx(expected[key]["y"][stat])


def test_stats_chi2_spark_group_order(spark_session):
    pdf = pd.DataFrame(
        {"g": np.arange(60) % 6, "y": (np.arange(60) % 4).astype(float)}
    )
    pandas_ds, spark_ds = _datasets(spark_session, pdf)
    expected = StatsChi2TestExtension().calc(
        data=pandas_ds, group_col="g", target_cols=["y"]
    )
    actual = StatsChi2TestExtension().calc(
        data=spark_ds, group_col="g", target_cols=["y"]
    )
    assert list(actual) == [0, 1, 2, 3, 4, 5]
    assert list(actual) == list(expected)
    for key in expected:
        exp_vc = {float(k): int(v) for k, v in expected[key]["y"]["value_counts"].items()}
        act_vc = {float(k): int(v) for k, v in actual[key]["y"]["value_counts"].items()}
        assert act_vc == exp_vc


def _layout(name):
    if name == "cyclic":
        return np.arange(60) % 6
    if name == "reversed":
        return (np.arange(60) % 6)[::-1]
    return np.random.default_rng(1).integers(0, 10, 500)


@pytest.mark.parametrize("layout", ["cyclic", "reversed", "random"])
def test_iter_groups_spark_sorted(spark_session, layout):
    """Guard test: iter_groups order matches pandas groupby (passes at base)."""
    g = _layout(layout)
    pdf = pd.DataFrame({"g": g, "y": np.arange(len(g), dtype=float)})
    pandas_ds, spark_ds = _datasets(spark_session, pdf)
    expected = [k for k, _ in pandas_ds.groupby("g")]
    actual = [k for k, _ in spark_ds.groupby("g")]
    assert actual == expected == sorted(set(g.tolist()))


def _nan_group_pdf():
    g = np.tile([1.0, np.nan, 0.0, 2.0], 15)
    return pd.DataFrame({"g": g, "y": np.arange(60.0) % 4})


def _assert_nan_last(keys):
    keys = list(keys)
    assert keys[:3] == [0.0, 1.0, 2.0]
    assert all(isinstance(k, float) and np.isnan(k) for k in keys[3:])
    assert len(keys) > 3


def test_stats_aggregation_spark_group_order_nan_key(spark_session):
    _, spark_ds = _datasets(spark_session, _nan_group_pdf())
    actual = StatsAggregationExtension().calc(
        spark_ds, ["g"], ["y"], stats=["mean", "count"]
    )
    _assert_nan_last(actual)


def test_stats_chi2_spark_group_order_nan_key(spark_session):
    _, spark_ds = _datasets(spark_session, _nan_group_pdf())
    actual = StatsChi2TestExtension().calc(
        data=spark_ds, group_col="g", target_cols=["y"]
    )
    _assert_nan_last(actual)
