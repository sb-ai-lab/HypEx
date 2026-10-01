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
