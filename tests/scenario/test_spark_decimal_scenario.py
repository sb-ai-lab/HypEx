"""Spark Decimal support in high-level experiments."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from hypex import AATest, ABTest, HomogeneityTest, Matching
from hypex.dataset import Dataset, FeatureRole, InfoRole, TargetRole, TreatmentRole
from hypex.utils import BackendsEnum

from ._utils import resume_row, to_pandas

pytestmark = [pytest.mark.slow, pytest.mark.spark]

ROLES = {
    "id": InfoRole(),
    "treat": TreatmentRole(),
    "x": FeatureRole(),
    "y": TargetRole(),
}


def _decimal_dataset(spark_session) -> Dataset:
    rows = []
    for i in range(80):
        treat = i % 2
        x_value = i % 20 + 0.125
        rows.append(
            (
                i,
                treat,
                Decimal(f"{x_value:.3f}"),
                Decimal(f"{2 * x_value + treat:.3f}"),
            )
        )
    spark_df = spark_session.createDataFrame(
        rows,
        schema="id bigint, treat int, x decimal(13,3), y decimal(13,3)",
    )
    return Dataset(
        roles=ROLES,
        data=spark_df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )


@pytest.mark.parametrize("float32", [False, True])
def test_aa_decimal_targets_and_features(spark_session, float32) -> None:
    dataset = _decimal_dataset(spark_session)

    output = AATest(n_iterations=1, float32=float32).execute(dataset)
    experiments = to_pandas(output.experiments)

    for feature in ("x", "y"):
        assert f"{feature} TTest p-value test_1" in experiments.columns
        assert f"{feature} KSTest p-value test_1" in experiments.columns
    assert not to_pandas(output.aa_score).empty
    assert dict(dataset.data.dtypes)["y"] == "decimal(13,3)"


@pytest.mark.parametrize("float32", [False, True])
def test_ab_decimal_targets_and_features(spark_session, float32) -> None:
    output = ABTest(float32=float32, additional_tests=["t-test", "ks-test"]).execute(
        _decimal_dataset(spark_session)
    )
    row = resume_row(output, "y", 1)

    assert float(row["difference"]) == pytest.approx(3.0)
    assert pd.notna(row["TTest p-value"])
    assert pd.notna(row["KSTest p-value"])


def test_homogeneity_decimal_targets(spark_session) -> None:
    output = HomogeneityTest().execute(_decimal_dataset(spark_session))
    row = resume_row(output, "y", 1)

    assert float(row["difference"]) == pytest.approx(3.0)
    assert pd.notna(row["TTest p-value"])
    assert pd.notna(row["KSTest p-value"])


def test_matching_decimal_features_and_targets(spark_session) -> None:
    output = Matching(
        distance="l2",
        faiss_mode="base",
        bias_estimation=False,
        quality_tests=["t-test", "ks-test"],
    ).execute(_decimal_dataset(spark_session))
    quality = to_pandas(output.quality_results)

    assert "x" in set(quality["feature"])
    assert not to_pandas(output.resume).empty
