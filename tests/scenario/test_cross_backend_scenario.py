"""Pandas vs Spark parity for the public A/B, A/A and Homogeneity tests (I5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex import AATest, ABTest, HomogeneityTest
from hypex.dataset import FeatureRole, InfoRole, TargetRole, TreatmentRole
from hypex.utils import BackendsEnum

from ._utils import ab_frame, make_dataset, resume_row, to_pandas

pytestmark = [pytest.mark.slow, pytest.mark.spark]

TOL = 1e-6
NUMERIC = ["control mean", "test mean", "difference", "difference %", "TTest p-value"]


def _both(df, roles, session):
    return (
        make_dataset(df, roles, BackendsEnum.pandas),
        make_dataset(df, roles, BackendsEnum.spark, session),
    )


def _compare_rows(left: pd.Series, right: pd.Series, columns=NUMERIC) -> None:
    for column in columns:
        assert float(left[column]) == pytest.approx(
            float(right[column]), rel=1e-5, abs=1e-9
        ), column
    for column in left.index:
        if column.endswith("pass"):
            assert left[column] == right[column], column


def test_ab_two_groups_parity(spark_session) -> None:
    df = ab_frame(n=800, effect=0.4, seed=1)
    pandas_ds, spark_ds = _both(df, None, spark_session)
    left = resume_row(ABTest().execute(pandas_ds))
    right = resume_row(ABTest().execute(spark_ds))
    _compare_rows(left, right)


def test_ab_three_groups_and_multitest_parity(spark_session) -> None:
    df = ab_frame(n=1200, effect=0.3, groups=3, seed=2)
    pandas_ds, spark_ds = _both(df, None, spark_session)
    left_out = ABTest(multitest_method="holm").execute(pandas_ds)
    right_out = ABTest(multitest_method="holm").execute(spark_ds)
    for group in (1, 2):
        _compare_rows(
            resume_row(left_out, "y", group), resume_row(right_out, "y", group)
        )
    left = to_pandas(left_out.multitest).sort_values("group")
    right = to_pandas(right_out.multitest).sort_values("group")
    np.testing.assert_allclose(
        left["new p-value"].astype(float), right["new p-value"].astype(float), rtol=1e-5
    )
    assert [str(v) for v in left["rejected"]] == [str(v) for v in right["rejected"]]


def test_ab_group_sizes_parity(spark_session) -> None:
    df = ab_frame(n=700, effect=0.2, seed=3)
    pandas_ds, spark_ds = _both(df, None, spark_session)
    left = to_pandas(ABTest().execute(pandas_ds).sizes)
    right = to_pandas(ABTest().execute(spark_ds).sizes)
    for column in ("control size", "test size"):
        assert int(left[column].iloc[0]) == int(right[column].iloc[0])


ROLES_HOMO = {
    "id": InfoRole(),
    "treat": TreatmentRole(),
    "age": TargetRole(),
    "spend": TargetRole(),
}


def _homo_frame(imbalance: float, seed: int = 4, n: int = 800) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    treat = rng.randint(0, 2, n)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "treat": treat,
            "age": rng.normal(40, 10, n) + imbalance * treat,
            "spend": rng.normal(100, 20, n),
        }
    )


@pytest.mark.xfail(
    strict=True,
    raises=DeprecationWarning,
    reason="Issue: HomogeneityTest() construction emits the DeprecationWarning of the deprecated HomoDatasetReporter (hypex/ui/homo.py)",
)
@pytest.mark.parametrize("imbalance", [0.0, 4.0])
def test_homogeneity_parity(spark_session, imbalance) -> None:
    df = _homo_frame(imbalance)
    pandas_ds, spark_ds = _both(df, ROLES_HOMO, spark_session)
    left = to_pandas(HomogeneityTest().execute(pandas_ds).resume).set_index("feature")
    right = to_pandas(HomogeneityTest().execute(spark_ds).resume).set_index("feature")
    assert set(left.index) == set(right.index)
    for feature in left.index:
        _compare_rows(left.loc[feature], right.loc[feature])
        # KS is exact on pandas and histogram-based on Spark: the verdict must agree
        assert left.loc[feature, "KSTest pass"] == right.loc[feature, "KSTest pass"]


@pytest.mark.xfail(
    strict=True,
    raises=DeprecationWarning,
    reason="Issue: HomogeneityTest() construction emits the DeprecationWarning of the deprecated HomoDatasetReporter (hypex/ui/homo.py)",
)
def test_homogeneity_ks_pvalue_is_close(spark_session) -> None:
    df = _homo_frame(0.0)
    pandas_ds, spark_ds = _both(df, ROLES_HOMO, spark_session)
    left = to_pandas(HomogeneityTest().execute(pandas_ds).resume).set_index("feature")
    right = to_pandas(HomogeneityTest().execute(spark_ds).resume).set_index("feature")
    for feature in left.index:
        assert float(right.loc[feature, "KSTest p-value"]) == pytest.approx(
            float(left.loc[feature, "KSTest p-value"]), abs=0.1
        )


AA_ROLES = {"id": InfoRole(), "x": FeatureRole(), "y": TargetRole()}


def _aa_frame(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {"id": np.arange(n), "x": rng.normal(0, 1, n), "y": rng.normal(10, 2, n)}
    )


def test_aa_runs_on_both_backends_with_the_same_structure(spark_session) -> None:
    df = _aa_frame()
    pandas_ds, spark_ds = _both(df, AA_ROLES, spark_session)
    left = AATest(n_iterations=3, random_states=[1, 2, 3]).execute(pandas_ds)
    right = AATest(n_iterations=3, random_states=[1, 2, 3]).execute(spark_ds)
    assert list(to_pandas(left.experiments).columns) == list(
        to_pandas(right.experiments).columns
    )
    assert len(to_pandas(left.experiments)) == len(to_pandas(right.experiments)) == 3
    assert (
        to_pandas(left.resume).iloc[0]["result"]
        == to_pandas(right.resume).iloc[0]["result"]
    )


@pytest.mark.xfail(
    strict=True,
    reason="Issue: random_split_labels uses MD5 on Pandas and Murmur3 on Spark, so the same "
    "random_states produce different A/A splits (and different iteration statistics)",
)
def test_aa_iterations_are_identical_across_backends(spark_session) -> None:
    df = _aa_frame()
    pandas_ds, spark_ds = _both(df, AA_ROLES, spark_session)
    left = AATest(n_iterations=3, random_states=[1, 2, 3]).execute(pandas_ds)
    right = AATest(n_iterations=3, random_states=[1, 2, 3]).execute(spark_ds)
    pd.testing.assert_frame_equal(
        to_pandas(left.experiments).reset_index(drop=True),
        to_pandas(right.experiments).reset_index(drop=True),
    )


def test_spark_ab_reports_the_group_totals_as_a_set(spark_session) -> None:
    """Even when control/test are swapped, the two group sizes are the true group counts."""
    df = ab_frame(n=700, effect=0.2, seed=3)
    sizes = to_pandas(
        ABTest()
        .execute(make_dataset(df, None, BackendsEnum.spark, spark_session))
        .sizes
    )
    reported = sorted(
        [int(sizes["control size"].iloc[0]), int(sizes["test size"].iloc[0])]
    )
    assert reported == sorted(df.groupby("treat").size().tolist())
