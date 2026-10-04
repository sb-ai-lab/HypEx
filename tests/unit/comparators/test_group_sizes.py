"""Tests for GroupSizes math."""

from __future__ import annotations

import pytest

from hypex.comparators import GroupSizes
from hypex.dataset import ExperimentData, TreatmentRole

from ._utils import TREAT_TARGET_ROLES, build_dataset, result_frame, three_groups_df


@pytest.mark.parametrize(
    "a,b,pa,pb",
    [
        (10, 30, 25.0, 75.0),
        (50, 50, 50.0, 50.0),
        (0, 10, 0.0, 100.0),
        (7, 0, 100.0, 0.0),
    ],
)
def test_inner_function_shares(a, b, pa, pb) -> None:
    res = GroupSizes._inner_function({"count": a}, {"count": b})
    assert res["control size"] == a
    assert res["test size"] == b
    assert res["control size %"] == pytest.approx(pa, abs=1e-9)
    assert res["test size %"] == pytest.approx(pb, abs=1e-9)


def test_inner_function_empty_groups_give_zero_percent() -> None:
    res = GroupSizes._inner_function({"count": 0}, {"count": 0})
    assert res["control size %"] == 0.0 and res["test size %"] == 0.0


def test_inner_function_missing_count_defaults_to_zero() -> None:
    res = GroupSizes._inner_function({}, {"count": 4})
    assert res["control size"] == 0
    assert res["test size %"] == 100.0


def test_percentages_sum_to_100() -> None:
    res = GroupSizes._inner_function({"count": 13}, {"count": 29})
    assert res["control size %"] + res["test size %"] == pytest.approx(100.0)


def test_execute_unequal_groups(backend, spark_session) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {"g": ["a"] * 30 + ["b"] * 10, "y": [float(i) for i in range(40)]}
    )
    ds = build_dataset(df, TREAT_TARGET_ROLES, backend, spark_session)
    ex = GroupSizes(grouping_role=TreatmentRole())
    out = ex.execute(ExperimentData(ds))
    row = result_frame(out, ex).iloc[0]
    assert row["control size"] == 30
    assert row["test size"] == 10
    assert row["control size %"] == pytest.approx(75.0)
    assert row["test size %"] == pytest.approx(25.0)


def test_execute_three_groups_compares_each_to_baseline(backend, spark_session) -> None:
    ds = build_dataset(
        three_groups_df(n=20), TREAT_TARGET_ROLES, backend, spark_session
    )
    ex = GroupSizes(grouping_role=TreatmentRole())
    table = result_frame(ExperimentData_exec(ex, ds), ex)
    assert len(table) == 2
    assert (table["control size"] == 20).all()
    assert (table["test size"] == 20).all()


def ExperimentData_exec(ex, ds):
    return ex.execute(ExperimentData(ds))


def test_uses_grouping_role_as_target() -> None:
    ex = GroupSizes(grouping_role=TreatmentRole())
    assert ex.target_roles is ex.grouping_role or type(ex.target_roles) is type(
        ex.grouping_role
    )
    assert GroupSizes.REQUIRED_STATS == ["count"]
