"""Tests for GroupDifference math (difference, percentage, 95% CI)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hypex.comparators import GroupDifference
from hypex.dataset import ExperimentData, TreatmentRole

from ._utils import (
    TREAT_TARGET_ROLES,
    build_dataset,
    result_frame,
    three_groups_df,
)

TOL = 1e-6


def _stats(mean, var, count):
    return {"mean": mean, "var": var, "count": count}


def test_inner_function_exact_formula() -> None:
    base, comp = _stats(10.0, 4.0, 100), _stats(12.0, 9.0, 50)
    res = GroupDifference._inner_function(base, comp)

    se = math.sqrt(4.0 / 100 + 9.0 / 50)
    assert res["control mean"] == 10.0
    assert res["test mean"] == 12.0
    assert res["difference"] == pytest.approx(2.0, abs=TOL)
    assert res["difference %"] == pytest.approx(20.0, abs=TOL)
    assert res["ci lower"] == pytest.approx(2.0 - 1.96 * se, abs=TOL)
    assert res["ci upper"] == pytest.approx(2.0 + 1.96 * se, abs=TOL)


def test_inner_function_negative_difference() -> None:
    res = GroupDifference._inner_function(_stats(10.0, 1.0, 10), _stats(5.0, 1.0, 10))
    assert res["difference"] == pytest.approx(-5.0, abs=TOL)
    assert res["difference %"] == pytest.approx(-50.0, abs=TOL)


def test_inner_function_zero_control_mean_has_no_percentage() -> None:
    res = GroupDifference._inner_function(_stats(0.0, 1.0, 10), _stats(3.0, 1.0, 10))
    assert res["difference"] == 3.0
    assert res["difference %"] is None
    assert res["ci lower"] is not None


@pytest.mark.parametrize("count", [0, 1])
def test_inner_function_small_sample_has_no_ci(count) -> None:
    res = GroupDifference._inner_function(_stats(1.0, 1.0, count), _stats(2.0, 1.0, 10))
    assert res["difference"] == 1.0
    assert res["ci lower"] is None and res["ci upper"] is None


def test_inner_function_missing_means_returns_none_difference() -> None:
    res = GroupDifference._inner_function({"mean": None}, _stats(2.0, 1.0, 10))
    assert res["difference"] is None
    assert res["difference %"] is None


def test_inner_function_ci_contains_difference() -> None:
    res = GroupDifference._inner_function(_stats(1.0, 2.0, 30), _stats(3.0, 2.0, 30))
    assert res["ci lower"] < res["difference"] < res["ci upper"]


def test_required_stats_and_search_types() -> None:
    assert GroupDifference.REQUIRED_STATS == ["mean", "var", "count"]
    assert set(GroupDifference().search_types) >= {int, float}


def test_execute_matches_numpy(backend, spark_session) -> None:
    df = three_groups_df()
    ds = build_dataset(df, TREAT_TARGET_ROLES, backend, spark_session)
    ex = GroupDifference(grouping_role=TreatmentRole())
    out = ex.execute(ExperimentData(ds))
    table = result_frame(out, ex).sort_index()

    ctrl = df.y[df.g == "a"]
    for group in ("b", "c"):
        test = df.y[df.g == group]
        row = table.loc[f"{group}┆y"]
        diff = test.mean() - ctrl.mean()
        se = math.sqrt(ctrl.var() / len(ctrl) + test.var() / len(test))
        assert row["control mean"] == pytest.approx(ctrl.mean(), abs=TOL)
        assert row["test mean"] == pytest.approx(test.mean(), abs=TOL)
        assert row["difference"] == pytest.approx(diff, abs=TOL)
        assert row["difference %"] == pytest.approx((test.mean() / ctrl.mean() - 1) * 100, abs=1e-4)
        assert row["ci lower"] == pytest.approx(diff - 1.96 * se, abs=TOL)
        assert row["ci upper"] == pytest.approx(diff + 1.96 * se, abs=TOL)


def test_execute_stores_stats_table(backend, spark_session) -> None:
    ds = build_dataset(three_groups_df(), TREAT_TARGET_ROLES, backend, spark_session)
    ex = GroupDifference(grouping_role=TreatmentRole())
    out = ex.execute(ExperimentData(ds))
    stats_ids = [k for k in out.analysis_tables if k.startswith(ex.id) and k.endswith("stats")]
    assert len(stats_ids) == 1


def test_calc_with_group_col_stats() -> None:
    stats = {
        "a": {"y": _stats(1.0, 1.0, 10)},
        "b": {"y": _stats(2.0, 1.0, 10)},
    }
    result = GroupDifference.calc(group_col_stats=stats)
    assert len(result) == 1
    assert float(to_val(result[0], "difference")) == pytest.approx(1.0, abs=TOL)


def to_val(ds, column):
    from ._utils import to_pandas

    return to_pandas(ds)[column].iloc[0]


def test_calc_single_group_returns_empty() -> None:
    assert GroupDifference.calc(group_col_stats={"a": {"y": _stats(1.0, 1.0, 5)}}) == []


def test_calc_requires_data_or_stats() -> None:
    with pytest.raises(ValueError):
        GroupDifference.calc()


def test_constant_groups_zero_difference() -> None:
    res = GroupDifference._inner_function(_stats(5.0, 0.0, 10), _stats(5.0, 0.0, 10))
    assert res["difference"] == 0.0
    assert res["ci lower"] == res["ci upper"] == 0.0
    assert np.isfinite(res["ci lower"])
