"""Edge cases and execute() paths of the stats-based hypothesis tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hypex.comparators import (
    StatsChi2Test,
    StatsKSTest,
    StatsTTest,
    StatsUTest,
    StatsZTest,
)
from hypex.dataset import ExperimentData, TargetRole, TreatmentRole
from hypex.utils import BackendsEnum, NoColumnsError
from hypex.utils.errors import NotSuitableFieldError

from ._utils import build_dataset, to_pandas

NONE_RESULT = {"p-value": None, "statistic": None, "pass": None}


def _hist(x, edges):
    return dict(enumerate(np.histogram(x, edges)[0]))


# ---------------------------------------------------------------------------
# StatsUTest._inner_function
# ---------------------------------------------------------------------------
def test_utest_matches_scipy_for_fine_buckets() -> None:
    rng = np.random.RandomState(7)
    a, b = rng.normal(0, 1, 300), rng.normal(0.3, 1, 350)
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 5001)
    res = StatsUTest._inner_function(
        {"histogram": _hist(a, edges), "count": len(a)},
        {"histogram": _hist(b, edges), "count": len(b)},
    )
    ref = stats.mannwhitneyu(a, b)
    # same convention as scipy / GroupUTest: U1 of the baseline sample
    assert res["statistic"] == pytest.approx(ref.statistic, abs=1.0)
    assert res["p-value"] == pytest.approx(ref.pvalue, rel=0.02, abs=1e-6)
    assert res["pass"] == (res["p-value"] < 0.05)


@pytest.mark.parametrize("n1,n2", [(0, 5), (5, 0)])
def test_utest_empty_group_returns_none(n1, n2) -> None:
    res = StatsUTest._inner_function(
        {"histogram": {0: n1}, "count": n1}, {"histogram": {0: n2}, "count": n2}
    )
    assert res == NONE_RESULT


def test_utest_no_buckets_is_identical() -> None:
    res = StatsUTest._inner_function(
        {"histogram": {}, "count": 3}, {"histogram": {}, "count": 4}
    )
    assert res == {"p-value": 1.0, "statistic": 0.0, "pass": False}


def test_utest_single_shared_bucket_is_identical() -> None:
    res = StatsUTest._inner_function(
        {"histogram": {0: 10}, "count": 10}, {"histogram": {0: 10}, "count": 10}
    )
    assert res == {"p-value": 1.0, "statistic": 0.0, "pass": False}


def test_utest_fully_separated_groups_are_significant() -> None:
    res = StatsUTest._inner_function(
        {"histogram": {0: 30}, "count": 30}, {"histogram": {5: 30}, "count": 30}
    )
    assert res["statistic"] == 0.0  # U2 = 0: every baseline value is below compared
    assert res["p-value"] < 1e-6
    assert res["pass"] is True


def test_utest_pass_flag_follows_reliability() -> None:
    base = {"histogram": {0: 10, 1: 10}, "count": 20}
    comp = {"histogram": {0: 6, 1: 14}, "count": 20}
    loose = StatsUTest._inner_function(base, comp, reliability=1.0)
    strict = StatsUTest._inner_function(base, comp, reliability=1e-12)
    assert loose["pass"] is True and strict["pass"] is False
    assert loose["p-value"] == strict["p-value"]


def test_utest_swapping_groups_gives_complementary_statistic_same_p_value() -> None:
    base = {"histogram": {0: 10, 2: 10}, "count": 20}
    comp = {"histogram": {1: 12, 2: 8}, "count": 20}
    a = StatsUTest._inner_function(base, comp)
    b = StatsUTest._inner_function(comp, base)
    assert a["statistic"] + b["statistic"] == pytest.approx(20 * 20)
    assert a["p-value"] == pytest.approx(b["p-value"])


# ---------------------------------------------------------------------------
# StatsKSTest / StatsZTest / StatsChi2Test edge cases
# ---------------------------------------------------------------------------
def test_kstest_no_buckets_is_identical() -> None:
    res = StatsKSTest._inner_function(
        {"histogram": {}, "count": 3}, {"histogram": {}, "count": 4}
    )
    assert res == {"p-value": 1.0, "statistic": 0.0, "pass": True}


def test_kstest_pass_flag_follows_reliability() -> None:
    base = {"histogram": {0: 10, 1: 10}, "count": 20}
    comp = {"histogram": {0: 5, 1: 15}, "count": 20}
    assert StatsKSTest._inner_function(base, comp, reliability=1.0)["pass"] is True
    assert StatsKSTest._inner_function(base, comp, reliability=1e-9)["pass"] is False


@pytest.mark.parametrize("cls", [StatsKSTest, StatsUTest], ids=["ks", "u"])
def test_compute_stats_is_not_supported(cls) -> None:
    with pytest.raises(NotImplementedError):
        cls._compute_stats(None, ["x"])


def test_kstest_and_utest_search_numeric_types_only() -> None:
    assert set(StatsKSTest().search_types) == {int, float}
    assert set(StatsUTest().search_types) == {int, float}


def test_ztest_zero_pooled_variance_returns_none_even_with_counts() -> None:
    # nobody converted in either group
    assert (
        StatsZTest._inner_function({"count": 5, "sum": 0}, {"count": 7, "sum": 0})
        == NONE_RESULT
    )


def test_ztest_symmetric_proportions_have_zero_statistic() -> None:
    res = StatsZTest._inner_function(
        {"count": 100, "sum": 30}, {"count": 100, "sum": 30}
    )
    assert res["statistic"][0] == pytest.approx(0.0)
    assert res["p-value"][0] == pytest.approx(1.0)
    assert res["pass"][0] is np.False_ or res["pass"][0] is False


def test_chi2_drops_all_zero_columns() -> None:
    res = StatsChi2Test._inner_function(
        {"value_counts": {"u": 30, "v": 10, "w": 0}},
        {"value_counts": {"u": 15, "v": 25, "w": 0}},
    )
    ref = stats.chi2_contingency(np.array([[30, 10], [15, 25]]))
    assert res["statistic"] == pytest.approx(ref[0])
    assert res["p-value"] == pytest.approx(ref[1])


def test_chi2_only_one_nonzero_column_is_identical() -> None:
    res = StatsChi2Test._inner_function(
        {"value_counts": {"u": 30, "v": 0}}, {"value_counts": {"u": 15, "v": 0}}
    )
    assert res == {"p-value": 1.0, "statistic": 0.0, "pass": False}


def test_chi2_all_zero_row_returns_none() -> None:
    res = StatsChi2Test._inner_function(
        {"value_counts": {"u": 0, "v": 0}}, {"value_counts": {"u": 3, "v": 4}}
    )
    assert res == NONE_RESULT


def test_chi2_value_error_from_scipy_means_no_difference(monkeypatch) -> None:
    from hypex.comparators import stats_hypothesis_testing as mod

    def boom(*a, **k):
        raise ValueError("bad table")

    monkeypatch.setattr(mod, "chi2_contingency", boom)
    res = StatsChi2Test._inner_function(
        {"value_counts": {"u": 3, "v": 4}}, {"value_counts": {"u": 5, "v": 6}}
    )
    assert res == {"p-value": 1.0, "statistic": 0.0, "pass": False}


def test_chi2_and_ztest_search_types() -> None:
    assert int in StatsZTest().search_types
    assert str in StatsChi2Test().search_types


# ---------------------------------------------------------------------------
# execute() paths
# ---------------------------------------------------------------------------
def _frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    y = np.r_[rng.normal(0, 1, n), rng.normal(0.5, 1.5, n), rng.normal(1.2, 1, n)]
    return pd.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n + ["c"] * n,
            "y": y,
            "cat": np.where(y > 0.3, "u", "v"),
            "bin": (y > 0.3).astype(int),
        }
    )


def _data(df, col, backend, session) -> ExperimentData:
    return ExperimentData(
        build_dataset(
            df[["g", col]], {"g": TreatmentRole(), col: TargetRole()}, backend, session
        )
    )


def _table(out: ExperimentData, key: str) -> pd.DataFrame:
    return to_pandas(out.analysis_tables[key])


@pytest.mark.parametrize(
    ("cls", "master"), [(StatsKSTest, "KSTest"), (StatsUTest, "UTest")], ids=["ks", "u"]
)
def test_pandas_execute_raises_type_error(cls, master) -> None:
    ex = cls(grouping_role=TreatmentRole())
    with pytest.raises(TypeError, match="Spark") as exc:
        ex.execute(_data(_frame(), "y", BackendsEnum.pandas, None))
    assert master in str(exc.value)


@pytest.mark.parametrize("cls", [StatsKSTest, StatsUTest], ids=["ks", "u"])
def test_execute_without_target_columns_raises(cls, spark_session) -> None:
    df = _frame()[["g", "y"]]
    ds = build_dataset(
        df,
        {"g": TreatmentRole(), "y": TreatmentRole()},
        BackendsEnum.spark,
        spark_session,
    )
    with pytest.raises(NoColumnsError):
        cls(grouping_role=TreatmentRole()).execute(ExperimentData(ds))


@pytest.mark.parametrize("cls", [StatsKSTest, StatsUTest], ids=["ks", "u"])
def test_execute_with_two_grouping_columns_raises(cls, spark_session) -> None:
    df = _frame()
    df["g2"] = 1
    ds = build_dataset(
        df[["g", "g2", "y"]],
        {"g": TreatmentRole(), "g2": TreatmentRole(), "y": TargetRole()},
        BackendsEnum.spark,
        spark_session,
    )
    with pytest.raises(NotSuitableFieldError):
        cls(grouping_role=TreatmentRole()).execute(ExperimentData(ds))


@pytest.mark.spark
def test_spark_kstest_execute_matches_inner_function(spark_session) -> None:
    df = _frame()
    ex = StatsKSTest(grouping_role=TreatmentRole(), n_bins=500)
    out = ex.execute(_data(df, "y", BackendsEnum.spark, spark_session))
    table = _table(out, ex.id)
    assert list(table.index) == ["b", "c"]
    for grp in ("b", "c"):
        ref = stats.ks_2samp(df.y[df.g == "a"], df.y[df.g == grp])
        assert table.loc[grp, "statistic"] == pytest.approx(ref.statistic, abs=0.03)
        assert 0.0 <= table.loc[grp, "p-value"] <= 1.0
    assert bool(table.loc["c", "pass"]) is True


@pytest.mark.spark
def test_spark_kstest_single_group_gives_empty_result(spark_session) -> None:
    df = _frame()
    df = df[df.g == "a"]
    ex = StatsKSTest(grouping_role=TreatmentRole())
    out = ex.execute(_data(df, "y", BackendsEnum.spark, spark_session))
    assert len(_table(out, ex.id)) == 0


@pytest.mark.spark
def test_spark_utest_execute(spark_session) -> None:
    df = _frame()
    ex = StatsUTest(grouping_role=TreatmentRole(), n_bins=2000)
    out = ex.execute(_data(df, "y", BackendsEnum.spark, spark_session))
    table = _table(out, ex.id)
    assert list(table.index) == ["b", "c"]
    for grp in ("b", "c"):
        ref = stats.mannwhitneyu(df.y[df.g == "a"], df.y[df.g == grp])
        assert table.loc[grp, "statistic"] == pytest.approx(ref.statistic, rel=0.01)
        assert table.loc[grp, "p-value"] == pytest.approx(ref.pvalue, rel=0.1)


def _big_baseline_frame() -> pd.DataFrame:
    rng = np.random.RandomState(3)
    n = 300
    return pd.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n,
            "y": np.r_[rng.normal(1.0, 1, n), rng.normal(0.0, 1, n)],
        }
    )


@pytest.mark.spark
def test_spark_utest_statistic_matches_scipy_when_baseline_is_larger(
    spark_session,
) -> None:
    df = _big_baseline_frame()
    ex = StatsUTest(grouping_role=TreatmentRole(), n_bins=2000)
    out = ex.execute(_data(df, "y", BackendsEnum.spark, spark_session))
    row = _table(out, ex.id).loc["b"]
    ref = stats.mannwhitneyu(df.y[df.g == "a"], df.y[df.g == "b"])
    assert row["statistic"] == pytest.approx(ref.statistic, rel=1e-3)
    assert row["p-value"] == pytest.approx(ref.pvalue, rel=0.1)


@pytest.mark.spark
@pytest.mark.parametrize("cls", [StatsKSTest, StatsUTest], ids=["ks", "u"])
def test_spark_stats_tests_ignore_nan(cls, spark_session) -> None:
    df = _big_baseline_frame()
    df.loc[5, "y"] = np.nan
    ex = cls(grouping_role=TreatmentRole(), n_bins=2000)
    out = ex.execute(_data(df, "y", BackendsEnum.spark, spark_session))
    row = _table(out, ex.id).loc["b"]
    a, b = df.y[df.g == "a"].dropna(), df.y[df.g == "b"]
    ref = stats.ks_2samp(a, b) if cls is StatsKSTest else stats.mannwhitneyu(a, b)
    assert row["statistic"] == pytest.approx(ref.statistic, rel=0.02)
    assert row["p-value"] == pytest.approx(ref.pvalue, rel=0.5, abs=1e-12)
    assert row["p-value"] < 1e-6  # not the NaN-poisoned 1.0


def test_chi2_execute_matches_scipy(backend, spark_session) -> None:
    df = _frame()
    ex = StatsChi2Test(grouping_role=TreatmentRole())
    out = ex.execute(_data(df, "cat", backend, spark_session))
    table = _table(out, ex.id)
    for grp in ("b", "c"):
        ct = pd.crosstab(df.g[df.g.isin(["a", grp])], df.cat[df.g.isin(["a", grp])])
        ref = stats.chi2_contingency(ct.to_numpy())
        assert table.loc[f"{grp}┆cat", "statistic"] == pytest.approx(ref[0], abs=1e-6)
        assert table.loc[f"{grp}┆cat", "p-value"] == pytest.approx(ref[1], abs=1e-6)


def test_ztest_execute_statistic_matches_formula(backend, spark_session) -> None:
    df = _frame()
    ex = StatsZTest(grouping_role=TreatmentRole())
    out = ex.execute(_data(df, "bin", backend, spark_session))
    table = _table(out, ex.id)
    a = df[df.g == "a"]["bin"]
    for grp in ("b", "c"):
        c = df[df.g == grp]["bin"]
        pool = (a.sum() + c.sum()) / (len(a) + len(c))
        z = (a.mean() - c.mean()) / math.sqrt(
            pool * (1 - pool) * (1 / len(a) + 1 / len(c))
        )
        assert table.loc[f"{grp}┆bin", "statistic"] == pytest.approx(z, abs=1e-6)


def test_ttest_execute_without_target_raises_or_skips(backend, spark_session) -> None:
    df = _frame()
    ds = build_dataset(
        df[["g", "y"]],
        {"g": TreatmentRole(), "y": TreatmentRole()},
        backend,
        spark_session,
    )
    with pytest.raises(Exception):
        StatsTTest(grouping_role=TreatmentRole()).execute(ExperimentData(ds))


# ---------------------------------------------------------------------------
# StatsKSTestExtension
# ---------------------------------------------------------------------------
def test_ks_extension_pandas_reports_counts_only() -> None:
    from hypex.extensions.stats_hypothesis_testing import StatsKSTestExtension

    ds = build_dataset(_frame()[["g", "y"]], {"g": TreatmentRole(), "y": TargetRole()})
    res = StatsKSTestExtension().calc(ds, group_col="g", target_cols=["y"])
    assert set(res) == {"a", "b", "c"}
    assert all(res[g]["y"] == {"histogram": {}, "count": 40} for g in res)


@pytest.mark.spark
def test_ks_extension_spark_histogram_counts_sum_to_group_size(spark_session) -> None:
    from hypex.extensions.stats_hypothesis_testing import StatsKSTestExtension

    ds = build_dataset(
        _frame()[["g", "y"]],
        {"g": TreatmentRole(), "y": TargetRole()},
        BackendsEnum.spark,
        spark_session,
    )
    res = StatsKSTestExtension(n_bins=20).calc(ds, group_col="g", target_cols=["y"])
    for group in ("a", "b", "c"):
        stat = res[group]["y"]
        assert stat["count"] == 40
        assert sum(stat["histogram"].values()) == 40
        assert all(0 <= b < 20 for b in stat["histogram"])


@pytest.mark.spark
def test_ks_extension_spark_constant_column_uses_single_bucket(spark_session) -> None:
    from hypex.extensions.stats_hypothesis_testing import StatsKSTestExtension

    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "y": [3.0] * 10})
    ds = build_dataset(
        df, {"g": TreatmentRole(), "y": TargetRole()}, BackendsEnum.spark, spark_session
    )
    res = StatsKSTestExtension().calc(ds, group_col="g", target_cols=["y"])
    assert res["a"]["y"]["histogram"] == {0: 5}
    assert res["b"]["y"]["histogram"] == {0: 5}
