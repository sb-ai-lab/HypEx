"""Stats-based hypothesis tests (pre-aggregated inputs) vs ``scipy.stats``."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hypex.comparators import StatsChi2Test, StatsKSTest, StatsTTest, StatsZTest
from hypex.dataset import ExperimentData, TargetRole, TreatmentRole

from ._utils import build_dataset, result_frame, three_groups_df

TOL = 1e-6


def _agg(x):
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "count": len(x)}


@pytest.fixture
def equal_var_samples():
    rng = np.random.RandomState(3)
    return rng.normal(0, 1.0, 50), rng.normal(0.5, 1.1, 70)


@pytest.fixture
def unequal_var_samples():
    rng = np.random.RandomState(4)
    return rng.normal(0, 1.0, 40), rng.normal(0.5, 3.0, 90)


# ---------------------------------------------------------------------------
# StatsTTest
# ---------------------------------------------------------------------------
def test_ttest_similar_variances_uses_pooled_student(equal_var_samples) -> None:
    a, b = equal_var_samples
    res = StatsTTest._inner_function(_agg(a), _agg(b))
    ref = stats.ttest_ind(a, b, equal_var=True)
    assert res["statistic"] == pytest.approx(ref.statistic, abs=TOL)
    assert res["p-value"] == pytest.approx(ref.pvalue, abs=TOL)


def test_ttest_dissimilar_variances_uses_welch(unequal_var_samples) -> None:
    a, b = unequal_var_samples
    res = StatsTTest._inner_function(_agg(a), _agg(b))
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert res["statistic"] == pytest.approx(ref.statistic, abs=TOL)
    assert res["p-value"] == pytest.approx(ref.pvalue, abs=TOL)


@pytest.mark.parametrize("equal_variance", [True, False])
def test_ttest_equal_variance_override(unequal_var_samples, equal_variance) -> None:
    a, b = unequal_var_samples
    res = StatsTTest._inner_function(_agg(a), _agg(b), equal_variance=equal_variance)
    ref = stats.ttest_ind(a, b, equal_var=equal_variance)
    assert res["p-value"] == pytest.approx(ref.pvalue, abs=TOL)


def test_ttest_pass_flag_follows_reliability(equal_var_samples) -> None:
    a, b = equal_var_samples
    p = stats.ttest_ind(a, b).pvalue
    loose = StatsTTest._inner_function(_agg(a), _agg(b), reliability=min(1.0, 2 * p))
    strict = StatsTTest._inner_function(_agg(a), _agg(b), reliability=p / 2)
    assert bool(loose["pass"]) is True
    assert bool(strict["pass"]) is False


@pytest.mark.parametrize("n1,n2", [(1, 10), (10, 1), (0, 0)])
def test_ttest_tiny_samples_return_none(n1, n2) -> None:
    res = StatsTTest._inner_function(
        {"mean": 1.0, "std": 1.0, "count": n1}, {"mean": 2.0, "std": 1.0, "count": n2}
    )
    assert res == {"p-value": None, "statistic": None, "pass": None}


def test_ttest_zero_variance_equal_means() -> None:
    res = StatsTTest._inner_function(
        {"mean": 3.0, "std": 0.0, "count": 10}, {"mean": 3.0, "std": 0.0, "count": 12}
    )
    assert res == {"p-value": 1.0, "statistic": 0.0, "pass": False}


def test_ttest_zero_variance_different_means() -> None:
    res = StatsTTest._inner_function(
        {"mean": 3.0, "std": 0.0, "count": 10}, {"mean": 4.0, "std": 0.0, "count": 12}
    )
    assert res["p-value"] == 0.0 and res["statistic"] == math.inf and res["pass"] is True


def test_ttest_one_zero_variance_falls_back_to_welch() -> None:
    a = {"mean": 3.0, "std": 0.0, "count": 10}
    b = {"mean": 4.0, "std": 2.0, "count": 12}
    res = StatsTTest._inner_function(a, b)
    se = math.sqrt(0.0 / 10 + 4.0 / 12)
    assert res["statistic"] == pytest.approx((3.0 - 4.0) / se, abs=TOL)
    df = (4.0 / 12) ** 2 / ((4.0 / 12) ** 2 / 11)
    assert res["p-value"] == pytest.approx(2 * stats.t.sf(abs(res["statistic"]), df), abs=TOL)


def test_ttest_statistic_sign_is_baseline_minus_compared() -> None:
    res = StatsTTest._inner_function(
        {"mean": 1.0, "std": 1.0, "count": 30}, {"mean": 3.0, "std": 1.0, "count": 30}
    )
    assert res["statistic"] < 0


def test_ttest_degrees_of_freedom_formulas() -> None:
    assert StatsTTest._degrees_of_freedom((10, 20), similar_var=True) == 28
    welch = StatsTTest._degrees_of_freedom((10, 20), (4.0, 9.0), similar_var=False)
    expected = (4 / 10 + 9 / 20) ** 2 / ((4 / 10) ** 2 / 9 + (9 / 20) ** 2 / 19)
    assert welch == pytest.approx(expected, abs=TOL)


def test_ttest_execute_end_to_end(backend, spark_session) -> None:
    df = three_groups_df()
    ds = build_dataset(df, {"g": TreatmentRole(), "y": TargetRole()}, backend, spark_session)
    ex = StatsTTest(grouping_role=TreatmentRole())
    table = result_frame(ex.execute(ExperimentData(ds)), ex)
    base = df.y[df.g == "a"]
    for group in ("b", "c"):
        comp = df.y[df.g == group]
        res = StatsTTest._inner_function(_agg(base), _agg(comp))
        row = table.loc[f"{group}┆y"]
        assert row["p-value"] == pytest.approx(res["p-value"], abs=TOL)
        assert row["statistic"] == pytest.approx(res["statistic"], abs=TOL)


# ---------------------------------------------------------------------------
# StatsZTest (two proportions)
# ---------------------------------------------------------------------------
def test_ztest_statistic_matches_pooled_proportion_formula() -> None:
    res = StatsZTest._inner_function({"count": 200, "sum": 60}, {"count": 300, "sum": 120})
    p1, p2, pool = 60 / 200, 120 / 300, 180 / 500
    z = (p1 - p2) / math.sqrt(pool * (1 - pool) * (1 / 200 + 1 / 300))
    assert res["statistic"][0] == pytest.approx(z, abs=TOL)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: StatsZTest takes p-value from Student t (df=n1+n2-2) instead of the "
    "standard normal, and wraps results in lists",
)
def test_ztest_pvalue_uses_normal_distribution() -> None:
    res = StatsZTest._inner_function({"count": 20, "sum": 6}, {"count": 30, "sum": 18})
    p1, p2, pool = 6 / 20, 18 / 30, 24 / 50
    z = (p1 - p2) / math.sqrt(pool * (1 - pool) * (1 / 20 + 1 / 30))
    assert res["p-value"] == pytest.approx(2 * stats.norm.sf(abs(z)), abs=TOL)


@pytest.mark.parametrize(
    "a,b",
    [
        ({"count": 0, "sum": 0}, {"count": 10, "sum": 5}),
        ({"count": 10, "sum": 0}, {"count": 10, "sum": 0}),
        ({"count": 10, "sum": 10}, {"count": 10, "sum": 10}),
    ],
)
def test_ztest_degenerate_inputs_return_none(a, b) -> None:
    assert StatsZTest._inner_function(a, b) == {"p-value": None, "statistic": None, "pass": None}


# ---------------------------------------------------------------------------
# StatsKSTest (bucketed)
# ---------------------------------------------------------------------------
def test_kstest_statistic_is_max_cdf_gap() -> None:
    base = {"histogram": {0: 10, 1: 10, 2: 0}, "count": 20}
    comp = {"histogram": {0: 0, 1: 10, 2: 10}, "count": 20}
    res = StatsKSTest._inner_function(base, comp)
    assert res["statistic"] == pytest.approx(0.5, abs=TOL)
    en = math.sqrt(20 * 20 / 40)
    expected_p = stats.kstwobign.sf((en + 0.12 + 0.11 / en) * 0.5)
    assert res["p-value"] == pytest.approx(expected_p, abs=TOL)
    assert res["pass"] == (expected_p < 0.05)


def test_kstest_matches_scipy_statistic_for_fine_buckets() -> None:
    rng = np.random.RandomState(5)
    a, b = rng.normal(0, 1, 500), rng.normal(0.3, 1, 600)
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 2001)
    ha = dict(enumerate(np.histogram(a, edges)[0]))
    hb = dict(enumerate(np.histogram(b, edges)[0]))
    res = StatsKSTest._inner_function(
        {"histogram": ha, "count": len(a)}, {"histogram": hb, "count": len(b)}
    )
    assert res["statistic"] == pytest.approx(stats.ks_2samp(a, b).statistic, abs=5e-3)


@pytest.mark.parametrize("n1,n2", [(0, 5), (5, 0)])
def test_kstest_empty_group_returns_none(n1, n2) -> None:
    res = StatsKSTest._inner_function(
        {"histogram": {0: n1}, "count": n1}, {"histogram": {0: n2}, "count": n2}
    )
    assert res == {"p-value": None, "statistic": None, "pass": None}


@pytest.mark.xfail(
    strict=True,
    reason="Issue: identical distributions give p=1.0 but pass=True; 'pass' elsewhere "
    "means p < reliability (difference detected)",
)
def test_kstest_identical_distributions_do_not_pass() -> None:
    h = {"histogram": {0: 5, 1: 5}, "count": 10}
    assert StatsKSTest._inner_function(h, dict(h))["pass"] is False


# ---------------------------------------------------------------------------
# StatsChi2Test
# ---------------------------------------------------------------------------
def test_chi2_matches_scipy_contingency() -> None:
    res = StatsChi2Test._inner_function(
        {"value_counts": {"u": 30, "v": 10}}, {"value_counts": {"u": 15, "v": 25}}
    )
    ref = stats.chi2_contingency(np.array([[30, 10], [15, 25]]))
    assert res["statistic"] == pytest.approx(ref[0], abs=TOL)
    assert res["p-value"] == pytest.approx(ref[1], abs=TOL)
    assert res["pass"] == (ref[1] < 0.05)


def test_chi2_union_of_categories_is_used() -> None:
    res = StatsChi2Test._inner_function(
        {"value_counts": {"u": 20, "v": 20}}, {"value_counts": {"u": 20, "w": 20}}
    )
    ref = stats.chi2_contingency(np.array([[20, 20, 0], [20, 0, 20]]))
    # column order is a set order -> compare the order-invariant statistic
    assert res["statistic"] == pytest.approx(ref[0], abs=TOL)


@pytest.mark.parametrize(
    "a,b", [({}, {}), ({}, {"u": 3}), ({"u": 3}, {})]
)
def test_chi2_empty_inputs_return_none(a, b) -> None:
    res = StatsChi2Test._inner_function({"value_counts": a}, {"value_counts": b})
    assert res == {"p-value": None, "statistic": None, "pass": None}


def test_chi2_single_category_returns_identical() -> None:
    res = StatsChi2Test._inner_function({"value_counts": {"u": 5}}, {"value_counts": {"u": 9}})
    assert res["p-value"] == 1.0 and res["statistic"] == 0.0
