"""Tests for SMD, MatchingMetrics and Bias operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from hypex.dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    TargetRole,
    TreatmentRole,
)
from hypex.extensions import PandasMatchingMetricsExtension
from hypex.operators import SMD, Bias, MatchingMetrics
from hypex.utils import BackendsEnum

TOL = 1e-6


def _ds(df: pd.DataFrame, **roles) -> Dataset:
    return Dataset(roles=roles, data=df, backend=BackendsEnum.pandas)


# ---------------------------------------------------------------------------
# SMD
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="Issue: SMD._inner_function computes (mean1 + mean2) / std instead of the "
    "standardized difference (mean_t - mean_c) / pooled std",
)
def test_smd_is_standardized_mean_difference() -> None:
    rng = np.random.RandomState(0)
    a, b = rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)
    ds_a = _ds(pd.DataFrame({"x": a}), x=FeatureRole())
    ds_b = _ds(pd.DataFrame({"x": b}), x=FeatureRole())
    value = float(SMD._inner_function(ds_b, ds_a))
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    assert value == pytest.approx((b.mean() - a.mean()) / pooled, abs=1e-2)


def test_smd_requires_test_data() -> None:
    ds_a = _ds(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), x=FeatureRole())
    with pytest.raises(ValueError, match="test_data"):
        SMD._inner_function(ds_a, None)


# ---------------------------------------------------------------------------
# MatchingMetrics: pure statistics
# ---------------------------------------------------------------------------
def _ext(metric="auto", n_neighbors=1) -> PandasMatchingMetricsExtension:
    return PandasMatchingMetricsExtension(TreatmentRole(), [TargetRole()], metric, n_neighbors)


STATS_C = {"count": 40, "mean": 1.0, "var": 4.0, "sum": 40.0, "sq_sum": 60.0}
STATS_T = {"count": 50, "mean": 2.0, "var": 9.0, "sum": 50.0, "sq_sum": 80.0}


def test_calc_se_formula() -> None:
    se = PandasMatchingMetricsExtension._calc_se(40, 50, 4.0, 9.0, 1.5, 2.0)
    assert se == pytest.approx(np.sqrt(1.5 * 4.0 / 40 + 2.0 * 9.0 / 50), abs=TOL)


def test_calc_metrics_att_formulas() -> None:
    result = _ext("att")._calc_metrics(STATS_C, STATS_T)
    assert list(result) == ["ATT"]
    estimate, se, p, lo, hi = result["ATT"]
    m, n = 40, 50
    expected_se = np.sqrt(m * 60.0 / n**2 * 4.0 / m + 1.0 * 9.0 / n)
    assert estimate == 2.0
    assert se == pytest.approx(expected_se, abs=TOL)
    assert lo == pytest.approx(2.0 - 1.96 * expected_se, abs=TOL)
    assert hi == pytest.approx(2.0 + 1.96 * expected_se, abs=TOL)
    assert p == pytest.approx(2 * norm.sf(abs(2.0 / expected_se)), abs=1e-4)


def test_calc_metrics_atc_formulas() -> None:
    result = _ext("atc")._calc_metrics(STATS_C, STATS_T)
    assert list(result) == ["ATC"]
    estimate, se, _, lo, hi = result["ATC"]
    m, n = 40, 50
    expected_se = np.sqrt(1.0 * 4.0 / m + n * 80.0 / m**2 * 9.0 / n)
    assert estimate == 1.0
    assert se == pytest.approx(expected_se, abs=TOL)
    assert (lo, hi) == pytest.approx((1.0 - 1.96 * expected_se, 1.0 + 1.96 * expected_se), abs=TOL)


def test_calc_metrics_auto_returns_all_three_and_ate_is_weighted_mean() -> None:
    result = _ext("auto")._calc_metrics(STATS_C, STATS_T)
    assert set(result) == {"ATT", "ATC", "ATE"}
    expected_ate = (2.0 * 50 + 1.0 * 40) / 90
    assert result["ATE"][0] == pytest.approx(expected_ate, abs=TOL)
    N = 90
    ate_var = (4.0 * (40 + 2 * 40.0 + 60.0) + 9.0 * (50 + 2 * 50.0 + 80.0)) / N**2
    assert result["ATE"][1] == pytest.approx(np.sqrt(ate_var), abs=TOL)


def test_scaled_counts_divide_by_neighbors() -> None:
    df = pd.DataFrame({"n1": [3, 3, 4], "n2": [4, 5, 5]})
    counts = PandasMatchingMetricsExtension._calc_scaled_counts(df, ["n1", "n2"], 2)
    assert counts.to_dict() == {3: 1.0, 4: 1.0, 5: 1.0}


def test_scaled_counts_single_neighbor() -> None:
    df = pd.DataFrame({"n1": [7, 7, 8]})
    counts = PandasMatchingMetricsExtension._calc_scaled_counts(df, ["n1"], 1)
    assert counts.to_dict() == {7: 2.0, 8: 1.0}


# ---------------------------------------------------------------------------
# MatchingMetrics / Bias end to end on a matched dataset
# ---------------------------------------------------------------------------
@pytest.fixture
def matched():
    """Control rows 0..29, treated rows 30..59; 1-NN match on x in the opposite group."""
    rng = np.random.RandomState(0)
    n = 30
    x_c, x_t = rng.uniform(0, 10, n), rng.uniform(2, 12, n)
    x = np.r_[x_c, x_t]
    treat = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    y = 2.0 * x + 5.0 * treat + rng.normal(0, 1, 2 * n)
    nn = np.empty(2 * n, dtype=int)
    for i in range(2 * n):
        pool = np.where(treat != treat[i])[0]
        nn[i] = pool[np.argmin(np.abs(x[pool] - x[i]))]
    df = pd.DataFrame({"treat": treat, "x": x, "y": y, "nn": nn})
    roles = {
        "treat": TreatmentRole(),
        "x": FeatureRole(),
        "y": TargetRole(),
        "nn": AdditionalMatchingRole(),
    }
    return df, roles


def _expected_effects(df: pd.DataFrame, bias=None):
    y_matched = df.y.to_numpy()[df.nn.to_numpy()]
    bias = np.zeros(len(df)) if bias is None else bias
    it = np.where(df.treat == 1, df.y - y_matched + bias, y_matched - df.y - bias)
    itt, itc = it[df.treat == 1], it[df.treat == 0]
    return itt, itc


def test_matching_metrics_execute_without_bias(matched) -> None:
    df, roles = matched
    operator = MatchingMetrics(grouping_role=TreatmentRole(), metric="auto")
    out = operator.execute(ExperimentData(_ds(df, **roles)))
    result = out.variables[operator.id]
    itt, itc = _expected_effects(df)
    assert result["ATT"][0] == pytest.approx(itt.mean(), abs=TOL)
    assert result["ATC"][0] == pytest.approx(itc.mean(), abs=TOL)
    ate = (itt.sum() + itc.sum()) / (len(itt) + len(itc))
    assert result["ATE"][0] == pytest.approx(ate, abs=TOL)
    # CI is symmetric around the estimate with half-width 1.96 * se
    est, se, _, lo, hi = result["ATT"]
    assert hi - lo == pytest.approx(2 * 1.96 * se, abs=TOL)
    assert (lo + hi) / 2 == pytest.approx(est, abs=TOL)


def test_matching_metrics_att_standard_error_uses_scaled_counts(matched) -> None:
    df, roles = matched
    operator = MatchingMetrics(grouping_role=TreatmentRole(), metric="att")
    out = operator.execute(ExperimentData(_ds(df, **roles)))
    itt, itc = _expected_effects(df)
    counts = pd.Series(df.nn.to_numpy()).value_counts()
    control_ids = np.arange(30)
    sq_c = float((counts.reindex(control_ids).fillna(0) ** 2).sum())
    m, n = len(itc), len(itt)
    # control rows only appear as neighbours of treated rows
    expected_se = np.sqrt(m * sq_c / n**2 * itc.var(ddof=1) / m + itt.var(ddof=1) / n)
    assert out.variables[operator.id]["ATT"][1] == pytest.approx(expected_se, abs=TOL)


def test_matching_metrics_recovers_true_effect(matched) -> None:
    df, roles = matched
    operator = MatchingMetrics(grouping_role=TreatmentRole(), metric="att")
    out = operator.execute(ExperimentData(_ds(df, **roles)))
    estimate = out.variables[operator.id]["ATT"][0]
    naive = df.y[df.treat == 1].mean() - df.y[df.treat == 0].mean()
    # confounded by x: naive diff is biased upward, matching is much closer to 5
    assert abs(estimate - 5.0) < abs(naive - 5.0)
    assert abs(estimate - 5.0) < 1.5


@pytest.mark.parametrize("metric,keys", [("att", {"ATT"}), ("atc", {"ATC"}), ("auto", {"ATT", "ATC", "ATE"})])
def test_matching_metrics_metric_selects_outputs(matched, metric, keys) -> None:
    df, roles = matched
    operator = MatchingMetrics(grouping_role=TreatmentRole(), metric=metric)
    out = operator.execute(ExperimentData(_ds(df, **roles)))
    assert set(out.variables[operator.id]) == keys


def test_matching_metrics_default_metric_is_auto() -> None:
    assert MatchingMetrics().metric == "auto"


def test_matching_metrics_without_neighbors_raises(matched) -> None:
    df, roles = matched
    roles = {k: v for k, v in roles.items() if k != "nn"}
    operator = MatchingMetrics(grouping_role=TreatmentRole())
    with pytest.raises(Exception):
        operator.execute(ExperimentData(_ds(df.drop(columns="nn"), **roles)))


@pytest.mark.spark
def test_matching_metrics_pandas_spark_parity(matched, spark_session) -> None:
    df, roles = matched
    results = {}
    for backend in (BackendsEnum.pandas, BackendsEnum.spark):
        ds = Dataset(
            roles=dict(roles),
            data=df.copy(),
            backend=backend,
            session=spark_session if backend == BackendsEnum.spark else None,
        )
        operator = MatchingMetrics(grouping_role=TreatmentRole(), metric="auto")
        results[backend] = operator.execute(ExperimentData(ds)).variables[operator.id]
    for key in ("ATT", "ATC", "ATE"):
        np.testing.assert_allclose(
            results[BackendsEnum.pandas][key], results[BackendsEnum.spark][key], rtol=1e-5
        )


def test_bias_execute_matches_numpy_ols(matched) -> None:
    df, roles = matched
    operator = Bias(grouping_role=TreatmentRole(), target_roles=[TargetRole()])
    out = operator.execute(ExperimentData(_ds(df, **roles)))
    table = out.additional_fields.backend_data.data
    bias_col = [c for c in table.columns if "bias" in str(c)][0]
    bias = table[bias_col].reindex(df.index).to_numpy(dtype=float)

    nn = df.nn.to_numpy()
    x_m, y_m = df.x.to_numpy()[nn], df.y.to_numpy()[nn]
    expected = np.zeros(len(df))
    for group in (0, 1):
        mask = (df.treat == group).to_numpy()
        design = np.c_[np.ones(mask.sum()), x_m[mask]]
        coef = np.linalg.lstsq(design, y_m[mask], rcond=None)[0][1]
        expected[mask] = (x_m[mask] - df.x.to_numpy()[mask]) * coef
    np.testing.assert_allclose(bias, expected, atol=TOL)


def test_bias_correction_reduces_matching_discrepancy(matched) -> None:
    """With the bias term, ATT moves towards the true effect (5)."""
    df, roles = matched
    plain = MatchingMetrics(grouping_role=TreatmentRole(), metric="att")
    est_plain = plain.execute(ExperimentData(_ds(df, **roles))).variables[plain.id]["ATT"][0]

    data = ExperimentData(_ds(df, **roles))
    data = Bias(grouping_role=TreatmentRole(), target_roles=[TargetRole()]).execute(data)
    corrected = MatchingMetrics(grouping_role=TreatmentRole(), metric="att")
    est_corrected = corrected.execute(data).variables[corrected.id]["ATT"][0]
    assert abs(est_corrected - 5.0) <= abs(est_plain - 5.0) + 1e-9


def test_group_operator_requires_two_targets() -> None:
    with pytest.raises(ValueError, match="2 targets"):
        SMD._execute_inner_function([], target_fields=["only_one"])
