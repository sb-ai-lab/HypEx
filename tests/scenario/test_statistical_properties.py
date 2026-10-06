"""Monte-Carlo checks of the statistical guarantees (I0). All tests are slow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hypex.comparators import StatsTTest
from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    PreTargetRole,
    TargetRole,
    TreatmentRole,
)
from hypex.experiments import Experiment
from hypex.ml import FaissNearestNeighbors
from hypex.operators import MatchingMetrics
from hypex.transformers import CUPEDTransformer
from hypex.utils import BackendsEnum
from hypex.utils.tutorial_data_creation import (
    gen_control_variates_df,
    gen_special_medicine_df,
)

from ._utils import make_dataset, to_pandas

pytestmark = pytest.mark.slow

N_RUNS = 200


def _ttest_pvalue(df: pd.DataFrame) -> float:
    """p-value of the library's StatsTTest between group 0 and group 1."""
    ds = make_dataset(df)
    result = StatsTTest.calc(
        target_fields_data=ds[["y"]],
        group_field_data=ds[["treat"]],
    )
    row = to_pandas(result[0])
    return float(row["p-value"].iloc[0])


def _null_frame(seed: int, n: int = 200, effect: float = 0.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    treat = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    y = rng.normal(0, 1, n) + effect * treat
    return pd.DataFrame({"treat": treat, "y": y})


@pytest.fixture(scope="module")
def null_pvalues() -> np.ndarray:
    return np.array([_ttest_pvalue(_null_frame(seed)) for seed in range(N_RUNS)])


def test_pvalue_uniform_under_null(null_pvalues) -> None:
    statistic, p = stats.kstest(null_pvalues, "uniform")
    assert p > 0.01, f"p-values are not uniform (KS={statistic:.3f}, p={p:.4f})"


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_false_positive_rate_matches_alpha(null_pvalues, alpha) -> None:
    rate = float((null_pvalues < alpha).mean())
    # 99.9% binomial interval around alpha
    sigma = np.sqrt(alpha * (1 - alpha) / N_RUNS)
    assert abs(rate - alpha) < 3.3 * sigma, f"FPR={rate:.3f}, expected {alpha}"


def test_power_is_monotone_in_effect_size() -> None:
    effects = [0.0, 0.2, 0.4, 0.8]
    power = []
    for effect in effects:
        pvals = [
            _ttest_pvalue(_null_frame(1000 + s, effect=effect)) for s in range(N_RUNS)
        ]
        power.append(float(np.mean(np.array(pvals) < 0.05)))
    assert power == sorted(power), power
    assert power[0] < 0.12 and power[-1] > 0.95


def test_power_matches_theory_for_known_effect() -> None:
    """n=200 (100/group), effect 0.4 sd -> theoretical power ~ 0.51."""
    effect, n_per_group = 0.4, 100
    ncp = effect / np.sqrt(2 / n_per_group)
    theory = 1 - stats.nct.cdf(stats.t.ppf(0.975, 198), 198, ncp)
    pvals = [_ttest_pvalue(_null_frame(5000 + s, effect=effect)) for s in range(400)]
    observed = float(np.mean(np.array(pvals) < 0.05))
    assert observed == pytest.approx(theory, abs=0.08)


def test_matching_beats_naive_on_confounded_data() -> None:
    """gen_special_medicine_df: treatment adds +1 to the mean lifetime, but sicker
    patients are treated more often, so the naive difference is biased downwards."""
    true_effect = 1.0
    naive_errors, matched_errors = [], []
    for seed in range(6):
        raw = gen_special_medicine_df(3000, dependent_division=True, random_state=seed)
        raw = raw.rename(
            columns={"experimental_treatment": "treat", "residual_lifetime": "y"}
        )
        naive = raw.y[raw.treat == 1].mean() - raw.y[raw.treat == 0].mean()
        roles = {
            "treat": TreatmentRole(),
            "y": TargetRole(),
            "age": FeatureRole(),
            "disease_degree": FeatureRole(),
        }
        data = ExperimentData(
            Dataset(roles=roles, data=raw.copy(), backend=BackendsEnum.pandas)
        )
        pipeline = Experiment(
            [
                FaissNearestNeighbors(two_sides=True, grouping_role=TreatmentRole()),
                MatchingMetrics(grouping_role=TreatmentRole(), metric="att"),
            ]
        )
        out = pipeline.execute(data)
        key = out.get_one_id(
            MatchingMetrics,
            __import__(
                "hypex.utils", fromlist=["ExperimentDataEnum"]
            ).ExperimentDataEnum.variables,
        )
        att = out.variables[key]["ATT"][0]
        naive_errors.append(abs(naive - true_effect))
        matched_errors.append(abs(att - true_effect))
    assert np.mean(naive_errors) > 1.0  # confounding really biases the naive estimate
    assert np.mean(matched_errors) < 0.5 * np.mean(naive_errors)


def test_cuped_reduces_variance_by_expected_amount() -> None:
    """Var reduction of CUPED equals rho^2 between outcome and covariate."""
    raw = gen_control_variates_df(20_000, dependent_division=False, random_state=0)
    roles = {
        "Treatment": TreatmentRole(),
        "Target": TargetRole(),
        "Target_lag_1": PreTargetRole(),
        "X": FeatureRole(),
        "X_lag_1": FeatureRole(),
    }
    ds = Dataset(roles=roles, data=raw.copy(), backend=BackendsEnum.pandas)
    adjusted = to_pandas(CUPEDTransformer.calc(ds, {"Target": "Target_lag_1"}))[
        "Target_cuped"
    ]
    rho2 = np.corrcoef(raw.Target, raw.Target_lag_1)[0, 1] ** 2
    reduction = 1 - adjusted.var() / raw.Target.var()
    assert reduction == pytest.approx(rho2, abs=0.01)
    assert 0.05 < reduction < 0.25  # sanity: the covariate is only weakly informative
    assert adjusted.mean() == pytest.approx(raw.Target.mean(), rel=1e-9)
