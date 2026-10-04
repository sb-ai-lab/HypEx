"""Matching scenarios on confounded data (I4).

Matching is exercised on Spark only: the pandas path of ``MatchingOutput`` is broken
(see ``test_matching_on_pandas``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex import Matching
from hypex.dataset import FeatureRole, InfoRole, TargetRole, TreatmentRole
from hypex.utils import BackendsEnum

from ._utils import ab_frame, make_dataset, to_pandas

pytestmark = [pytest.mark.slow, pytest.mark.spark]

TRUE_EFFECT = 2.0
ROLES = {
    "id": InfoRole(),
    "treat": TreatmentRole(),
    "x": FeatureRole(),
    "y": TargetRole(),
}


@pytest.fixture(scope="module")
def confounded() -> pd.DataFrame:
    return ab_frame(n=600, effect=TRUE_EFFECT, confounded=True, seed=0)


def _naive(df: pd.DataFrame) -> float:
    return df.y[df.treat == 1].mean() - df.y[df.treat == 0].mean()


@pytest.fixture(scope="module")
def run(spark_session, confounded):
    cache: dict = {}

    def _run(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in cache:
            ds = make_dataset(confounded, ROLES, BackendsEnum.spark, spark_session)
            cache[key] = Matching(**kwargs).execute(ds)
        return cache[key]

    return _run


def _effects(output) -> pd.DataFrame:
    return to_pandas(output.resume).astype(float)


def test_default_matching_recovers_true_effect(run, confounded) -> None:
    effects = _effects(run())
    assert set(effects.index) == {"ATT", "ATC", "ATE"}
    for name in ("ATT", "ATC", "ATE"):
        assert effects.loc[name, "Effect Size"] == pytest.approx(TRUE_EFFECT, abs=0.5)
    assert (
        abs(effects.loc["ATE", "Effect Size"] - TRUE_EFFECT)
        < abs(_naive(confounded) - TRUE_EFFECT) / 3
    )


def test_confidence_interval_covers_true_effect(run) -> None:
    row = _effects(run()).loc["ATE"]
    assert row["CI Lower"] <= TRUE_EFFECT <= row["CI Upper"]


def test_confidence_interval_is_centered_on_estimate(run) -> None:
    for name, row in _effects(run()).iterrows():
        assert (row["CI Lower"] + row["CI Upper"]) / 2 == pytest.approx(
            row["Effect Size"], abs=0.02
        )
        assert row["CI Upper"] - row["CI Lower"] == pytest.approx(
            2 * 1.96 * row["Standard Error"], abs=0.03
        )


def test_real_effect_is_significant(run) -> None:
    assert (_effects(run())["P-value"] < 0.01).all()


def test_distance_modes_agree_for_a_single_feature(run) -> None:
    mahalanobis = _effects(run(distance="mahalanobis"))
    l2 = _effects(run(distance="l2"))
    pd.testing.assert_frame_equal(mahalanobis, l2)


def test_bias_estimation_changes_estimate_but_stays_near_truth(run) -> None:
    with_bias = _effects(run(bias_estimation=True)).loc["ATE", "Effect Size"]
    without_bias = _effects(run(bias_estimation=False)).loc["ATE", "Effect Size"]
    assert with_bias != without_bias
    assert abs(with_bias - TRUE_EFFECT) <= abs(without_bias - TRUE_EFFECT) + 0.05


def test_more_neighbors_shrink_standard_error(run) -> None:
    one = _effects(run(n_neighbors=1)).loc["ATE", "Standard Error"]
    three = _effects(run(n_neighbors=3)).loc["ATE", "Standard Error"]
    assert three < one


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Issue: Matching().execute returns empty `indexes` (0 rows) with the default "
    "compute_indexes=False",
)
def test_indexes_cover_every_row(run, confounded) -> None:
    indexes = to_pandas(run().indexes)
    assert len(indexes) == len(confounded)


def test_matched_neighbors_come_from_the_opposite_group(run, confounded) -> None:
    indexes = to_pandas(run().indexes)
    treat = confounded.treat
    for column in indexes.columns:
        matched = indexes[column].astype(int)
        valid = matched >= 0
        assert (
            treat.loc[matched[valid].to_numpy()].to_numpy()
            != treat.loc[indexes.index[valid]].to_numpy()
        ).all()


@pytest.mark.xfail(
    strict=True,
    reason="Issue: see test_matched_neighbors_come_from_the_opposite_group (neighbour ids in "
    "the output are not the nearest neighbours on x)",
)
def test_matched_neighbors_are_nearest_on_the_covariate(run, confounded) -> None:
    indexes = to_pandas(run().indexes).iloc[:, 0].astype(int).reindex(confounded.index)
    x, t = confounded.x.to_numpy(), confounded.treat.to_numpy()
    expected = np.array(
        [
            np.where(t != t[i])[0][np.argmin(np.abs(x[t != t[i]] - x[i]))]
            for i in range(len(x))
        ]
    )
    assert (indexes.to_numpy() == expected).mean() > 0.95


def test_quality_table_contains_balance_diagnostics(run) -> None:
    quality = to_pandas(run().quality_results)
    assert {"feature", "group"} <= set(quality.columns)
    assert "x" in set(quality["feature"])


def test_matching_on_pandas(confounded) -> None:
    Matching().execute(make_dataset(confounded, ROLES))


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Issue: Matching(group_match=True) fails with 'No group keys passed!' on plain "
    "treatment/target/feature roles",
)
def test_group_match_runs(spark_session, confounded) -> None:
    Matching(group_match=True).execute(
        make_dataset(confounded, ROLES, BackendsEnum.spark, spark_session)
    )
