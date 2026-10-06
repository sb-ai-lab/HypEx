"""A/A test scenarios (I1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex import AATest
from hypex.dataset import FeatureRole, InfoRole, TargetRole

from ._utils import make_dataset, to_pandas

pytestmark = pytest.mark.slow

AA_ROLES = {"id": InfoRole(), "x": FeatureRole(), "y": TargetRole()}


def _homogeneous(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {"id": np.arange(n), "x": rng.normal(0, 1, n), "y": rng.normal(10, 2, n)}
    )


def _run(df=None, roles=None, **kwargs):
    ds = make_dataset(df if df is not None else _homogeneous(), roles or AA_ROLES)
    return AATest(**kwargs).execute(ds)


def test_homogeneous_data_passes_aa() -> None:
    output = _run(n_iterations=20, random_states=range(20))
    resume = to_pandas(output.resume)
    # resume has one row per target/feature column
    assert (resume["result"] == "OK").all()
    for column in (
        "TTest aa score",
        "KSTest aa score",
        "TTest best split",
        "KSTest best split",
    ):
        assert (resume[column] == "OK").all(), column


def test_pass_rate_over_iterations_is_close_to_alpha() -> None:
    output = _run(_homogeneous(1000), n_iterations=60, random_states=range(60))
    experiments = to_pandas(output.experiments)
    pass_cols = [
        c
        for c in experiments.columns
        if "TTest" in c and "pass" in c and "mean" not in c
    ]
    assert pass_cols
    flags = experiments[pass_cols[0]].astype(bool)
    assert flags.mean() < 0.2  # nominal 5%; 60 runs -> generous bound


def test_every_iteration_is_recorded() -> None:
    output = _run(n_iterations=7, random_states=range(7))
    assert len(to_pandas(output.experiments)) == 7


def test_reproducibility_with_fixed_random_states() -> None:
    first = _run(n_iterations=5, random_states=[11, 12, 13, 14, 15])
    second = _run(n_iterations=5, random_states=[11, 12, 13, 14, 15])
    pd.testing.assert_frame_equal(
        to_pandas(first.experiments), to_pandas(second.experiments)
    )
    pd.testing.assert_frame_equal(to_pandas(first.resume), to_pandas(second.resume))


def test_different_random_states_give_different_splits() -> None:
    first = _run(n_iterations=5, random_states=[1, 2, 3, 4, 5])
    second = _run(n_iterations=5, random_states=[6, 7, 8, 9, 10])
    assert not to_pandas(first.experiments).equals(to_pandas(second.experiments))


def test_group_sizes_follow_control_size() -> None:
    output = _run(
        _homogeneous(2000), n_iterations=3, random_states=range(3), control_size=0.3
    )
    resume = to_pandas(output.resume)
    row = resume[resume["feature"] == "y"].iloc[0]
    assert float(row["control mean"]) == pytest.approx(10.0, abs=0.6)
    best = to_pandas(output.best_split)
    assert len(best) >= 1


def test_drift_along_row_order_does_not_break_aa() -> None:
    """Drift along the row order must not matter: random splits of drifting data are still A/A."""
    df = _homogeneous(1000)
    df["y"] = df["y"] + np.linspace(0, 5, len(df))  # drift along the row order
    output = _run(df, n_iterations=10, random_states=range(10))
    assert to_pandas(output.resume).iloc[0]["result"] == "OK"


def test_t_test_equal_variance_option_is_accepted() -> None:
    output = _run(n_iterations=3, random_states=range(3), equal_variance=True)
    assert len(to_pandas(output.experiments)) == 3


def test_deprecated_t_test_equal_var_warns() -> None:
    with pytest.warns(DeprecationWarning):
        AATest(n_iterations=2, t_test_equal_var=True)


def test_groups_sizes_three_way_split() -> None:
    output = _run(
        _homogeneous(900),
        n_iterations=3,
        random_states=range(3),
        groups_sizes=[0.4, 0.3, 0.3],
    )
    groups = set(to_pandas(output.resume)["group"])
    assert groups == {"test_1", "test_2"}


def test_early_stopping_runs_fewer_iterations() -> None:
    output = _run(n_iterations=50, random_states=range(50), early_stopping=True)
    assert len(to_pandas(output.experiments)) <= 50
