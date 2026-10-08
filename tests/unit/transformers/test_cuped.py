"""Tests for CUPEDTransformer math and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import PreTargetRole, TargetRole
from hypex.transformers import CUPEDTransformer

from ._utils import make_ds, make_ed, to_pandas

TOL = 1e-6
ROLES = {"y": TargetRole(), "x": PreTargetRole()}


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    x = rng.normal(10, 2, 500)
    y = 3.0 * x + rng.normal(0, 1, 500) + 5
    return pd.DataFrame({"y": y, "x": x})


def _expected(df: pd.DataFrame) -> np.ndarray:
    cov = np.mean(df.y * df.x) - df.x.mean() * df.y.mean()
    var = np.mean(df.x * df.x) - df.x.mean() ** 2
    theta = cov / var
    return (df.y - (df.x - df.x.mean()) * theta).to_numpy()


def test_cuped_exact_formula(frame, backend, spark_session) -> None:
    ds = make_ds(frame, ROLES, backend, spark_session)
    out = to_pandas(CUPEDTransformer.calc(ds, {"y": "x"}))
    expected = _expected(frame)
    np.testing.assert_allclose(
        np.sort(out["y_cuped"].to_numpy(dtype=float)), np.sort(expected), atol=TOL
    )


def test_cuped_preserves_mean(frame) -> None:
    out = to_pandas(CUPEDTransformer.calc(make_ds(frame, ROLES), {"y": "x"}))
    assert out["y_cuped"].mean() == pytest.approx(frame.y.mean(), abs=TOL)


def test_cuped_reduces_variance_for_correlated_covariate(frame) -> None:
    out = to_pandas(CUPEDTransformer.calc(make_ds(frame, ROLES), {"y": "x"}))
    assert out["y_cuped"].var() < frame.y.var() * 0.1


def test_cuped_variance_reduction_equals_r_squared(frame) -> None:
    out = to_pandas(CUPEDTransformer.calc(make_ds(frame, ROLES), {"y": "x"}))
    r2 = np.corrcoef(frame.y, frame.x)[0, 1] ** 2
    reduction = 1 - out["y_cuped"].var() / frame.y.var()
    assert reduction == pytest.approx(r2, abs=1e-3)


def test_cuped_adds_column_with_target_role_and_keeps_originals(frame) -> None:
    ds = make_ds(frame, ROLES)
    result = CUPEDTransformer.calc(ds, {"y": "x"})
    assert set(result.columns) == {"y", "x", "y_cuped"}
    assert isinstance(result.roles["y_cuped"], TargetRole)
    np.testing.assert_allclose(
        to_pandas(result)["y"].to_numpy(dtype=float), frame.y.to_numpy()
    )


def test_cuped_does_not_mutate_input(frame) -> None:
    ds = make_ds(frame, ROLES)
    CUPEDTransformer.calc(ds, {"y": "x"})
    assert "y_cuped" not in ds.columns


def test_cuped_zero_variance_covariate_is_identity() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [5.0, 5.0, 5.0, 5.0]})
    out = to_pandas(CUPEDTransformer.calc(make_ds(df, ROLES), {"y": "x"}))
    np.testing.assert_allclose(
        out["y_cuped"].to_numpy(dtype=float), df.y.to_numpy(), atol=TOL
    )


def test_cuped_independent_covariate_barely_changes_variance() -> None:
    rng = np.random.RandomState(7)
    df = pd.DataFrame({"y": rng.normal(0, 1, 2000), "x": rng.normal(0, 1, 2000)})
    out = to_pandas(CUPEDTransformer.calc(make_ds(df, ROLES), {"y": "x"}))
    assert out["y_cuped"].var() == pytest.approx(df.y.var(), rel=0.02)


def test_cuped_multiple_features() -> None:
    rng = np.random.RandomState(1)
    x1, x2 = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
    df = pd.DataFrame(
        {
            "y1": 2 * x1 + rng.normal(0, 0.1, 300),
            "x1": x1,
            "y2": -x2 + rng.normal(0, 0.1, 300),
            "x2": x2,
        }
    )
    roles = {
        "y1": TargetRole(),
        "x1": PreTargetRole(),
        "y2": TargetRole(),
        "x2": PreTargetRole(),
    }
    out = to_pandas(CUPEDTransformer.calc(make_ds(df, roles), {"y1": "x1", "y2": "x2"}))
    assert {"y1_cuped", "y2_cuped"} <= set(out.columns)
    assert out["y1_cuped"].var() < df.y1.var() * 0.05
    assert out["y2_cuped"].var() < df.y2.var() * 0.05


def _reduction(out, executor, feature: str = "y_cuped") -> float:
    """variance_reduction_pct of ``feature`` from the executor's analysis table."""
    report = to_pandas(out.analysis_tables[executor.id])
    return float(
        report.loc[report["feature"] == feature, "variance_reduction_pct"].iloc[0]
    )


def test_cuped_execute_stores_variance_reduction(frame) -> None:
    ed = make_ed(frame, ROLES)
    executor = CUPEDTransformer(cuped_features={"y": "x"})
    out = executor.execute(ed)
    assert "y_cuped" in out.ds.columns
    expected = (1 - np.var(_expected(frame), ddof=1) / frame.y.var()) * 100
    assert _reduction(out, executor) == pytest.approx(expected, abs=1e-4)


def test_cuped_execute_zero_variance_target_gives_zero_reduction() -> None:
    df = pd.DataFrame({"y": [2.0, 2.0, 2.0, 2.0], "x": [1.0, 2.0, 3.0, 4.0]})
    executor = CUPEDTransformer(cuped_features={"y": "x"})
    out = executor.execute(make_ed(df, ROLES))
    assert _reduction(out, executor) == 0.0


def test_cuped_is_transformer() -> None:
    assert CUPEDTransformer({"y": "x"})._is_transformer is True
