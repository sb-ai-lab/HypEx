"""Tests for CVFilter, ConstFilter, NanFilter, CorrFilter and OutliersFilter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import FeatureRole, InfoRole, TargetRole
from hypex.utils import BackendsEnum
from hypex.transformers import (
    ConstFilter,
    CorrFilter,
    CVFilter,
    NanFilter,
    OutliersFilter,
)

from ._utils import make_ds, make_ed, to_pandas


def _role_names(ds) -> dict:
    return {c: type(r).__name__ for c, r in ds.roles.items()}


def _is_info(ds, col) -> bool:
    return isinstance(ds.roles[col], InfoRole)


# ---------------------------------------------------------------------------
# ConstFilter
# ---------------------------------------------------------------------------
@pytest.fixture
def const_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "const": [1.0] * 20,
            "mostly": [1.0] * 19 + [2.0],  # 95% share
            "varied": list(range(20)),
        }
    )


def test_const_filter_marks_constant_column_as_info(const_frame, backend, spark_session, request) -> None:
    if backend == BackendsEnum.spark:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="Issue: Spark value_counts(normalize=True) returns mismatched roles "
                "(RoleColumnError)",
            )
        )
    roles = {c: FeatureRole() for c in const_frame}
    ds = make_ds(const_frame, roles, backend, spark_session)
    out = ConstFilter._inner_function(ds, target_cols=list(const_frame), threshold=0.95)
    assert _is_info(out, "const")
    assert not _is_info(out, "varied")


def test_const_filter_threshold_is_strict(const_frame) -> None:
    ds = make_ds(const_frame, {c: FeatureRole() for c in const_frame})
    out = ConstFilter._inner_function(ds, target_cols=["mostly"], threshold=0.95)
    assert not _is_info(out, "mostly")  # 0.95 is not > 0.95
    ds = make_ds(const_frame, {c: FeatureRole() for c in const_frame})
    out = ConstFilter._inner_function(ds, target_cols=["mostly"], threshold=0.9)
    assert _is_info(out, "mostly")


def test_const_filter_execute_uses_role_search(const_frame) -> None:
    roles = {"const": FeatureRole(), "mostly": TargetRole(), "varied": FeatureRole()}
    out = ConstFilter(threshold=0.9).execute(make_ed(const_frame, roles))
    assert _is_info(out.ds, "const")
    assert not _is_info(out.ds, "mostly")  # TargetRole not searched by default


def test_const_filter_keeps_data_unchanged(const_frame) -> None:
    ds = make_ds(const_frame, {c: FeatureRole() for c in const_frame})
    out = ConstFilter._inner_function(ds, target_cols=list(const_frame))
    pd.testing.assert_frame_equal(to_pandas(out).reset_index(drop=True), const_frame, check_dtype=False)


# ---------------------------------------------------------------------------
# NanFilter
# ---------------------------------------------------------------------------
@pytest.fixture
def nan_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mostly_nan": [np.nan] * 9 + [1.0],  # 90%
            "some_nan": [np.nan] * 2 + [1.0] * 8,  # 20%
            "clean": [1.0] * 10,
        }
    )


def test_nan_filter_marks_columns_above_threshold(nan_frame, backend, spark_session, request) -> None:
    if backend == BackendsEnum.spark:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="Issue: Spark isna().sum() yields None, so NaN share cannot be computed",
            )
        )
    ds = make_ds(nan_frame, {c: FeatureRole() for c in nan_frame}, backend, spark_session)
    out = NanFilter._inner_function(ds, target_cols=list(nan_frame), threshold=0.8)
    assert _is_info(out, "mostly_nan")
    assert not _is_info(out, "some_nan")
    assert not _is_info(out, "clean")


@pytest.mark.parametrize("threshold,expected", [(0.1, True), (0.2, False), (0.19, True)])
def test_nan_filter_threshold_boundary(nan_frame, threshold, expected) -> None:
    ds = make_ds(nan_frame, {c: FeatureRole() for c in nan_frame})
    out = NanFilter._inner_function(ds, target_cols=["some_nan"], threshold=threshold)
    assert _is_info(out, "some_nan") is expected


def test_nan_filter_execute(nan_frame) -> None:
    out = NanFilter(threshold=0.5).execute(make_ed(nan_frame, {c: FeatureRole() for c in nan_frame}))
    assert _is_info(out.ds, "mostly_nan")
    assert not _is_info(out.ds, "some_nan")


# ---------------------------------------------------------------------------
# CVFilter
# ---------------------------------------------------------------------------
@pytest.fixture
def cv_frame() -> pd.DataFrame:
    # CV (std/mean): low ~ 0.0002, mid ~ 0.4, high ~ 2.2
    return pd.DataFrame(
        {
            "low": [100.0, 100.01, 99.99, 100.0] * 5,
            "mid": [6.0, 10.0, 14.0, 10.0] * 5,
            "high": [0.1, 0.1, 0.1, 10.0] * 5,
        }
    )


def test_cv_filter_lower_bound(cv_frame) -> None:
    ds = make_ds(cv_frame, {c: FeatureRole() for c in cv_frame})
    out = CVFilter._inner_function(ds, target_cols=list(cv_frame), lower_bound=0.01)
    assert _is_info(out, "low")
    assert not _is_info(out, "mid") and not _is_info(out, "high")


def test_cv_filter_upper_bound(cv_frame) -> None:
    ds = make_ds(cv_frame, {c: FeatureRole() for c in cv_frame})
    out = CVFilter._inner_function(ds, target_cols=list(cv_frame), upper_bound=1.0)
    assert _is_info(out, "high")
    assert not _is_info(out, "low") and not _is_info(out, "mid")


def test_cv_filter_no_bounds_keeps_everything(cv_frame) -> None:
    ds = make_ds(cv_frame, {c: FeatureRole() for c in cv_frame})
    out = CVFilter._inner_function(ds, target_cols=list(cv_frame))
    assert not any(_is_info(out, c) for c in cv_frame)


def test_cv_filter_execute_skips_non_numeric() -> None:
    df = pd.DataFrame({"num": [1.0, 1.0, 1.0, 1.0], "cat": ["a", "b", "a", "b"]})
    roles = {"num": FeatureRole(), "cat": FeatureRole()}
    out = CVFilter(lower_bound=1e-3).execute(make_ed(df, roles))
    assert _is_info(out.ds, "num")
    assert not _is_info(out.ds, "cat")


@pytest.mark.xfail(
    strict=True,
    reason="Issue: bound check uses truthiness (`if lower_bound and ...`), so a bound "
    "of 0.0 is silently ignored",
)
def test_cv_filter_zero_upper_bound_is_applied(cv_frame) -> None:
    ds = make_ds(cv_frame, {c: FeatureRole() for c in cv_frame})
    out = CVFilter._inner_function(ds, target_cols=["mid"], upper_bound=0.0)
    assert _is_info(out, "mid")


# ---------------------------------------------------------------------------
# CorrFilter
# ---------------------------------------------------------------------------
@pytest.fixture
def corr_frame() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 200)
    return pd.DataFrame(
        {
            "a": a,
            "a_copy": a * 2 + 1,
            "indep": rng.normal(0, 1, 200),
            "y": rng.normal(0, 1, 200),
        }
    )


def _corr_roles():
    return {"a": FeatureRole(), "a_copy": FeatureRole(), "indep": FeatureRole(), "y": TargetRole()}


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: CorrFilter passes method= to Dataset.corr, which does not accept it",
)
def test_corr_filter_drops_highly_correlated_feature(corr_frame) -> None:
    ds = make_ds(corr_frame, _corr_roles())
    out = CorrFilter._inner_function(
        ds,
        target_cols=["a", "a_copy", "indep"],
        corr_space_cols=["a", "a_copy", "indep", "y"],
        threshold=0.8,
    )
    dropped = [c for c in ["a", "a_copy", "indep"] if _is_info(out, c)]
    assert len(dropped) >= 1
    assert "indep" not in dropped
    assert not _is_info(out, "y")


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: CorrFilter passes method= to Dataset.corr, which does not accept it",
)
def test_corr_filter_high_threshold_keeps_all(corr_frame) -> None:
    ds = make_ds(corr_frame, _corr_roles())
    out = CorrFilter._inner_function(
        ds,
        target_cols=["a", "a_copy", "indep"],
        corr_space_cols=["a", "a_copy", "indep", "y"],
        threshold=1.0,
    )
    assert not any(_is_info(out, c) for c in ["a", "a_copy", "indep"])


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: CorrFilter passes method= to Dataset.corr, which does not accept it",
)
def test_corr_filter_execute(corr_frame) -> None:
    out = CorrFilter(threshold=0.8).execute(make_ed(corr_frame, _corr_roles()))
    info_cols = [c for c in corr_frame if _is_info(out.ds, c)]
    assert set(info_cols) <= {"a", "a_copy"}
    assert len(info_cols) >= 1


@pytest.mark.xfail(
    strict=True,
    reason="Issue (currently masked by the Dataset.corr(method=) TypeError): "
    "`data.roles[column] in corr_target_cols` compares a role object with "
    "column names, so the 'cv' drop_policy tie-break never runs; the target is always dropped",
)
def test_corr_filter_cv_policy_drops_lower_cv_column() -> None:
    rng = np.random.RandomState(0)
    base = rng.normal(0, 1, 300)
    df = pd.DataFrame({"hi_cv": base + 0.1, "lo_cv": base * 0.01 + 100.0})
    ds = make_ds(df, {"hi_cv": FeatureRole(), "lo_cv": FeatureRole()})
    out = CorrFilter._inner_function(
        ds, target_cols=["hi_cv", "lo_cv"], corr_space_cols=["hi_cv", "lo_cv"], threshold=0.9
    )
    assert _is_info(out, "lo_cv")
    assert not _is_info(out, "hi_cv")


# ---------------------------------------------------------------------------
# OutliersFilter
# ---------------------------------------------------------------------------
@pytest.fixture
def outlier_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": list(range(1, 101)), "y": list(range(100, 0, -1))}).astype(float)


def test_outliers_filter_default_percentiles_drop_nothing(outlier_frame) -> None:
    ds = make_ds(outlier_frame, {c: FeatureRole() for c in outlier_frame})
    out = OutliersFilter._inner_function(ds, target_cols=["x"])
    assert len(out) == len(outlier_frame)


def test_outliers_filter_upper_percentile_drops_tail(outlier_frame) -> None:
    ds = make_ds(outlier_frame, {c: FeatureRole() for c in outlier_frame})
    out = OutliersFilter._inner_function(ds, target_cols=["x"], upper_percentile=0.9)
    kept = to_pandas(out)["x"]
    assert kept.max() <= np.quantile(outlier_frame.x, 0.9)
    assert len(out) == 90


def test_outliers_filter_two_sided(outlier_frame) -> None:
    ds = make_ds(outlier_frame, {c: FeatureRole() for c in outlier_frame})
    out = OutliersFilter._inner_function(
        ds, target_cols=["x"], lower_percentile=0.1, upper_percentile=0.9
    )
    kept = to_pandas(out)["x"]
    assert kept.min() >= np.quantile(outlier_frame.x, 0.1)
    assert kept.max() <= np.quantile(outlier_frame.x, 0.9)


def test_outliers_filter_multiple_columns_drops_union(outlier_frame) -> None:
    ds = make_ds(outlier_frame, {c: FeatureRole() for c in outlier_frame})
    out = OutliersFilter._inner_function(ds, target_cols=["x", "y"], upper_percentile=0.9)
    # x>q90 are rows 91..100; y>q90 are the first 10 rows -> 20 distinct rows dropped
    assert len(out) == 80


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="Issue: OutliersFilter.execute assigns read-only ExperimentData.additional_fields",
)
def test_outliers_filter_execute_filters_additional_fields(outlier_frame) -> None:
    ed = make_ed(outlier_frame, {c: FeatureRole() for c in outlier_frame})
    out = OutliersFilter(target_roles=FeatureRole(), upper_percentile=0.9).execute(ed)
    assert len(out.ds) < len(outlier_frame)
