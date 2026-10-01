"""Tests for NaFiller and NaDropper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import FeatureRole, TargetRole
from hypex.transformers import NaDropper, NaFiller

from ._utils import make_ds, make_ed, to_pandas


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [np.nan, 2.0, 3.0, np.nan, 5.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


# ---------------------------------------------------------------------------
# NaDropper
# ---------------------------------------------------------------------------
def test_dropper_how_any(frame, backend, spark_session) -> None:
    ds = make_ds(frame, {c: TargetRole() for c in frame}, backend, spark_session)
    out = NaDropper._inner_function(ds, target_cols=["a", "b"], how="any")
    assert sorted(to_pandas(out)["c"]) == [3.0, 5.0]


def test_dropper_how_all(frame, backend, spark_session) -> None:
    ds = make_ds(frame, {c: TargetRole() for c in frame}, backend, spark_session)
    out = NaDropper._inner_function(ds, target_cols=["a", "b"], how="all")
    assert len(out) == 4  # only row 3 (both NaN) is dropped


def test_dropper_only_considers_target_columns(frame) -> None:
    ds = make_ds(frame, {c: TargetRole() for c in frame})
    out = NaDropper._inner_function(ds, target_cols=["a"])
    assert to_pandas(out)["a"].notna().all()
    assert len(out) == 3
    assert to_pandas(out)["b"].isna().sum() == 1  # NaN in untouched column kept


def test_dropper_empty_targets_is_noop(frame) -> None:
    ds = make_ds(frame, {c: TargetRole() for c in frame})
    assert len(NaDropper._inner_function(ds, target_cols=[])) == len(frame)


def test_dropper_execute_uses_target_role(frame) -> None:
    roles = {"a": TargetRole(), "b": FeatureRole(), "c": FeatureRole()}
    out = NaDropper().execute(make_ed(frame, roles))
    assert len(out.ds) == 3  # only "a" considered


def test_dropper_execute_no_target_columns_returns_data(frame) -> None:
    roles = {c: FeatureRole() for c in frame}
    ed = make_ed(frame, roles)
    assert len(NaDropper().execute(ed).ds) == len(frame)


def test_dropper_preserves_roles(frame) -> None:
    roles = {"a": TargetRole(), "b": FeatureRole(), "c": FeatureRole()}
    out = NaDropper().execute(make_ed(frame, roles))
    assert isinstance(out.ds.roles["a"], TargetRole)
    assert isinstance(out.ds.roles["b"], FeatureRole)


# ---------------------------------------------------------------------------
# NaFiller
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: Dataset.__setitem__ iterates a DataFrame (column names) when type-checking "
    "a Dataset value, so NaFiller always raises 'Value type does not match'",
)
def test_filler_scalar_value(frame) -> None:
    ds = make_ds(frame, {c: FeatureRole() for c in frame})
    out = NaFiller._inner_function(ds, target_cols=["a", "b"], values=0.0)
    result = to_pandas(out)
    assert result["a"].tolist() == [1.0, 0.0, 3.0, 0.0, 5.0]
    assert result["b"].tolist() == [0.0, 2.0, 3.0, 0.0, 5.0]


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: Dataset.__setitem__ iterates a DataFrame (column names) when type-checking "
    "a Dataset value, so NaFiller always raises 'Value type does not match'",
)
def test_filler_dict_values_per_column(frame) -> None:
    ds = make_ds(frame, {c: FeatureRole() for c in frame})
    out = NaFiller._inner_function(ds, target_cols=["a", "b"], values={"a": -1.0, "b": -2.0})
    result = to_pandas(out)
    assert result["a"].tolist() == [1.0, -1.0, 3.0, -1.0, 5.0]
    assert result["b"].tolist() == [-2.0, 2.0, 3.0, -2.0, 5.0]


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: Dataset.__setitem__ iterates a DataFrame (column names) when type-checking "
    "a Dataset value, so NaFiller always raises 'Value type does not match'",
)
@pytest.mark.parametrize(
    "method,expected",
    [
        ("ffill", [1.0, 1.0, 3.0, 3.0, 5.0]),
        ("bfill", [1.0, 3.0, 3.0, 5.0, 5.0]),
    ],
)
def test_filler_method(frame, method, expected) -> None:
    ds = make_ds(frame, {c: FeatureRole() for c in frame})
    out = NaFiller._inner_function(ds, target_cols=["a"], method=method)
    assert to_pandas(out)["a"].tolist() == expected


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: Dataset.__setitem__ iterates a DataFrame (column names) when type-checking "
    "a Dataset value, so NaFiller always raises 'Value type does not match'",
)
def test_filler_leaves_other_columns_untouched(frame) -> None:
    ds = make_ds(frame, {c: FeatureRole() for c in frame})
    out = NaFiller._inner_function(ds, target_cols=["a"], values=0.0)
    assert to_pandas(out)["b"].isna().sum() == 2


def test_filler_without_value_or_method_raises(frame) -> None:
    ds = make_ds(frame, {c: FeatureRole() for c in frame})
    with pytest.raises(ValueError, match="Value or filling method"):
        NaFiller._inner_function(ds, target_cols=["a"])


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Issue: Dataset.__setitem__ iterates a DataFrame (column names) when type-checking "
    "a Dataset value, so NaFiller always raises 'Value type does not match'",
)
def test_filler_execute_defaults_to_feature_role(frame) -> None:
    roles = {"a": FeatureRole(), "b": TargetRole(), "c": FeatureRole()}
    out = NaFiller(values=0.0).execute(make_ed(frame, roles))
    result = to_pandas(out.ds)
    assert result["a"].isna().sum() == 0
    assert result["b"].isna().sum() == 2  # TargetRole not filled


def test_filler_default_target_roles_is_feature_role() -> None:
    assert isinstance(NaFiller(values=1).target_roles, FeatureRole)
